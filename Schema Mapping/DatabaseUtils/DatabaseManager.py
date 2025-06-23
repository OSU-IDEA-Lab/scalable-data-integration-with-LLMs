import os
import sqlite3
import time
from dataclasses import dataclass

from adodbapi import OperationalError

from DatabaseUtils.ConfigReader import get_config
from DatabaseUtils.DataLoader import get_sql_for_insert
from DatabaseUtils.Schema import Schema
import pandas as pd
from enum import Enum

import threading
import queue

from collections import Counter

TEMP_PATH = get_config()["paths"]["temp_dbs"]

# What information is included in the prompt
class Constraint(Enum):

    # Attribute-level
    DATA_TYPE = "data type"
    DATA_SIZE = "data size"
    NULLABLE = "nullable"

    # Relation-level
    UNIQUE = "unique"

class QueryStatus(Enum):
    OK = "OK"
    ERROR = "Error"
    TIMEOUT = "Timeout"

class QueryResults:

    def __init__(self, rows: list[tuple], attribute_names: list[str]):
        self.rows: list[tuple] = rows
        self.attribute_names: list[str] = attribute_names

    def get_col_multiset(self) -> dict[str, Counter[str]]:

        # Build multiset (Counter) over values for each attribute (column)
        vals_by_col = {n: [] for n in self.attribute_names}
        for row in self.rows:
            for idx, n in enumerate(self.attribute_names):
                vals_by_col[n].append(row[idx])
        return {n: Counter(vals_by_col[n]) for n in vals_by_col}

class DatabaseManager(object):

    def __init__(self):
        self.connection: sqlite3.Connection = None
        self.namespace_to_file: dict[str, str] = {}
        self.namespace_to_schema: dict[str, Schema] = {}
        self.namespace_to_constraints: dict[str, bool] = {}
        self._data_import_cache: dict[str, str] = {}

        if not os.path.exists(TEMP_PATH):
            os.makedirs(TEMP_PATH)

    def __del__(self):
        self._close_connection()

    def _reset(self):
        for schema_namespace in self.namespace_to_schema:
            database_schema = self.namespace_to_schema[schema_namespace]
            db_file_name = self.namespace_to_file[schema_namespace]
            with_constraints = self.namespace_to_constraints[schema_namespace]

            self.connection.execute(f"ATTACH DATABASE '{TEMP_PATH}/{db_file_name}' AS {schema_namespace}")
            self.connection.executescript(database_schema.as_sql(schema_namespace, with_constraints))

    def reset_databases(self):
        self._close_connection()
        self._open_connection()
        self._reset()

    def _open_connection(self):
        self.connection = sqlite3.connect(':memory:', check_same_thread=False)

        self._pk_map: dict[str, dict[str, dict[str, int]]] = {}

        def _map_pk(tbl, attr, val):
            if tbl not in self._pk_map:
                self._pk_map[tbl] = {attr: {val: 0}}
            elif attr not in self._pk_map[tbl]:
                self._pk_map[tbl][attr] = {val: 0}
            elif val not in self._pk_map[tbl][attr]:
                self._pk_map[tbl][attr][val] = len(self._pk_map[tbl][attr])
            return self._pk_map[tbl][attr][val]

        self.connection.create_function("map_pk", 3, _map_pk)

    def attach_database(self, database_schema: Schema, schema_namespace: str,
                        with_constraints: bool, filename_prefix: str = None):

        if self.connection is None:
            self._open_connection()

        assert len(schema_namespace) > 0, f"Namespace must not be an empty string."
        assert schema_namespace not in self.namespace_to_schema, f"Namespace '{schema_namespace}' already in use."

        # Track details for easy resets
        self.namespace_to_schema[schema_namespace] = database_schema
        self.namespace_to_constraints[schema_namespace] = with_constraints
        db_file_name = f"{schema_namespace}-{time.time_ns()}.db"
        if filename_prefix is not None:
            db_file_name = f"{filename_prefix}-{db_file_name}"
        self.namespace_to_file[schema_namespace] = db_file_name

        # Spin up the database
        self.connection.execute(f"ATTACH DATABASE '{TEMP_PATH}/{db_file_name}' AS {schema_namespace}")
        self.connection.executescript(database_schema.as_sql(schema_namespace, with_constraints))

    def run_sql(self, sql: str):
        self.connection.executescript(sql)

    def _remove_temp_file(self, db_file: str):
        try:
            os.remove(f"{TEMP_PATH}/{db_file}")
        except OSError as e:
            print(f"An error occurred while trying to cleanup {TEMP_PATH}/{db_file}:\n {e}")

    def _close_connection(self):
        self.connection.close()
        self.connection = None
        for db_file in self.namespace_to_file.values():
            self._remove_temp_file(db_file)

    def import_data(self, namespace: str, use_eval_data: bool = False):
        assert namespace in self.namespace_to_schema, (f"Namespace '{namespace}' not attached. "
                                                       f"Must be one of {self.namespace_to_schema.keys()}")

        if namespace not in self._data_import_cache:
            self._data_import_cache[namespace] = "\n".join(get_sql_for_insert(self.namespace_to_schema[namespace],
                                                                              namespace=namespace,
                                                                              use_eval_data=use_eval_data))

        # Full script
        self.connection.executescript(self._data_import_cache[namespace])

    def get_config(self) -> dict:
        return {namespace: {"dataset": schema.dataset_name, "database": schema.database_name}
                for namespace, schema in self.namespace_to_schema.items()}

    def get_table_constraint_count(self, namespace: str):
        schema = self.namespace_to_schema[namespace]
        constraint_total_by_table = {}
        for relation in schema.relations.values():
            constraint_total_by_table[relation.name] = (len([attr for attr in relation.attributes
                                                             if not attr.nullable]) +
                                                        len([attr for attr in relation.attributes if
                                                             (not attr.get_supertype() == "TEXT") or
                                                             (attr.data_size is not None)]) +
                                                        len(relation.unique_constraints) +
                                                        # +2 for PK (UNIQUE and NOT NULL)
                                                        2)  # + 1 for the mandatory primary key
        return constraint_total_by_table

    def get_table_constraint_violations(self, namespace: str):
        schema = self.namespace_to_schema[namespace]
        violations_by_table = {}
        for relation in schema.relations.values():
            violations_by_table[relation.name] = {}
            this_relation = violations_by_table[relation.name]

            for c in Constraint:
                this_relation[c] = Counter()

            table_data_df = self.get_table_data(namespace, relation.name)
            for attr in relation.attributes:

                for value in table_data_df[attr.name]:

                    violation_key = (relation.name, attr.name)

                    # Test nullability
                    if value is None:
                        if not attr.nullable:
                            this_relation[Constraint.NULLABLE].update([violation_key])
                        continue

                    # Data type and size adherence
                    if attr.get_supertype() == "INTEGER":
                        if not value.isdecimal():
                            this_relation[Constraint.DATA_TYPE].update([violation_key])
                    elif attr.get_supertype() == "TEXT":

                        if attr.data_size is not None and len(value) > attr.data_size:
                            this_relation[Constraint.DATA_SIZE].update([violation_key])

                # Uniqueness constraints
                for unique_attrs in relation.unique_constraints + [relation.primary_key]:
                    if table_data_df[unique_attrs].duplicated().sum() > 0:
                        this_relation[Constraint.UNIQUE].update([(relation.name, tuple(unique_attrs))])

        return violations_by_table

    def execute_sql(self, query: str, timeout: int = 30) -> tuple[QueryStatus, QueryResults]:
        result_queue = queue.Queue()

        def _run(in_conn: sqlite3.Connection):
            try:
                cursor = in_conn.cursor()
                cursor.execute(query)

                attributes = []
                if cursor.description is not None:
                    attributes = [d[0] for d in cursor.description]

                results = QueryResults(rows=cursor.fetchall(), attribute_names=attributes)
                result_queue.put((QueryStatus.OK, results))
            except (sqlite3.OperationalError, sqlite3.IntegrityError) as ex:
                result_queue.put((QueryStatus.ERROR, str(ex)))

        # # Make a copy of the connection for this thread
        # conn = sqlite3.connect(self.connection.database)

        thread = threading.Thread(target=_run, args=(self.connection,))
        thread.start()

        thread.join(timeout)

        if thread.is_alive():
            try:
                self.connection.interrupt()  # Interrupt the long-running query
            except Exception as e:
                return QueryStatus.ERROR, f"{str(e)}"
            return QueryStatus.TIMEOUT, f"Query timed out after {timeout} seconds."
        else:
            status, payload = result_queue.get()
            if status == QueryStatus.OK:
                return status, payload
            else:
                return QueryStatus.ERROR, payload

    def get_table_data(self, namespace: str, table_name: str) -> pd.DataFrame:
        return pd.read_sql_query(f"SELECT * FROM {namespace}.{table_name}", self.connection)