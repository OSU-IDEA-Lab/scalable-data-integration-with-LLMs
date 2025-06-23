import math
from dataclasses import replace

from AnalysisUtils import Experiment
from sqlglot import expressions as sqlglot_expression, parse

from AnalysisUtils.Experiment import Query, Results, OverlapResults
from DatabaseUtils.DatabaseManager import DatabaseManager, QueryStatus
from DatabaseUtils.MappingLoader import load_gt_clusters
from DatabaseUtils.SchemaLoader import load_schema

SOURCE_NAMESPACE = "source"
GOLD_NAMESPACE = "gold_target"
PREDICTED_NAMESPACE = "target"

class Evaluator:
    def __init__(self, prefetch_dataset: str, prefetch_source: str, prefetch_target: str):

        self._schema_to_tables = {}

        self._initialized = False
        self._db_manager = None
        self._source_schema = None
        self._target_schema = None
        self._source_table_names = None
        self._target_table_names = None
        self._gold_mapping = None

        self._select_to_positive_count = {}

        # Load source schema and import eval data.
        self._source_schema = load_schema(dataset_name=prefetch_dataset, schema_name=prefetch_source)
        self._target_schema = load_schema(dataset_name=prefetch_dataset, schema_name=prefetch_target)
        self._source_table_names = [rel.name for rel in self._source_schema.relations.values()]
        self._target_table_names = [rel.name for rel in self._target_schema.relations.values()]

        # Load gold mapping data
        self._gold_mapping = load_gt_clusters(prefetch_dataset, prefetch_source, prefetch_target)
        for m in self._gold_mapping:
            m["queries"] = parse(m["gold_sql"].format(ns_src=SOURCE_NAMESPACE, ns_tgt=GOLD_NAMESPACE))

        self._gold_rule_by_cluster = {}
        for m in self._gold_mapping:
            self._gold_rule_by_cluster[frozenset(m["source_relations"])] = {
                frozenset(m["target_relations"]): m
            }

        self._db_manager = DatabaseManager()

    def init_databases(self, filename_prefix: str = None):
        self._db_manager.attach_database(self._source_schema, SOURCE_NAMESPACE,
                                         with_constraints=True, filename_prefix=filename_prefix)
        self._db_manager.attach_database(self._target_schema, GOLD_NAMESPACE,
                                         with_constraints=False, filename_prefix=filename_prefix)
        self._db_manager.attach_database(self._target_schema, PREDICTED_NAMESPACE,
                                         with_constraints=False, filename_prefix=filename_prefix)
        self._initialized = True

    def _mark_queries_to_ignore(self, experiment: Experiment):
        statement_counts = {}

        for prompt in experiment.prompt_details:
            for query in prompt.get_valid_queries():

                if type(query.ast).__name__ not in statement_counts:
                    statement_counts[type(query.ast).__name__] = 0
                statement_counts[type(query.ast).__name__] += 1

                # Ignore CREATE SCHEMA statements
                if not(issubclass(type(query.ast), sqlglot_expression.DDL)
                       or issubclass(type(query.ast), sqlglot_expression.DML)
                       or isinstance(query.ast, sqlglot_expression.Drop)):
                    query.ignore_reason = f"Stray {type(query.ast)} statement ignored"
                    continue
                elif isinstance(query.ast, sqlglot_expression.Create):
                    if hasattr(query.ast, "kind") and query.ast.kind == "SCHEMA":
                        query.ignore_reason = f"Trying to CREATE SCHEMA. Statement ignored"
                        continue
                    table_name = query.ast.find(sqlglot_expression.Table).this.this
                    if table_name in self._target_table_names:
                        query.ignore_reason = f"Trying to CREATE {table_name}. Statement ignored"
                        continue
                elif isinstance(query.ast, sqlglot_expression.Drop):
                    table_name = query.ast.find(sqlglot_expression.Table).this.this
                    if table_name in self._target_table_names:
                        query.ignore_reason = f"Trying to DROP {table_name}. Statement ignored"
                        continue

    def _append_schema_namespaces(self, experiment: Experiment):
        """
            Given experiment, will append namespace to any table which does not already have a namespace AND whose
            name is in table_names. Produces "{namespace}.table" in final sql.
        """

        for prompt in experiment.prompt_details:
            for query in prompt.get_valid_queries():

                for table_node in query.ast.find_all(sqlglot_expression.Table):
                    if table_node.args["db"] is None:
                        if table_node.name in self._source_table_names and table_node.name in self._target_table_names:
                            continue
                        elif table_node.name in self._source_table_names:
                            table_node.args["db"] = sqlglot_expression.Identifier(this=SOURCE_NAMESPACE, quoted=False)
                        elif table_node.name in self._target_table_names:
                            table_node.args["db"] = sqlglot_expression.Identifier(this=PREDICTED_NAMESPACE, quoted=False)

    def _add_mapping_id(self, experiment: Experiment):
        for prompt in experiment.prompt_details:
            src_key = frozenset(prompt.source_cluster)
            tgt_key = frozenset(prompt.target_cluster)
            prompt.gold_rule_id = self._gold_rule_by_cluster[src_key][tgt_key]["id"]

    def prepare_queries(self, experiment: Experiment):

        self._mark_queries_to_ignore(experiment)
        self._append_schema_namespaces(experiment)

        if experiment.prompt_config["clusterings"] == "ground_truth":
            self._add_mapping_id(experiment)

    def _reset_databases(self):
        assert self._initialized, "Must call init_databases"
        self._db_manager.reset_databases()
        self._db_manager.import_data(SOURCE_NAMESPACE, use_eval_data=True)

    def _get_a_except_b_sql(self, a_sql: str, b_sql: str):
        return "{A_SQL} {SET_OP} {B_SQL}".format(A_SQL=a_sql, B_SQL=b_sql, SET_OP="EXCEPT")

    def _get_a_intersect_b_sql(self, a_sql: str, b_sql: str):
        return "{A_SQL} {SET_OP} {B_SQL}".format(A_SQL=a_sql, B_SQL=b_sql, SET_OP="INTERSECT")

    def _get_overlap(self, select_predicted_sql: str, select_gold_sql: str) -> OverlapResults:

        overlap_results = OverlapResults()

        #   false_negatives = gold_table - predicted_table
        status, fn_results = self._db_manager.execute_sql(
            self._get_a_except_b_sql(a_sql=select_gold_sql, b_sql=select_predicted_sql)
        )

        if status != QueryStatus.OK:
            assert status == QueryStatus.TIMEOUT, \
                f"Expected timeout for failed overlap query, got ({status}: {fn_results})"

            if select_gold_sql not in self._select_to_positive_count:
                _, results = self._db_manager.execute_sql(select_gold_sql)
                self._select_to_positive_count[select_gold_sql] = len(results.rows)

            overlap_results.overlap_count = replace(overlap_results.overlap_count,
                                                    **{"FN": self._select_to_positive_count[select_gold_sql],
                                                       "FP": math.inf})

        else:

            #   false_positives = predicted_table - gold_table
            _, fp_results = self._db_manager.execute_sql(
                self._get_a_except_b_sql(a_sql=select_predicted_sql, b_sql=select_gold_sql)
            )

            #   true_positives = predicted_table INTERSECT gold_table
            _, tp_results = self._db_manager.execute_sql(
                self._get_a_intersect_b_sql(a_sql=select_predicted_sql, b_sql=select_gold_sql)
            )

            #   column-level differences
            # cols_in_fn_rows = fn_results.get_col_multiset()
            # cols_in_fp_rows = fp_results.get_col_multiset()
            # fn_cols = {n: cols_in_fn_rows[n] - cols_in_fp_rows[n] for n in cols_in_fn_rows.keys()}
            # fp_cols = {n: cols_in_fp_rows[n] - cols_in_fn_rows[n] for n in cols_in_fp_rows.keys()}

            overlap_results.overlap_count = replace(overlap_results.overlap_count,
                                                    **{"FN": len(fn_results.rows),
                                                   "FP": len(fp_results.rows),
                                                   "TP": len(tp_results.rows)})
            overlap_results.overlap_data = replace(overlap_results.overlap_data,
                                                   **{"FN_rows": fn_results.rows[:3],
                                                  "FP_rows": fp_results.rows[:3]})
                                                  # "FN_cols": fn_cols, "FP_cols": fp_cols})

        return overlap_results

    def _evaluate_tables(self, tables_to_evaluate: list[str] = None) -> dict[str, OverlapResults]:

        table_overlap: dict[str, OverlapResults] = {}

        if tables_to_evaluate is None:
            relations = self._target_schema.relations.values()
        else:
            relations = [rel for rel in self._target_schema.relations.values() if rel.name in tables_to_evaluate]

        # For each table, generate a SELECT statement over that tables columns MINUS the primary keys
        for rel in relations:

            # Get all meaningful attributes (columns). Skip comparing rows from this relation if there are no meaningful cols
            # this would happen, for instance, on intersection tables only containing FKs to arbitrary PK columns
            meaningful_attrs = [attr for attr in rel.attributes if attr.meaningful]
            if len(meaningful_attrs) == 0:
                continue

            # Generate select statements
            select_clause = ', '.join([f"{a.name} AS \"{rel.name}.{a.name}\"" for a in meaningful_attrs])
            select_sql = f"SELECT {select_clause} FROM {{ns}}.{rel.name}"
            select_predicted_sql = select_sql.format(ns=PREDICTED_NAMESPACE)
            select_gold_sql = select_sql.format(ns=GOLD_NAMESPACE)

            overlap_results = self._get_overlap(select_predicted_sql, select_gold_sql)

            if (overlap_results.overlap_count.fn + overlap_results.overlap_count.fp
                    + overlap_results.overlap_count.tp == 0):
                # warnings.warn(f"Empty table '{rel.name}' matches ground truth (also empty). Removing from evaluation. "
                #       "May want to reconsider this once we use something other than ground_truth for clusters.")
                continue

            table_overlap[rel.name] = overlap_results

        return table_overlap

    def _evaluate_joins(self, gt_clusters: list[dict]) -> dict[str, OverlapResults]:

        join_overlap: dict[str, OverlapResults] = {}

        # For each table, generate a SELECT statement over that tables columns MINUS the primary keys
        for rule in gt_clusters:

            # Don't test Logical Relations containing no joins (1 table), these are already tested via Table Overlap
            if len(rule["target_relations"]) == 1:
                continue

            # Generate select statements
            select_sql = rule["join_overlap_sql"]
            select_gold_sql = select_sql.format(ns=GOLD_NAMESPACE)
            select_predicted_sql = select_sql.format(ns=PREDICTED_NAMESPACE)

            overlap_results = self._get_overlap(select_predicted_sql, select_gold_sql)

            join_overlap[rule["id"]] = overlap_results

        return join_overlap

    def _evaluate(self, source_cluster_tables: list[str] = None,
                  target_cluster_tables: list[str] = None):

        if source_cluster_tables is None and target_cluster_tables is None:
            gold_rules_to_execute = self._gold_mapping
            tables_to_evaluate = self._target_table_names
        else:
            assert source_cluster_tables is not None and target_cluster_tables is not None, \
                "Must specify both or neither"
            gold_rules_to_execute = [self._gold_rule_by_cluster
                                     [frozenset(source_cluster_tables)]
                                     [frozenset(target_cluster_tables)]]
            tables_to_evaluate = target_cluster_tables

        # Apply gold mapping
        for gold_rule in gold_rules_to_execute:
            for query in gold_rule["queries"]:
                self._db_manager.execute_sql(query.sql(dialect="sqlite"))

        # Test for constraint violations

        # Evaluate table overlap
        table_overlap = self._evaluate_tables(tables_to_evaluate)

        # Evaluate join overlap
        join_overlap = self._evaluate_joins(gold_rules_to_execute)

        return table_overlap, join_overlap

    def _apply_queries(self, queries: list[Query]):
        for query in queries:
            status, payload = self._db_manager.execute_sql(query.ast.sql(dialect="sqlite"))

            query.execution_status = status
            if status != QueryStatus.OK:
                query.execution_error = payload

    def evaluate_rules(self, experiment: Experiment):
        assert self._initialized, "Must call init_databases prior to evaluating"
        assert experiment.prompt_config["clusterings"] == "ground_truth", \
            "Rule evaluation only supported for ground_truth clusterings."

        for prompt in experiment.prompt_details:

            self._reset_databases()

            # Apply predicted rule
            self._apply_queries(prompt.get_valid_queries())

            table_overlap, join_overlap = (self._evaluate(prompt.source_cluster, prompt.target_cluster))

            results = Results()
            results.table_overlap = table_overlap
            results.join_overlap = join_overlap
            experiment.rule_results_by_id[prompt.gold_rule_id] = results

    def evaluate_mapping(self, experiment: Experiment):
        assert self._initialized, "Must call init_databases prior to evaluating"

        self._reset_databases()

        # Apply predicted mapping
        for prompt in experiment.prompt_details:
            self._apply_queries(prompt.get_valid_queries())

        # Evaluate
        table_overlap, join_overlap = self._evaluate()

        experiment.mapping_results.table_overlap = table_overlap
        experiment.mapping_results.join_overlap = join_overlap