import csv
import json
import os
import warnings
from typing import Union
import bibtexparser.middlewares as m
from DatabaseUtils.ConfigReader import get_config
import bibtexparser as bib

from DatabaseUtils.Schema import Schema, Relation, Attribute
from DatabaseUtils.enums import Dataset

CONFIG = get_config()

def _cast_value(value: str, type: str):

    # if value.isdecimal():
    #     return value
    # else:
    #     value = value.replace("'", "''")
    #     return f"'{value}'"

    if value is None:
        return "NULL"
    elif isinstance(value, str):
        value = value.replace("\'", "\'\'")
        value = f"'{value}'"

    return f"CAST({value} AS {type})"

class InsertBuilder(object):
    def __init__(self):
        self._insert_attributes = {}
        self._insert_rows = {}

    def get_insert_sql(self, fields_dict: Union[dict, None], target_relation: str, attribute_literals: dict[str, str],
                       attr_types: dict[str, dict[str, str]], attribute_mapping: dict[str, list[str]] = {}) -> str:
        attributes = []
        values = []

        # Add literals
        for target_attr, literal in attribute_literals.items():
            attributes.append(target_attr)
            values.append(_cast_value(literal, attr_types[target_relation][target_attr]))

        # Add attributes in fields_dict (entry) based on attribute_mapping
        for target_attr, source_attrs_in_order in attribute_mapping.items():
            attributes.append(target_attr)

            for attr in source_attrs_in_order:
                if attr in fields_dict:
                    try:
                        val = fields_dict[attr].value # If XML node
                    except AttributeError:
                        val = fields_dict[attr] # If basic dict

                    values.append(_cast_value(val, attr_types[target_relation][target_attr]))
                    break
            if not len(attributes) == len(values):
                values.append("NULL")

        if target_relation not in self._insert_rows:
            self._insert_attributes[target_relation] = set()
        self._insert_attributes[target_relation].update(attributes)

        if target_relation not in self._insert_rows:
            self._insert_rows[target_relation] = []
        self._insert_rows[target_relation].append({attr: val for attr, val in zip(attributes, values)})
        # return f"INSERT INTO {target_relation} ({', '.join(attributes)}) VALUES ({', '.join(values)});"

    def get_inserts(self):

        inserts = []

        for table in self._insert_rows:
            attributes = list(self._insert_attributes[table])
            rows_str = []
            for r in self._insert_rows[table]:
                rows_str.append(f"({', '.join([r.get(a, 'NULL') for a in attributes])})")

            inserts.append(f"INSERT INTO {table} ({', '.join(attributes)}) VALUES {', '.join(rows_str)};")

        return inserts

class EHRImports:

    BASE_PATH = f"{CONFIG['paths']['datasets']}/ehr/"

    @classmethod
    def _inserts_for_mimic(cls, schema_namespace: str, attr_types: dict[str, dict[str, str]]) -> list[str]:

        mimic_inserts = []

        if schema_namespace == "source":
            base_dir = f"{EHRImports.BASE_PATH}/mimic/concepts"

            for filename in os.listdir(base_dir):

                if not filename.endswith(".csv"):
                    continue

                with open(os.path.join(base_dir, filename), "r", newline="") as csv_file:
                    concept_reader = csv.reader(csv_file, delimiter=",", quotechar='"')

                    relation_name = f"gcpt_{filename.replace('.csv', '')}"

                    attribute_names = next(concept_reader)
                    all_rows = []

                    for row_num, this_row in enumerate(concept_reader):
                        if row_num == 0:

                            relation = Relation(relation_name)

                            for col_num, value in enumerate(this_row):
                                if value.isdecimal():
                                    type = "int4"
                                else:
                                    type = "varchar"
                                relation.add_attribute(Attribute(name=attribute_names[col_num], data_type=type))

                            mimic_inserts.append(relation.as_sql(schema_namespace))

                        if sum([len(value) for value in this_row]) > 0:
                            all_rows.append(this_row)

                    insert_rows_str = ", ".join([f"({', '.join([_quote_value(value) for value in this_row])})"
                                                 for this_row in all_rows])
                    mimic_inserts.append(f"INSERT INTO {schema_namespace}.{relation_name} ({', '.join(attribute_names)}) VALUES {insert_rows_str};")


        return mimic_inserts


class AmalgamImports:

    BASE_PATH = f"{CONFIG['paths']['datasets']}/amalgam"

    @classmethod
    def _apply_datatype(cls, value, table_name: str, column_name: str, this_schema: Schema):
        column_info = this_schema.relations[table_name].get_attribute(column_name)
        assert column_info is not None, (f"Column '{column_name}' not found in table {table_name}. "
                                         f"Expected one of {[a.name for a in this_schema.relations[table_name].attributes]}")

        # Check for indications of NULL values
        if value is None or (isinstance(value, str) and value.lower() in ["", "null", "\\n"]):

            if not column_info.nullable:
                warnings.warn(f"{table_name}.{column_name}: not nullable, but value is empty or NULL ({value}).")

            return "NULL"

        data_type = column_info.get_supertype()
        if data_type == "TEXT":

            value = str(value)

            # check length
            if column_info.data_size is not None:
                if len(value) > column_info.data_size:
                    warnings.warn(f"{table_name}.{column_name}: value exceeds length ({value}).")

            return '"{val}"'.format(val=value.replace('"', '""'))
        else:
            return str(value)

    @classmethod
    def _generate_insert_statements_from_jsonl(cls, directory_path, this_schema, schema_namespace) \
            -> tuple[list[str], set[str]]:
        insert_statements = []
        tables_found = set()

        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.jsonl'):
                table_name = os.path.splitext(filename)[0]

                assert table_name in this_schema.relations, \
                    f"Table {table_name} not found in schema {this_schema.database_name}"
                assert table_name not in tables_found, f"Table {table_name} found twice in {directory_path}"

                tables_found.add(table_name)
                file_path = os.path.join(directory_path, filename)

                unique_rows = set()

                with open(file_path, 'r', encoding='utf-8') as f:
                    rows = []
                    for line in f:
                        if line.strip():  # skip blank lines

                            if line in unique_rows:
                                warnings.warn(f"Duplicate row in {file_path}: {line}")
                                continue
                            else:
                                unique_rows.add(line)

                            data = json.loads(line)
                            columns = list(data.keys())
                            formatted_values = []
                            for col in columns:
                                formatted_values.append(cls._apply_datatype(data[col], table_name, col, this_schema))
                            rows.append(f"({', '.join(formatted_values)})")

                    if rows:
                        insert = f'INSERT INTO {schema_namespace}.{table_name} ({", ".join(columns)}) VALUES\n' + ',\n'.join(
                            rows) + ';'
                        insert_statements.append(insert)

        return insert_statements, tables_found

    #################################################################################
    # Map things to a1
    #################################################################################
    @classmethod
    def _inserts_for_a1_dep(cls, schema_namespace: str, attr_types: dict[str, dict[str, str]],
                        use_eval_data: bool) -> list[str]:

        insert_builder = InsertBuilder()

        if use_eval_data:
            import_file = "a1/eval_data"
            import_dir = os.path.join(AmalgamImports.BASE_PATH, import_file)
            return cls._generate_insert_statements_from_jsonl(import_dir, schema_namespace)
        else:
            import_file = "a1/data.bib"

        file_path = os.path.join(AmalgamImports.BASE_PATH, import_file)
        lib = bib.parse_file(file_path, append_middleware=[m.names.SeparateCoAuthors(),
                                                             m.names.SplitNameParts()])

        # Autoincrement ID counters
        auth_id_counter = 0
        type_id_counter = 0

        type_to_relation_a1 = {"article": {"relation": "Article", "pk": "articleID", "jointable": "ArticlePublished"},
                               "book": {"relation": "Book", "pk": "bookID", "jointable": "BookPublished"},
                               "inproceedings": {"relation": "InProceedings", "pk": "inprocID",
                                                 "jointable": "InprocPublished"},
                               "misc": {"relation": "Misc", "pk": "miscID", "jointable": "MiscPublished"},
                               "manual": {"relation": "Manual", "pk": "manID", "jointable": "ManualPublished"},
                               "incollection": {"relation": "InCollection", "pk": "collID",
                                                "jointable": "InCollPublished"},
                               "techreport": {"relation": "TechReport", "pk": "techID", "jointable": "TechPublished"}}

        for entry in lib.entries:

            try:
                if entry.entry_type not in type_to_relation_a1:
                    # warnings.warn("For namespace (" + str(schema_namespace) + ") skipping {" + str(entry.entry_type)
                    #               + "} (" + str(entry.key) + "). No suitable relation?")
                    continue

                auth_ids = []
                if "author" in entry.fields_dict:
                    for auth in entry.fields_dict["author"].value:
                        insert_builder.get_insert_sql(fields_dict=None, target_relation=f"{schema_namespace}.Author",
                                       attr_types=attr_types,
                                       attribute_literals={"AuthID": auth_id_counter, "name": auth.merge_first_name_first})
                        auth_ids.append(auth_id_counter)
                        auth_id_counter += 1

                attribute_mapping = {
                    "title": ["title"],
                    "year": ["year"],
                    "month": ["month"],
                    "pages": ["pages"],
                    "vol": ["volume"],
                    "num": ["number"],
                    "loc": ["location"],
                    "class": ["category"],
                    "note": ["note"],
                    "annote": ["annote", "annotation"]
                }

                if type_to_relation_a1[entry.entry_type]["relation"] in ["InProceedings", "InCollection"]:
                    attribute_mapping["bktitle"] = ["booktitle"]
                if type_to_relation_a1[entry.entry_type]["relation"] in ["Article"]:
                    attribute_mapping["journal"] = ["journal", "journaltitle"]
                if type_to_relation_a1[entry.entry_type]["relation"] in ["Book"]:
                    attribute_mapping["publisher"] = ["publisher"]
                if type_to_relation_a1[entry.entry_type]["relation"] in ["TechReport"]:
                    attribute_mapping["inst"] = ["institution"]
                if type_to_relation_a1[entry.entry_type]["relation"] in ["Misc"]:
                    attribute_mapping["howpub"] = ["howpublished"]
                    attribute_mapping["confloc"] = ["location"]
                if type_to_relation_a1[entry.entry_type]["relation"] in ["Manual"]:
                    attribute_mapping["org"] = ["organization"]

                # Insert record
                insert_builder.get_insert_sql(fields_dict=entry.fields_dict,
                               target_relation=f"{schema_namespace}.{type_to_relation_a1[entry.entry_type]['relation']}",
                               attribute_literals={type_to_relation_a1[entry.entry_type]["pk"]: type_id_counter},
                               attr_types=attr_types,
                               attribute_mapping=attribute_mapping)

                # Insert references to authors
                for auth_id in auth_ids:
                    insert_builder.get_insert_sql(fields_dict=None,
                                   target_relation=f"{schema_namespace}.{type_to_relation_a1[entry.entry_type]['jointable']}",
                                   attr_types=attr_types,
                                   attribute_literals={type_to_relation_a1[entry.entry_type]["pk"]: type_id_counter,
                                                       "AuthID": auth_id})

                type_id_counter += 1
            except Exception as err:
                print()
                print("------------------------------------------------------------------------------------")
                print(err)
                print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
                print(entry)

        return insert_builder.get_inserts()

    #################################################################################
    # Map things to a2
    #################################################################################
    @classmethod
    def _inserts_for_a2_dep(cls, schema_namespace: str, attr_types: dict[str, dict[str, str]],
                        use_eval_data: bool) -> list[str]:

        insert_builder = InsertBuilder()

        def _normalize_name(name: str):
            return name.title()

        def _add_or_link_row(src_value: str, tgt_name: str, value_to_ID: dict, cite_id: int,
                            relation_name: str, relation_id_name: str, join_table_name: str):
            inserts = []

            src_value = _normalize_name(src_value)

            """
                (Whether the item already exists, either the ID of the existing item or the next largest int)
            """
            if src_value not in value_to_ID:  # Add new name

                if len(value_to_ID) == 0:
                    value_to_ID[src_value] = 0
                else:
                    value_to_ID[src_value] = max(list(value_to_ID.values())) + 1

                insert_builder.get_insert_sql(fields_dict=None, target_relation=f"{schema_namespace}.{relation_name}",
                               attr_types=attr_types,
                               attribute_literals={relation_id_name: value_to_ID[src_value], tgt_name: src_value})

            # Link this record to it
            insert_builder.get_insert_sql(fields_dict=None, target_relation=f"{schema_namespace}.{join_table_name}",
                           attr_types=attr_types,
                           attribute_literals={"citKey": cite_id, relation_id_name: value_to_ID[src_value]})

            return inserts

        if use_eval_data:
            raise NotImplementedError("a2 as source is not supported")
        else:
            import_file = "a2/data.bib"

        file_path = os.path.join(AmalgamImports.BASE_PATH, import_file)
        lib = bib.parse_file(file_path, append_middleware=[m.names.SeparateCoAuthors(),
                                                           m.names.SplitNameParts()])

        publisher_to_ID = {}
        journal_to_ID = {}
        series_to_ID = {}
        booktitle_to_ID = {}
        keyWord_to_ID = {}

        # Autoincrement ID counters
        cite_id_counter = 0

        for entry in lib.entries:

            try:
                insert_builder.get_insert_sql(fields_dict=None, target_relation=f"{schema_namespace}.allBibs",
                               attr_types=attr_types,
                               attribute_literals={"citKey": cite_id_counter})

                my_mapping = [{"src_attr": "isbn", "tgt_rel": "ISBN", "tgt_attr": "isbnNum"},
                              {"src_attr": "volume", "tgt_rel": "volumes", "tgt_attr": "volNum"},
                              {"src_attr": "number", "tgt_rel": "numbers", "tgt_attr": "num"},
                              {"src_attr": "month", "tgt_rel": "months", "tgt_attr": "mon"},
                              {"src_attr": "year", "tgt_rel": "years", "tgt_attr": "yr"},
                              {"src_attr": "pages", "tgt_rel": "pages", "tgt_attr": "pgRange"},
                              {"src_attr": "abstract", "tgt_rel": "abstracts", "tgt_attr": "txt"},
                              {"src_attr": "title", "tgt_rel": "titles", "tgt_attr": "title"},
                              {"src_attr": "address", "tgt_rel": "addresses", "tgt_attr": "address"},
                              {"src_attr": "school", "tgt_rel": "schools", "tgt_attr": "schoolNm"},
                              {"src_attr": "institution", "tgt_rel": "institutions", "tgt_attr": "institNm"},
                              {"src_attr": "entrysubtype", "tgt_rel": "citForm", "tgt_attr": "form"},
                              {"src_attr": "type", "tgt_rel": "types", "tgt_attr": "type"},

                              # Multiple notes
                              {"src_attr": "note", "tgt_rel": "notes", "tgt_attr": "note"},
                              {"src_attr": "annote", "tgt_rel": "notes", "tgt_attr": "note"},
                              {"src_attr": "annotation", "tgt_rel": "notes", "tgt_attr": "note"}]

                for map in my_mapping:
                    if map["src_attr"] in entry.fields_dict:

                        insert_builder.get_insert_sql(
                            fields_dict=entry.fields_dict, target_relation=f"{schema_namespace}.{map['tgt_rel']}",
                            attribute_mapping={map["tgt_attr"]: [map["src_attr"]]},
                            attr_types=attr_types,
                            attribute_literals={"citKey": cite_id_counter})

                # Map authors
                if "author" in entry.fields_dict:
                    for auth in entry.fields_dict["author"].value:
                        insert_builder.get_insert_sql(fields_dict=None, target_relation=f"{schema_namespace}.authors",
                                       attr_types=attr_types,
                                       attribute_literals={"citKey": cite_id_counter, "authNm": auth.merge_first_name_first})

                # map editors
                if "editor" in entry.fields_dict:
                    for auth in entry.fields_dict["editor"].value:
                        insert_builder.get_insert_sql(fields_dict=None, target_relation=f"{schema_namespace}.editors",
                                     attr_types=attr_types,
                                     attribute_literals={"citKey": cite_id_counter, "edNm": auth.merge_first_name_first})

                # Publisher
                if "publisher" in entry.fields_dict:
                    _add_or_link_row(src_value=entry.fields_dict["publisher"].value, tgt_name="pubNm",
                                    value_to_ID=publisher_to_ID,
                                    cite_id=cite_id_counter, relation_name="publisher", relation_id_name="pubID",
                                    join_table_name="citPublisher")

                # Journal
                if "journaltitle" in entry.fields_dict:
                    _add_or_link_row(src_value=entry.fields_dict["journaltitle"].value, tgt_name="jrnlNm",
                                    value_to_ID=journal_to_ID,
                                    cite_id=cite_id_counter, relation_name="journal", relation_id_name="jrnlID",
                                    join_table_name="citJournal")
                if "journal" in entry.fields_dict:
                    _add_or_link_row(src_value=entry.fields_dict["journal"].value, tgt_name="jrnlNm",
                                    value_to_ID=journal_to_ID,
                                    cite_id=cite_id_counter, relation_name="journal", relation_id_name="jrnlID",
                                    join_table_name="citJournal")

                # Series
                if "series" in entry.fields_dict:
                    _add_or_link_row(src_value=entry.fields_dict["series"].value, tgt_name="seriesNm",
                                    value_to_ID=series_to_ID,
                                    cite_id=cite_id_counter, relation_name="series", relation_id_name="seriesID",
                                    join_table_name="citSeries")

                # Booktitle
                if "booktitle" in entry.fields_dict:
                    _add_or_link_row(src_value=entry.fields_dict["booktitle"].value, tgt_name="bkTitleNm",
                                    value_to_ID=booktitle_to_ID,
                                    cite_id=cite_id_counter, relation_name="booktitle", relation_id_name="bktitleID",
                                    join_table_name="citBkTitle")

                # Keyword
                if "keywords" in entry.fields_dict:
                    for word in entry.fields_dict["keywords"].value.split(","):
                        _add_or_link_row(src_value=word.strip(), tgt_name="word", value_to_ID=keyWord_to_ID,
                                        cite_id=cite_id_counter, relation_name="keyWord", relation_id_name="keyWdID",
                                        join_table_name="citKeyWord")

                cite_id_counter += 1
            except Exception as err:
                print()
                print("------------------------------------------------------------------------------------")
                print(err)
                print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
                print(entry)


        return insert_builder.get_inserts()

    @classmethod
    def _inserts_for_dblp(cls, schema_namespace: str, this_schema: dict[str, dict[str, str]],
                        use_eval_data: bool) -> tuple[list[str], set[str]]:

        def get_with_remove(entry: dict, key):
            """
                Get the attribute and remove it from the entry--useful for checking if any attributes were missed
            """
            value = entry[key]
            del entry[key]
            return value

        # Build attribute -> type dictionary
        attr_types = {}
        for relation in this_schema.relations.values():
            attr_types[f"{schema_namespace}.{relation.name}"] = {attr.name: attr.data_type for attr in relation.attributes}

        insert_builder = InsertBuilder()

        if use_eval_data:
            import_file = "dblp/eval_data"
            import_dir = os.path.join(AmalgamImports.BASE_PATH, import_file)
            return cls._generate_insert_statements_from_jsonl(import_dir, this_schema, schema_namespace)
        else:
            import_file = "dblp/data.json"

        json_file_path = os.path.join(AmalgamImports.BASE_PATH, import_file)

        with open(json_file_path, 'r') as f:
            publications = json.load(f)

        type_to_relation_dblp = {'article': 'DArticle',
                                 'inproceedings': 'DInProceedings',
                                 'book': 'DBook',
                                 'phdthesis': 'PhDThesis',
                                 'mastersthesis': 'MasterThesis',
                                 'www': 'WWW'}

        type_to_attribute_mapping = {pub_type: {attr_name: [attr_name]
                                                for attr_name in attr_types[f"{schema_namespace}.{pub_type}"]
                                                if attr_name != "pid" and attr_name != "author"}
                                     for pub_type in type_to_relation_dblp.values()}

        # Autoincrement ID counters
        id_counter = {pub_type: 0 for pub_type in type_to_relation_dblp.values()} | {"PubAuthors": 0}

        for entry in publications:
            pub_type = type_to_relation_dblp[get_with_remove(entry, "pub_type")]
            id_counter[pub_type] += 1


            #
            #   Publication types without pid attributes, presumably because they can only have a single author
            #
            authors = get_with_remove(entry, "author")
            single_author = type(authors) is str
            if pub_type in ["MasterThesis", "PhDThesis"]:
                assert single_author, f"Expected exactly one author for {pub_type} but got {len(authors)}"
                literals = {"author": authors}
            else:
                literals = {"pid": id_counter[pub_type]}

                if single_author:
                    id_counter["PubAuthors"] += 1
                    authors = [authors]

                for auth in authors:
                    id_counter["PubAuthors"] += 1
                    insert_builder.get_insert_sql(fields_dict=None,
                                                  target_relation=f"{schema_namespace}.PubAuthors",
                                                  attribute_literals=literals | {"author": auth},
                                                  attr_types=attr_types)

            #
            #   Map entry.
            #

            # For each multi-valued attribute remaining, only take the first value
            for key in entry.keys():
                if type(entry[key]) is list:
                    entry[key] = entry[key][0]

            insert_builder.get_insert_sql(fields_dict=entry,
                                          target_relation=f"{schema_namespace}.{pub_type}",
                                          attribute_literals=literals,
                                          attr_types=attr_types,
                                          attribute_mapping=type_to_attribute_mapping[pub_type])

        return insert_builder.get_inserts(), set([rel for rel, count in id_counter.items() if count > 0])

    @classmethod
    def _inserts_for_a_schemas(cls, schema_namespace: str, this_schema: Schema,
                        use_eval_data: bool) -> tuple[list[str], set[str]]:

        database_name = this_schema.database_name

        if use_eval_data:
            import_dir = f"{database_name}/eval_data"
        else:
            import_dir = f"{database_name}/data"

        json_file_path = os.path.join(AmalgamImports.BASE_PATH, import_dir)

        return cls._generate_insert_statements_from_jsonl(json_file_path, this_schema, schema_namespace)


IMPORT_MAP = {
    Dataset.AMALGAM.value: {"a1": AmalgamImports._inserts_for_a_schemas, "a2": AmalgamImports._inserts_for_a_schemas,
                            "a3": AmalgamImports._inserts_for_a_schemas, "a4": AmalgamImports._inserts_for_a_schemas,
                            "dblp": AmalgamImports._inserts_for_dblp},
    Dataset.EHR.value: {"mimic": EHRImports._inserts_for_mimic}
}

def get_sql_for_insert(this_schema: Schema, namespace: str, use_eval_data: bool) -> list[str]:

    insert_statement_list, tables_found = (IMPORT_MAP[this_schema.dataset_name][this_schema.database_name]
                                           (namespace, this_schema, use_eval_data))

    for expected_relation in this_schema.relations:
        assert expected_relation in tables_found, \
            (f"Expected relation {expected_relation} but it was not found in data.")

    return insert_statement_list
