import json
from operator import index

from eval.eval_utils import parse_alignments_from_view, _remove_nulls


from prompt.InstanceDataHandler import InstanceDataHandler
from prompt.PromptICLTemplate import get_info_types
from utils.data_builder import get_schemas_for_id, selective_format, load_schemas
from utils.enums import GUIDELINE, COL_INFO







class LogitsConfidenceScoringPrompt(object):
    def __init__(self, dataset, swapped, reasoning = False, *args, **kwargs):
        self.n_prompts = True
        self.reasoning = reasoning
        self.dataset_name = dataset
        self.col_info_to_display = kwargs['col_info']
        self.system = ''.join((
            "Act as a schema matching expert. Given the attribute from the source schema, ",
            "which of the following target attributes is the best match? Provide only the attribute name of the best match."
        ))
        self.schemas = None
        self.swapped = swapped
        self.set_options_format()
        self.MCQ_format = '''Question: Target attributes:\n{options}\n\ninput query:\n{query}\n\nAnswer: '''





    def get_MCQ(self, test_case, mode):
        mcq_prompts = {}
        # mcq_dict = {}
        candidates = {}

        if mode == "n_to_one":
            match_idx = 0
        else:
            match_idx = 1

        # Iterate over the valid_predicted_mappings in the test_case
        for attribute, mappings in test_case['valid_predicted_mappings'].items():
            # If there are no mappings for the attribute, skip it
            if not mappings:
                continue

            mcq_options = {}
            candidate_list = []
            for idx, mapping in enumerate(mappings):
                candidate_list.append(mapping[match_idx])
                mcq_options[idx+1] = self.get_option(test_case["id"], mapping[match_idx], mode)  # chr(65) is 'A', chr(66) is 'B', etc.
                # mcq_options[chr(65 + idx)] = mapping[match_idx]  # chr(65) is 'A', chr(66) is 'B', etc.

            # Join the options in the format 'A:value, B:value, C:value, D:value, ...'
            mcq_str = '\n'.join(f"{key}. {value}" for key, value in mcq_options.items())

            input_query = self.get_option(test_case["id"], attribute, mode, isQuery=True)

            mcq_prompts[attribute] = self.MCQ_format.format(options=mcq_str, query=input_query)
            # mcq_dict[attribute] = mcq_options
            candidates[attribute] = candidate_list

        return mcq_prompts, candidates

    def get_instruction_template(self):
        return ''.join((
            f"{self.attributes_template_desc}\n"
            ))


    def get_answer_prefix(self):
        return None
        # if self.reasoning:
        #     return None
        # assistant = '{ "A": '
        # return assistant

    def get_gold_mapping(self, example):
        return example['gold_mapping']

    def set_options_format(self):
        self.attributes_template = '{attr_name}'
        self.attributes_template_desc = ""
        if COL_INFO.TYPE not in self.col_info_to_display and COL_INFO.COL_DESC not in self.col_info_to_display:
            return
        self.schemas = load_schemas(self.dataset_name, self.swapped)
        if COL_INFO.TYPE in self.col_info_to_display:
            self.attributes_template += '({attr_type})'
            template_desc = "Both the input query and the schema options are formatted as `attribute name (data type)`.\n"
        if COL_INFO.COL_DESC in self.col_info_to_display:
            self.attributes_template += ': {attr_desc}'
            template_desc = "Both the input query and the schema options are formatted as `attribute name (data type) : description`.\n"
        self.attributes_template_desc = template_desc

    def get_option(self, id, attr, mode, isQuery = False):
        if self.schemas is None:
            return attr
        if isQuery:
            if mode == "n_to_one": #query is from target
                schema = self.schemas[id]["target_schema"]
            else:# one to n > query is from source
                schema = self.schemas[id]["source_schema"]
        else: #option is n
            if mode == "n_to_one": #option is from source
                schema = self.schemas[id]["source_schema"]
            else:# one to n > query is from target
                schema = self.schemas[id]["target_schema"]

        column_desc = {}
        column_desc['attr_name'] = attr
        column_desc['attr_type'] = schema['columns'][attr.lower()]['type']
        column_desc['attr_desc'] = schema['columns'][attr.lower()]['column_description']
        return selective_format(self.attributes_template, **column_desc)



class TaDa(object):
    def __init__(self, *args, **kwargs):
        self.n_prompts = True
        self.col_info_to_display = kwargs['col_info']
        self.source_target_intro = ''.join((
            "The relation from the source schema is the following:\n\n{source_desc}\n\n"
            "The attribute from the target schema is the following:\n\n{target_desc}\n\n"
        ))
        self.attr_intro = "Attribute name: {attr_name}\nAttribute description: {attr_desc}\n"
        self.rel_intro = ''
        self.context_level, self.rel_intro  = self.get_context_level()
        self.system = ''.join((
            "Act as a schema matcher for relational schemas. Your task is to create semantic matches "
            "that specify how the elements of the source schema and the target schema semantically "
            "correspond to one another. Two attributes semantically match if and only if there exists "
            "an invertible function that maps all values of one attribute to the other. "
            f"{self.context_level}"
        ))
        self.json_format = "`{\"yes\": [], \"no\": [], \"unknown\": []}`"
        self.task_desc = ''.join(("Explain which of the source attributes semantically match to {target_attr} from {target_rel} of the target schema. Lets work this out step by step "
                            "to make sure we get it correct. After your explanation, give a final decision JSON-formatted "
                            "like this: {output_format}. Under each of the following keys, list all "
                            "target attributes of {target_rel} that apply: yes - if there is an invertible function that maps "
                            "all values of the source attribute to the target attribute; no - if there is no such function. unknown - if there is not enough "
                            "information to decide"))

        self.instance_data_handler = InstanceDataHandler('0') #need the functions

    def get_context_level(self):
        rel_intro = "Relation name: {rel_name}\n"
        table_desc = ''
        if COL_INFO.TABLE_DESC in self.col_info_to_display:
            table_desc = "the description of the relation "
            rel_intro += "Relation description: {rel_desc}\n"
        template_des = ''.join((
            "First, I will input the name of a single relation from the source schema, "
            f"{table_desc}"
            "and the name and description of all its attributes. After that, I will input the same information of a single relation "
            "and a single attribute from the target schema. "))
        return template_des, rel_intro

    def get_source_desc(self, source_info, source_rel):
        rel_name = source_rel['rel_name']
        attr_intro = f"In the following, I will list all attributes of {rel_name}.\n"
        source_desc = [selective_format(self.rel_intro, **source_rel), attr_intro]
        for attr in source_info:
            source_desc.append(selective_format(self.attr_intro, **source_info[attr]))
        return "\n".join(source_desc)

    def get_n_instruction(self, example):
        source_info, source_rel, target_info, target_rel = self.get_n_source_target_intro(example)
        source_desc = self.get_source_desc(source_info, source_rel)

        target_relation_intro = selective_format(self.rel_intro, **target_rel)

        n_intro = {}
        for attr in target_info:
            target_desc = '\n'.join([target_relation_intro,selective_format(self.attr_intro, **target_info[attr])])
            source_target_desc = self.source_target_intro.format(
                source_desc=source_desc,
                target_desc=target_desc
            )
            task_desc = self.task_desc.format(target_attr=attr, target_rel=target_rel["rel_name"], output_format=self.json_format)
            n_intro[attr] = "\n\n".join([source_target_desc,task_desc])
        return n_intro

    def get_rel_attr_info(self, schema):
        rel = {"rel_name": schema["name"]}
        attr_info = {}
        if COL_INFO.TABLE_DESC in self.col_info_to_display:
            rel["rel_desc"] = schema["description"]
        for col in schema['columns']:
            attr_info[col["name"]] = {"attr_name": col["name"], "attr_desc": col["column_description"]}
        return attr_info, rel


    def get_n_source_target_intro(self, example):
        source_schema = example["source_schema"]
        target_schema = example["target_schema"]
        source_info, source_rel = self.get_rel_attr_info(source_schema)
        target_info, target_rel = self.get_rel_attr_info(target_schema)
        return source_info, source_rel, target_info, target_rel

    def get_answer(self, example):
        return example['gold_mapping']



class BasicPromptRepr(object):

    def __init__(self, seed, dataset, reasoning=False, guidelines=None, n_rows=None, n_col_example=None,
                 data_instance_selector='random',
                 *args, **kwargs):
        self.reasoning = reasoning
        self.n_prompts = False
        self.n_rows = n_rows
        self.n_col_example = n_col_example
        self.guidelines = guidelines
        self.dataset_name = dataset
        self.col_info_to_display = kwargs['col_info']
        self.instance_data_handler = InstanceDataHandler(seed, n_col_example, data_instance_selector)

        self.source_target_intro = ''.join((
            "The information about the relation from the source schema is as follows:\n\n{source_attributes_desc}\n\n"
            "The information about the relation from the target schema is as follows:\n\n{target_attributes_desc}\n\n"
        ))

    def string_rows(self, source_random_rows, target_random_rows):
        # Format each list of rows in the desired '[]\n[]\...' format
        source_random_rows_str = '\n'.join([str(row) for row in source_random_rows])
        target_random_rows_str = '\n'.join([str(row) for row in target_random_rows])
        return source_random_rows_str, target_random_rows_str

    def format_column_examples(self, top_n_values):
        columns_info = []
        for column, values in top_n_values.items():
            values_str = ', '.join(map(str, values))
            columns_info.append(f"{column}: {values_str}")
        return '\n'.join(columns_info)

    def get_source_target_intro(self, example):
        raise NotImplementedError("get_source_target_intro not implemented by template")

    def get_instruction_template(self, example):
        raise NotImplementedError("get_instruction_template not implemented by template")

    def get_answer_prefix(self, example):
        return None

    def get_answer(self, example):
        raise NotImplementedError(f"Prompt does not implement get_answer(). In-context examples are not supported.")

    def get_gold_mapping(self, example):
        self.get_answer(example)

def get_schema_json_desc(n_col_example, col_info_to_display, schema, data_instances, col_descs, relation):
    if COL_INFO.DATA in col_info_to_display and n_col_example is None:
        raise ValueError("Argument 'n_col_example' must be passed and cannot be None.")

    relation["columns"] = []

    # Iterate over each column in the original schema
    for column in schema["columns"]:
        attr_name = column.get("name")
        attr_type = column.get("type")
        instances = data_instances.get(attr_name, [])

        # Build the new attribute info based on COL_INFO flags
        attribute_info = {
            "name": attr_name if COL_INFO.NAME in col_info_to_display else None,
            "type": attr_type if COL_INFO.TYPE in col_info_to_display else None,
            "description": col_descs.get(attr_name) if COL_INFO.COL_DESC in col_info_to_display else None,
            "sample data instances": instances if COL_INFO.DATA in col_info_to_display else None
        }

        attribute_info = {k: v for k, v in attribute_info.items() if v is not None}

        relation['columns'].append(attribute_info)

    return relation







# N source column and one target column
class N2One_Json(BasicPromptRepr):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_prompts = True
        self.info_types = get_info_types(self.col_info_to_display)
        self.system = (
            "Act as a schema matcher for relational schemas. Your task is to create semantic matches "
            "that specify how the elements of the source schema and the target schema semantically "
            "correspond to one another. I will first provide the information of a single relation from the source schema, "
            f"including the {self.info_types} of all its attributes. "
            "Next, I will provide the same information for a single relation and a single attribute from "
            "the target schema."
        )



    def get_instruction_(self, target_attr, target_rel):
        if not self.reasoning:
            return ''.join((
                f"Identify which of the source attributes semantically match to {target_attr} from {target_rel}.",
                "Format your output like this: `{\"matches\": [\"<source attribute>,",
                f"{target_attr}\"",
                ", ...]}`. ",
                "Do not mention an attribute if there is not enough information to decide. ",
                f"If there is no source attribute matching the target attribute, return \"None,{target_attr}\". ",
                "Do not include any explanation."
            ))
        else:
            return ''.join((
            f"Explain which of the source attributes semantically match to {target_attr} from {target_rel}. ",
            "Let's work this out step by step to make sure we get it correct. ",
            "After your explanation, give a final decision formatted like this:`{\"matches\": [\"<source attribute>,",
            f"{target_attr}\"",
            ", ...]}`. ",
            "Do not mention an attribute if there is not enough information to decide. ",
            f"If there is no source attribute matching the target attribute, return \"None,{target_attr}\". "
        ))

    def build_n_intro(self, source_info, target_info, target_rel_name):
        source_desc = str(source_info).replace("'", '"')

        n_intro = {}
        for column_desc in target_info["columns"]:
            attr_name = column_desc.get("name")
            instructions = self.get_instruction_(attr_name, target_rel_name)  ###changed. moved to get_source_target_intro
            target_desc = self.get_target_desc(target_info, column_desc)
            source_target_desc = self.source_target_intro.format(
                source_attributes_desc=source_desc,
                target_attributes_desc=target_desc
            )
            n_intro[attr_name] = "\n\n".join([source_target_desc,instructions])
        return n_intro

    def get_n_source_target_intro(self, example):
        source_data_instances, target_data_instances, source_column_types, target_column_types = self.instance_data_handler.get_n_distinct_col_values(
            example)

        if COL_INFO.COL_DESC in self.col_info_to_display:
            source_column_descs = {col['name']: col['column_description'] for col in
                                   example['source_schema']['columns']}
            target_column_descs = {col['name']: col['column_description'] for col in
                                   example['target_schema']['columns']}
        else:
            source_column_descs = None
            target_column_descs = None

        if COL_INFO.TABLE_DESC in self.col_info_to_display:
            source_relation = {"relation name": example['source_schema']["name"],
                               "relation description": example['source_schema']["description"]}
            target_relation = {"relation name": example['target_schema']["name"],
                               "relation description": example['target_schema']["description"]}

        elif "name" in example['source_schema']:
            source_relation = {"relation name": example['source_schema']["name"]}
            target_relation = {"relation name": example['target_schema']["name"]}
        else:
            source_relation = {"relation name": "source relation"}
            target_relation = {"relation name": "target relation"}

        source_info = get_schema_json_desc(self.n_col_example, self.col_info_to_display, example['source_schema'], source_data_instances, source_column_descs, source_relation)
        target_info = get_schema_json_desc(self.n_col_example, self.col_info_to_display, example['target_schema'], target_data_instances, target_column_descs, target_relation)

        n_intro = self.build_n_intro(source_info, target_info, target_relation["relation name"])

        return n_intro

    def get_answer_prefix(self, example):
        if self.reasoning:
            return None
        assistant = '{ "matches": ['
        return assistant

    def convert_to_json_format(self, attribute_pairs):
        # Convert list of attribute pairs into the desired JSON-like string format
        matches = [f"{source},{target}" for source, target in attribute_pairs]
        return {'matches': matches}

    def get_gold_mapping(self, example):
        return {'matches': example['gold_mapping']}

    def get_answer(self, example):
        gold_alignments = [f"{item[0]}, {item[1]}" for item in example['gold_mapping']]
        return {'matches': gold_alignments}

    def get_target_desc(self, target_info, column_desc):
        target_desc_json = {"relation name": target_info["relation name"]}
        if "relation description" in target_info:
            target_desc_json["relation description"] = target_info["relation description"]
        target_desc_json["column"] = column_desc
        return str(target_desc_json).replace("'", '"')










class CoTLogitsPrompt(N2One_Json):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_prompts = True
        self.info_types = get_info_types(self.col_info_to_display)
        self.system = (
                "Act as a schema matcher for relational schemas. Your task is to create semantic matches "
                "that specify how the elements of the source schema and the target schema semantically "
                "correspond to one another. I will first provide the information of a single relation from the source schema, "
                f"including the {self.info_types} of all its attributes. "
                "Next, I will provide the same information for a single relation and a single attribute from "
                "the target schema."
            )



    def get_instruction_(self, target_attr, target_rel):
        if not self.reasoning:
            return ''.join((
                f"Identify which one of the source attributes semantically best match to {target_attr} from {target_rel}.",
                "Give a final decision formatted like `Best Match:<target attribute>`",
                f"If there is no source attribute matching the target attribute, return `Best Match:None`. ",
                "Do not include any explanation."
            ))
        else:
            return ''.join((
            f"Explain which of the source attributes semantically match to {target_attr} from {target_rel}. ",
            "Let's work this out step by step to make sure we get it correct. ",
            "After your explanation, give a final decision formatted like `Best Match:<target attribute>`",
            f"If there is no source attribute matching the target attribute, return `Best Match:None`. ",

        ))

    def get_answer_prefix(self, example):
        return None

    def get_gold_mapping(self, example):
        return example['gold_mapping']

    def get_answer(self, example):
        return None








class ConfidenceScoringPrompt(object):
    def __init__(self, dataset, swapped, reasoning = False, *args, **kwargs):
        self.n_prompts = True
        self.reasoning = reasoning
        self.dataset_name = dataset
        self.col_info_to_display = kwargs['col_info']
        self.system = ''.join((
            "Act as a schema matching expert. Your task is given the input and the MCQ format of the schema,",
            "predict the likelihood or relation score from 0-100 of the input query being related to each option.",
            "Your scores will be calibrated."
        ))
        self.schemas = None
        self.swapped = swapped
        self.set_options_format()
        self.MCQ_format = '''MCQ schema options:\n{options}\n\ninput query:\n{query}\n\n'''

    def get_MCQ(self, test_case, mode):
        mcq_prompts = {}
        mcq_dict = {}

        if mode == "n_to_one":
            match_idx = 0
        else:
            match_idx = 1

        # Iterate over the valid_predicted_mappings in the test_case
        for attribute, mappings in test_case['valid_predicted_mappings'].items():
            # If there are no mappings for the attribute, skip it
            if not mappings:
                continue

            mcq_options = {}
            for idx, mapping in enumerate(mappings):
                mcq_options[chr(65 + idx)] = self.get_option(test_case["id"], mapping[match_idx], mode)  # chr(65) is 'A', chr(66) is 'B', etc.
                # mcq_options[chr(65 + idx)] = mapping[match_idx]  # chr(65) is 'A', chr(66) is 'B', etc.

            mcq_options[chr(65 + len(mappings))] = "None of the options"

            # Join the options in the format 'A:value, B:value, C:value, D:value, ...'
            mcq_str = '\n'.join(f"{key}-{value}" for key, value in mcq_options.items())

            input_query = self.get_option(test_case["id"], attribute, mode, isQuery=True)

            mcq_prompts[attribute] = self.MCQ_format.format(options=mcq_str, query=input_query)
            mcq_dict[attribute] = mcq_options

        return mcq_prompts, mcq_dict

    def get_instruction_template(self):
        return ''.join((
            "First, I will provide the input MCQ schema options.\n",
            "Next, I will provide an input query that needs to be evaluated against these options.\n",
            f"{self.attributes_template_desc}\n",
            "Assess each option independently and assign it a relation score that reflects the likelihood of the input query being semantically related to each option. Use a scale from 0-100, where:\n\n",
            "    0 means the option doesn't match with the input query at all.\n",
            "    100 means the option is a perfect match with the input query.\n",
            "    Use a range of scores between 0 and 100 to reflect varying levels of relevance, with higher scores indicating a closer match.\n",
            "    Every two options should have different scores, unless they both don't match the query, in which case they should each have a score of 0.\n",
            "    If none of the options are related to the query, assign a score of 100 to \"None of the options\".\n\n",
            "Let's work this out step by step to make sure we get it correct. ",
            "After your explanation, give a final decision formatted like this: {\"A\": score, \"B\": score, \"C\": score, ...}, using each MCQ letter as the key and the corresponding score as the value."
        ))


    def get_answer_prefix(self):
        return None
        # if self.reasoning:
        #     return None
        # assistant = '{ "A": '
        # return assistant

    def get_gold_mapping(self, example):
        return example['gold_mapping']

    def set_options_format(self):
        self.attributes_template = '{attr_name}'
        self.attributes_template_desc = ""
        if COL_INFO.TYPE not in self.col_info_to_display and COL_INFO.COL_DESC not in self.col_info_to_display:
            return
        self.schemas = load_schemas(self.dataset_name, self.swapped)
        if COL_INFO.TYPE in self.col_info_to_display:
            self.attributes_template += '({attr_type})'
            template_desc = "Both the input query and the schema options are formatted as `attribute name (data type)`.\n"
        if COL_INFO.COL_DESC in self.col_info_to_display:
            self.attributes_template += ': {attr_desc}'
            template_desc = "Both the input query and the schema options are formatted as `attribute name (data type) : description`.\n"
        self.attributes_template_desc = template_desc

    def get_option(self, id, attr, mode, isQuery = False):
        if self.schemas is None:
            return attr
        if isQuery:
            if mode == "n_to_one": #query is from target
                schema = self.schemas[id]["target_schema"]
            else:# one to n > query is from source
                schema = self.schemas[id]["source_schema"]
        else: #option is n
            if mode == "n_to_one": #option is from source
                schema = self.schemas[id]["source_schema"]
            else:# one to n > query is from target
                schema = self.schemas[id]["target_schema"]

        column_desc = {}
        column_desc['attr_name'] = attr
        column_desc['attr_type'] = schema['columns'][attr.lower()]['type']
        column_desc['attr_desc'] = schema['columns'][attr.lower()]['column_description']
        return selective_format(self.attributes_template, **column_desc)





