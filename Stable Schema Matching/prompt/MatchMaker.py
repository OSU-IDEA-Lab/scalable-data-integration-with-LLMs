import ast

from eval.eval_utils import validate_MM_candidates
from utils.data_builder import selective_format, load_schemas, get_schemas_for_id
from utils.enums import COL_INFO


class CandidateRefiner(object):
    def __init__(self, dataset, swapped, reasoning=False, *args, **kwargs):
        self.n_prompts = True
        self.reasoning = reasoning
        self.dataset_name = dataset
        self.col_info_to_display = kwargs['col_info']
        self.schema_name = self.get_schema_name(dataset)
        self.system = ''.join((
            f"""You are an expert {self.schema_name} matching ranker. Your task is to take the {self.schema_name} candidates and based""",
            "on the input, refine the candidates to select the 5 most likely matches to the input query. Return ",
            "ONLY the keys.",
            "\n—\n",
            "Follow the following format.\n",
            "Input Schema: List of key: value pairs\n",
            "Input Query: input query\n",
            "Reasoning: Let’s think step by step in order to {produce the refined_string_list}. We ...\n",
            "Refined String List: Five most likely matches to input query. Include maximum of the 5 most ",
            "likely matches to the input query. Return ONLY the keys.",
            "\n—\n"
        ))
        self.schemas = None
        self.swapped = swapped
        self.set_options_format()
        self.Q_format = '''Input Schema:\n{candidates}\nInput Query:\n{query}\nReasoning: Let’s think step by step in order to'''

    def set_options_format(self):
        self.attributes_template = '{tname}-' + '{attr_name}'
        if COL_INFO.TYPE not in self.col_info_to_display and COL_INFO.COL_DESC not in self.col_info_to_display:
            return
        self.schemas = load_schemas(self.dataset_name, self.swapped)
        if COL_INFO.TYPE in self.col_info_to_display:
            self.attributes_template += '({attr_type})'
        if COL_INFO.TABLE_DESC in self.col_info_to_display:
            self.attributes_template = self.attributes_template + ': Table {tname} details-{tdesc} Attribute {attr_name} details -{attr_desc}'
        elif COL_INFO.COL_DESC in self.col_info_to_display:
            self.attributes_template += ': Attribute {attr_name} details -{attr_desc}'

    def get_n_source_target_intro(self, example, mode='n_to_one'):
        q_prompts = {}

        # Iterate over the valid_predicted_mappings in the test_case
        for column in example['target_schema']['columns']:
            attribute = column['name']
            q_options = []
            for candidate in example['source_schema']['columns']:
                candidate_name = candidate['name']
                q_options.append(self.get_option(example["id"], candidate_name,
                                                 mode))

            input_query = self.get_option(example["id"], attribute, mode, isQuery=True)

            q_prompts[attribute] = self.Q_format.format(candidates=str(q_options), query=input_query)

        return q_prompts

    def get_instruction_template(self):
        return ''

    def get_answer_prefix(self, test_case):
        return None

    def get_gold_mapping(self, example):
        return example['gold_mapping']

    def get_option(self, id, attr, mode, isQuery=False):
        if self.schemas is None:
            return attr
        if isQuery:
            if mode == "n_to_one":  # query is from target
                schema = self.schemas[id]["target_schema"]
            else:  # one to n > query is from source
                schema = self.schemas[id]["source_schema"]
        else:  # option is n
            if mode == "n_to_one":  # option is from source
                schema = self.schemas[id]["source_schema"]
            else:  # one to n > query is from target
                schema = self.schemas[id]["target_schema"]

        column_desc = {}
        column_desc['tname'] = schema['name']
        if 'description' in schema:
            column_desc['tdesc'] = schema['description']
        column_desc['attr_name'] = attr
        column_desc['attr_type'] = schema['columns'][attr.lower()]['type']
        column_desc['attr_desc'] = schema['columns'][attr.lower()]['column_description']
        return selective_format(self.attributes_template, **column_desc)

    def get_schema_name(self, dataset):
        schema_name = {"ehr": "OMOP", "synthea": "OMOP", "bird": "BIRD"}
        return schema_name[dataset]


class MCQ_Formatter(object):
    def __init__(self, dataset, swapped, reasoning=False, *args, **kwargs):
        self.n_prompts = True
        self.system = ''.join((
            "You are an expert MCQ formatter. Your task is to take a list of schema values and convert them ",
            "into a multiple choice question format with (letter)Schema value, where the schema values should ",
            "be key(description).\n",
            "—\n",
            "Follow the following format. \n",
            "Input: input list of schema values Mcq: MCQ format of schema values e.g (A)Schema value, "
            "(B)Schema value. Do not include additional options, only the schema values as options. where the ",
            "schema values should be key(description). Add a No Match option.\n_\n"
        ))
        self.schemas = None
        self.template = '''Input : {input} Mcq: '''

    def get_n_source_target_intro(self, example, mode='n_to_one'):
        q_prompts = {}

        # Iterate over the valid_predicted_mappings in the test_case
        for attribute, predicted_mapping in example['predicted_mapping'].items():
            if 'Refined String List:' in predicted_mapping:
                input = predicted_mapping.split('Refined String List:')[1]
                q_prompts[attribute] = self.template.format(input=input)
        return q_prompts

    def get_instruction_template(self):
        return ''

    def get_answer_prefix(self, test_case):
        return None

    def get_gold_mapping(self, example):
        return example['gold_mapping']


class Evaluator(object):
    def __init__(self, dataset, swapped, reasoning=False, *args, **kwargs):
        self.n_prompts = True
        self.system = ''.join((
            "You are a schema matching expert, your task is to rate if any of the suggested matches are potential ",
            "good matches for the query. Be lenient and rate a match as good (4 or 5) if it is relevant to the ",
            "query. Rate the matches from 1-5. If none of the matches are good, rate 0. \n",
            "—\n",
            "Follow the following format.\n",
            "Query: The query.\n",
            "Answers: possible matches\n",
            "Reasoning: Let’s think step by step in order to {produce the rating}. We ...\n",
            "Rating: Rate if any of the suggested matches are good for the query from 1-5. Only output the ",
            "rating and nothing else.\n",
            "_\n"
        ))
        self.dataset_name = dataset
        self.schemas = load_schemas(self.dataset_name, swapped=False)
        self.template = '''Query: {query}\nAnswers: {answers}\nReasoning: Let’s think step by step in order to '''

    def get_n_source_target_intro(self, example, mode='n_to_one'):
        q_prompts = {}
        for attribute, predicted_mapping in example['predicted_mapping'].items():
            answers = self.get_answers(predicted_mapping)
            if answers is None:
                continue
            query = self.get_query(example["id"], attribute)
            q_prompts[attribute] = self.template.format(query=query, answers=answers)
        return q_prompts

    def get_query(self, id, attr):
        schema = self.schemas[id]["target_schema"]
        return schema['name'] + '-' + attr

    def get_instruction_template(self):
        return ''

    def get_answer_prefix(self, test_case):
        return None

    def get_gold_mapping(self, example):
        return example['gold_mapping']

    def get_answers(self, predicted_mapping):
        if predicted_mapping == []:
            return None
        return [x[0] for x in predicted_mapping]


class ConfidenceScoring(object):
    def __init__(self, dataset, swapped, reasoning=False, *args, **kwargs):
        self.n_prompts = True
        self.reasoning = reasoning
        self.dataset_name = dataset
        self.col_info_to_display = kwargs['col_info']
        self.system = ''.join((
            "You are a schema matching expert. Your task is given the input and the MCQ format of the schema,",
            "predict the likelihood or relation score from 0-100 of the input query being related to each option.",
            "Your scores will be calibrated. If there is no good match score No Match as 100",
            "\n—\n",
            "Follow the following format.",
            "Input Mcq: Input MCQ format of schema values",
            "Input Query: input query",
            "Relation: Relation score of input query being related to the option as value. Assess each independently",
            " including No Match, returning a score from 0-100 for each. Return with key as MCQ letter",
            "e.g (A) and score=value as JSON",
            "\n—\n"
        ))
        self.schemas = load_schemas(self.dataset_name, swapped=False)
        self.swapped = swapped

        self.attributes_template = '{attr_name}({attr_type})'
        self.MCQ_format = '''Input Mcq:\n{options}\nInput Query:\n{query}Relation:\n'''

    def get_n_source_target_intro(self, example, mode='n_to_one'):
        q_prompts = {}
        for attribute, predicted_mapping in example['predicted_mapping'].items():
            options = self.get_options(predicted_mapping, example)
            if options is None:
                continue
            query = self.get_query(example["id"], attribute)
            q_prompts[attribute] = self.MCQ_format.format(query=query, options=options)
        return q_prompts

    def get_instruction_template(self):
        return ''

    def get_answer_prefix(self, test_case):
        return None

    def get_gold_mapping(self, example):
        return example['gold_mapping']

    def get_query(self, id, attr):
        schema = self.schemas[id]["target_schema"]
        return schema['name'] + '-' + attr + '(' + schema['columns'][attr.lower()]['type'] + ')'

    def get_options(self, text, test_case):
        if '(A)' in text:
            return '(A)' + text.split('(A)')[1]
        return None

# including the implementation of formatter instead of prompting LLM for it
class ConfidenceScoring2(ConfidenceScoring):
    def get_options(self, text, test_case):
        source_schema, target_schema = get_schemas_for_id(test_case, self.dataset_name)
        valid_options = [source_schema['name'] + '-' + col['name'] + '(' + col['type'] + ')' for col in
                         source_schema['columns']]
        res = validate_MM_candidates(text, valid_options)
        print('res', res)
        if res is None:
            return None
        options = get_formatted_options(res)
        print('options', options)
        return options


def get_formatted_options(result):

    mcq_options = {}
    candidate_list = []
    for idx, candidate in enumerate(result):
        candidate_list.append(candidate)
        mcq_options[chr(65 + idx)] = candidate  # chr(65) is 'A', chr(66) is 'B', etc.
    mcq_options[chr(65 + len(result))] = "No Match."  # chr(65) is 'A', chr(66) is 'B', etc.

    # Join the options in the format 'A:value, B:value, C:value, D:value, ...'
    return '\n'.join(f"({key}){value}" for key, value in mcq_options.items())
