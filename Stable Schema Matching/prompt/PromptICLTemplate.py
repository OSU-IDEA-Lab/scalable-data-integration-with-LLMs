import copy

from utils.enums import COL_INFO


def get_info_types(col_info_to_display):
    info_types = []
    if COL_INFO.NAME in col_info_to_display:
        info_types.append('name')
    if COL_INFO.TYPE in col_info_to_display:
        info_types.append('type')
    if COL_INFO.COL_DESC in col_info_to_display:
        info_types.append('description')
    if COL_INFO.DATA in col_info_to_display:
        info_types.append('sample data instances')

    str_information = ', '.join(info_types[:-1]) + " and " + info_types[-1]

    if COL_INFO.TABLE_DESC in col_info_to_display:
        str_information = 'relation name and description, as well as the ' + str_information

    return str_information

def count_tokens(string: str, tokenizer):
    return len(tokenizer.encode(string))

class BaselinePrompt(object):

    def __init__(self, tokenizer, *args, **kwargs):
        self.tokenizer = tokenizer

    def format_prompt(self, test_case):
        n_prompt = {}
        n_prompt_intro = self.get_n_instruction(test_case)
        for key in n_prompt_intro.keys():
            n_prompt[key] = "\n\n".join([self.system, n_prompt_intro[key]])
        return n_prompt

    def format(self, target: dict, *args, **kwargs):
        prompt = self.format_prompt(target)
        sum_tokens = 0
        for single_prompt in prompt.values():
            sum_tokens += count_tokens(single_prompt, tokenizer=self.tokenizer)

        return {
            "id": target["id"],
            "n_prompts": self.n_prompts,
            "prompt_tokens": sum_tokens,
            "prompt": prompt,
            "gold_mapping": target["gold_mapping"],
            "source_schema": target["source_schema"],
            "target_schema": target["target_schema"]
        }


class BasicScoringPrompt(object):

    def __init__(self, tokenizer, *args, **kwargs):
        self.tokenizer = tokenizer

    def format_prompt(self, test_case, mode):

        system_component = [{
            "role": "system",
            "content": self.system,
        }]

        instructions = self.get_instruction_template()

        n_prompt = {}
        mcq_prompts, mcq_dict = self.get_MCQ(test_case, mode)
        for key in mcq_prompts.keys():
            question_components = [instructions, mcq_prompts[key]]
            question = "\n\n".join(question_components)
            question_component = [{
                "role": "user",
                "content": question,
            }]

            # n_prompt[key] = self.system + question
            n_prompt[key] = self.format_question(None, system_component, question_component)

        return n_prompt, mcq_dict


    def format_question(self,answer_prefix, system_component, question_component):

        prompt_components = system_component + question_component
        prompt = self.tokenizer.apply_chat_template(prompt_components, tokenize=False, add_generation_prompt=True)


        if answer_prefix is not None:
            prompt += f"\n{answer_prefix}"

        return prompt

    def format(self, target: dict, mode:str, *args, **kwargs):

        prompt, mcq_dict = self.format_prompt(target, mode)
        sum_tokens = 0
        for single_prompt in prompt.values():
            sum_tokens += count_tokens(single_prompt, tokenizer=self.tokenizer)



        return {
            "id": target["id"],
            "prompt": prompt,
            "mcq_dict": mcq_dict,
            "gold_mapping": target["gold_mapping"],
            "answer_prefix": self.get_answer_prefix(),
            "n_prompts": self.n_prompts,
            "prompt_tokens": sum_tokens
        }


class BasicICLPrompt(object):
    NUM_EXAMPLE = None
    SEP_EXAMPLE = "\n\n"

    def __init__(self, tokenizer, example_selector=None, *args, **kwargs):
        self.tokenizer = tokenizer
        self.example_selector = example_selector

    def get_example_prefix(self):
        return " Some examples are provided based on similar problems."

    def format_prompt(self, test_case, example_components={}):

        system_component = [{
            "role": "system",
            "content": self.system,
        }]

        # if len(example_components) > 0:
        #     system_component[0]["content"] += self.get_example_prefix()


        # instructions = self.get_instruction_template(example=test_case) ###changed. moved to get_source_target_intro
        answer_prefix = self.get_answer_prefix(test_case)

        # if not self.n_prompts:
        #     prompt_info = self.get_source_target_intro(example=test_case)
            # question_components = [prompt_info, instructions]
            # question = "\n\n".join(question_components)
            # question_component = [{
            #     "role": "user",
            #     "content": prompt_info,
            # }]
            #
            # prompt = self.format_question(answer_prefix, system_component, example_components, question_component)
            #
            # return prompt
        if self.n_prompts:
            n_prompt = {}
            n_prompt_intro = self.get_n_source_target_intro(example=test_case)
            for key in n_prompt_intro.keys():
                # question_components = [n_prompt_intro[key], instructions]
                # question = "\n\n".join(question_components)
                question_component = [{
                    "role": "user",
                    "content": n_prompt_intro[key],
                }]
                n_prompt[key] = self.format_question(answer_prefix, system_component, example_components.get(key,[]), question_component)

            return n_prompt


    def format_question(self,answer_prefix, system_component, example_components, question_component):

        prompt_components = system_component + example_components + question_component
        prompt = self.tokenizer.apply_chat_template(prompt_components, tokenize=False, add_generation_prompt=True)


        if answer_prefix is not None:
            prompt += f"\n{answer_prefix}"

        return prompt

    def get_instruction_template(self, example):
        raise NotImplementedError()

    def format_example(self, example: dict):
        #MM exaamples as prompt answer

        # Construct the list of dictionaries for the roles
        example_components = [
            {"role": "user", "content": example[0]},
            {"role": "assistant", "content": example[1]}
        ]

        return example_components


    def get_examples(self, target, max_seq_len, max_ans_len):
        prompt_example = {}
        n_valid_example = 0

        n_prompts = self.format_prompt(target, {})
        if self.example_selector and self.NUM_EXAMPLE != 0:
            examples = self.example_selector.get_examples(target, self.NUM_EXAMPLE)
            for example in examples:
                example_components = self.format_example(example)
                for attr, forward_prompt in n_prompts.items():
                    if attr not in prompt_example:
                        prompt_example[attr] = []
                    temp_ex = copy.deepcopy(prompt_example)
                    temp_ex[attr] += example_components
                    n_prompts = self.format_prompt(target, temp_ex)
                    forward_tokens = count_tokens(string=n_prompts[attr], tokenizer=self.tokenizer)

                    if forward_tokens + max_ans_len <= max_seq_len:
                        prompt_example[attr] = prompt_example[attr] + example_components

                        n_valid_example = int(len(prompt_example) / 2)  # 2 for user and assistant
                        if n_valid_example >= self.NUM_EXAMPLE:
                            break

        return prompt_example, n_valid_example

    # def get_examples(self, target, max_seq_len, max_ans_len):
    #     prompt_example = []
    #     example_ids = []
    #     example_components = []
    #     n_valid_example = 0
    #
    #     if self.example_selector and self.NUM_EXAMPLE != 0:
    #         examples = self.example_selector.get_examples(target, self.NUM_EXAMPLE)
    #         for example in examples:
    #
    #             example_components = self.format_example(example)
    #             forward_prompt = self.format_prompt(target, prompt_example + example_components)
    #             forward_tokens = count_tokens(string=forward_prompt, tokenizer=self.tokenizer)
    #
    #             if forward_tokens + max_ans_len <= max_seq_len:
    #                 prompt_example = prompt_example + example_components
    #                 example_ids.append(example['id'])
    #
    #                 n_valid_example = int(len(prompt_example) / 2)  # 2 for user and assistant
    #                 if n_valid_example >= self.NUM_EXAMPLE:
    #                     break
    #
    #     return prompt_example, n_valid_example, example_ids

    def format(self, target: dict, max_seq_len: int, max_ans_len: int, *args, **kwargs):

        example_components, n_valid_example = self.get_examples(target, max_seq_len, max_ans_len)

        if self.n_prompts:
            # if n_valid_example > 0:
            #     raise NotImplementedError("n prompts and examples greater than 0 is not implemented.")
            # prompt = self.format_prompt(target)
            prompt = self.format_prompt(target, example_components)
            sum_tokens = 0
            for single_prompt in prompt.values():
                sum_tokens += count_tokens(single_prompt, tokenizer=self.tokenizer)
        else:
            prompt = self.format_prompt(target)
            # prompt = self.format_prompt(target, example_components)
            sum_tokens = count_tokens(prompt, tokenizer=self.tokenizer)


        return {
            "n_prompts": self.n_prompts,
            "prompt_tokens": sum_tokens,
            "prompt": prompt,
            "answer_prefix": self.get_answer_prefix(target),
            # "gold_sql": target["gold_sql"],
            "gold_mapping": target["gold_mapping"],
            # "gold_mapping": self.get_gold_mapping(target),
            # "n_examples": n_valid_example,
            # "id_examples": example_ids,
            "id": target["id"],
            **({"source_schema": target["source_schema"]} if "source_schema" in target else {}),
            **({"target_schema": target["target_schema"]} if "target_schema" in target else {})
}





