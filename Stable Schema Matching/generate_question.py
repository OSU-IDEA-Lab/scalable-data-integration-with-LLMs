"""
Generate questions for LLMs and save it as a task
"""
# python generate_question.py --k_shot 2 --example_type prompt_as_example_style --selector_type random

import argparse
import os, getpass
import sys
import json

from prompt.prompt_builder import prompt_factory
from utils.data_builder import load_data, load_predicted_matches, get_seed
from utils.enums import *
from tqdm import tqdm
from transformers import AutoTokenizer

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

PATH_DATA = "data/"
OUTPUT_PATH = "dataset/"
OUTPUT_PATH = "dataset/"

sys.path.append("./")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--guidelines', nargs='+', choices=[GUIDELINE.DATA_VALUES, GUIDELINE.COLUMN_TYPE],
                        help=f"A list of guidelines for the json prompt", default=None)
    parser.add_argument("--dataset", type=str, choices=["valentine", "ehr", "bird", "synthea", 'gdc'], required=True)
    parser.add_argument("--k_shot", type=int, default=0, help="Number of examples")
    parser.add_argument("--prompt_repr", type=str, choices=[
                                                            REPR_TYPE.MM_Evaluator,
                                                            REPR_TYPE.MM_MCQ_Formatter,
                                                            REPR_TYPE.MMCandidate,
                                                            REPR_TYPE.MMConfidence,
                                                            REPR_TYPE.MMConfidence2,
                                                            REPR_TYPE.TaDa,
                                                            REPR_TYPE.ConfidenceScore,
                                                            REPR_TYPE.LogitsConfidenceScoringPrompt,
                                                            REPR_TYPE.CoTLogitsPrompt,
                                                            REPR_TYPE.One2N_Json,
                                                            REPR_TYPE.One2N_NL,
                                                            REPR_TYPE.N2One_Json,
                                                            REPR_TYPE.N2One_NL,
                                                            REPR_TYPE.N2M_NL,
                                                            REPR_TYPE.N2M_JSON
                                                            # REPR_TYPE.SOURCE_TARGET,
                                                            # REPR_TYPE.VIEW_NULL,
                                                            # REPR_TYPE.VIEW_NULL_ROWS,
                                                            # REPR_TYPE.SCHEMA_ALIGN_JSON,
                                                            # REPR_TYPE.SCHEMA_ALIGN_JSON_COL_EX,
                                                            # REPR_TYPE.SCHEMA_ALIGN_JSON_NO_SQL,
                                                            # REPR_TYPE.SCHEMA_ALIGN_JSON_NO_SQL_WITH_TYPE,
                                                            # REPR_TYPE.SCHEMA_ALIGN_JSON_NO_SQL_WITH_TYPE_VAL,
                                                            # REPR_TYPE.VIEW_NULL_COL_EX
                                                            ])
    parser.add_argument("--selector_type", type=str, choices=[SELECTOR_TYPE.RANDOM, SELECTOR_TYPE.NULL_COVERAGE, SELECTOR_TYPE.MatchMaker],
                        default=None,
                        help="The method by which few-shot examples are selected")
    parser.add_argument("--n_rows", type=int, default=None, help="Number of rows for each table in the prompt")
    parser.add_argument("--data_instance_selector", type=str, default=DataInstanceSelector.Random, choices=[DataInstanceSelector.Random, DataInstanceSelector.Most_Frequent, DataInstanceSelector.Weighted_Sampler, DataInstanceSelector.Random_Unique])
    parser.add_argument("--n_col_ex", type=int, default=None, help="Number of examples for each column")
    parser.add_argument("--max_seq_len", type=int, default=None, help="The maximal length that LLM takes")
    parser.add_argument("--max_ans_len", type=int, help="The maximal length that an answer takes")
    parser.add_argument("--tokenizer", type=str,
                        choices=[
                            # LLM.DeepSeekCoderInstruct,
                            # LLM.Phi3_3p8,
                            # LLM.DeepSeekCoderBase,
                            # LLM.Llama3p1_8Quantized,
                            # LLM.Llama3p1_8QuantizedCalib,
                            # LLM.CodeS,
                            # LLM.StarCoder15,
                            LLM.DeepSeekCoderLiteInstruct,
                            LLM.Qwen2_5_32B,
                            LLM.Llama3_1_GPTQ,
                            LLM.Llama3p1_8,
                            LLM.Phi3,
                            LLM.CodeLlama13GPTQ],
                        help="The tokenizer of the model we plan to run over the generated prompts")
    parser.add_argument('-c','--col_info', nargs='+',
                        help=f'If {REPR_TYPE.N2M_NL} specified, then must specify which column info to include.',
                        choices=[COL_INFO.NAME, COL_INFO.TYPE, COL_INFO.DATA, COL_INFO.COL_DESC, COL_INFO.TABLE_DESC], default=[COL_INFO.NAME, COL_INFO.TYPE])
    parser.add_argument("--seed", type=int, required=True, help="The seed used for randomizing different prompt inputs. For example, it determines what is sampled (e.g., for instance data, in-context examples) and the order that columns appear in the prompt.")
    parser.add_argument("--swap_tables", type=str2bool, required=True,
                        help="Whether source and target tables should be swapped. If True, will also swap the ground truth pairs so (a, b) becomes (b, a)")
    #The action="store_true" sets --reasoning to True if the flag is provided, and the default value is False when the flag is not used.

    parser.add_argument("--reasoning", type=str2bool, default=False, help="Enable greedy mode (True/False)")

    parser.add_argument("--folder_name", default=None,
                        help="for confidence score:\nThe name of the test case folder where the confidence score prompt should be added.\nfor MatchMaker: \nthe path to the Result-...")
    parser.add_argument("--ensemble_mode", default=None, choices=['disjoint', 'intersection', 'majority_vote', 'union'],
                        help="The ensemble mode to use for input predicted matches in the confidence score prompt.")
    parser.add_argument("--num_samples", type=int, default=None)

    parser.add_argument("--isConfidence2", type=str2bool, default=False)

    args = parser.parse_args()
    swapped_score = False
    if args.prompt_repr == REPR_TYPE.ConfidenceScore or args.prompt_repr == REPR_TYPE.LogitsConfidenceScoringPrompt :
        if "swap_T" in args.folder_name:
            swapped_score = True
        if (args.folder_name is None or args.ensemble_mode is None):
            print()
            print(f'ERROR!: If prompt_repr == {args.prompt_repr}, then  both the test case folder (`--folder_name`) and the ensemble mode (`--ensemble_mode`) must be specified.')
            exit()
    if (args.prompt_repr == REPR_TYPE.MM_MCQ_Formatter or args.prompt_repr == REPR_TYPE.MMConfidence or args.prompt_repr == REPR_TYPE.MMConfidence2) and args.folder_name is None:
            print()
            print(f'ERROR!: If prompt_repr == {args.prompt_repr}, then  both the test case folder (`--folder_name`) must be specified.')
            exit()

    elif args.prompt_repr == REPR_TYPE.N2M_NL and (args.col_info is None or len(args.col_info) == 0):
        print()
        print(f'ERROR!: If prompt_repr == {REPR_TYPE.N2M_NL}, then must specify which content is '
              f'displayed for each column/attribute with -c (--col_info)')
        exit()

    if args.dataset == "valentine" and (COL_INFO.COL_DESC in args.col_info or COL_INFO.TABLE_DESC in args.col_info):
        print()
        print(
            f'ERROR!: There is no desc available for dataset == valentine.')
        exit()

    if (not args.col_info is None) and len(args.col_info) > 0:
        args.col_info = sorted(args.col_info)


    username = getpass.getuser()
    os.environ['HF_HOME'] = f'/nfs/stak/users/{username}/hpc-share/huggingface'
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.max_seq_len is None:
        max_seq_len = tokenizer.model_max_length
    else:
        max_seq_len = args.max_seq_len

    # select the prompt
    prompt = (prompt_factory(repr_type=args.prompt_repr, k_shot=args.k_shot, selector_type=args.selector_type, isConfidence2=args.isConfidence2)
              (reasoning=args.reasoning,seed=args.seed, guidelines=args.guidelines, n_rows=args.n_rows, n_col_example=args.n_col_ex,
               tokenizer=tokenizer, data_instance_selector=args.data_instance_selector, col_info=args.col_info, dataset=args.dataset, isConfidence2=args.isConfidence2, swapped= swapped_score, repr_type=args.prompt_repr))

    # format all questions
    questions = list()
    token_cnt = 0

    func_name = f"get_dataset_json"



    if args.prompt_repr == REPR_TYPE.ConfidenceScore or args.prompt_repr == REPR_TYPE.LogitsConfidenceScoringPrompt or args.prompt_repr == REPR_TYPE.MM_MCQ_Formatter or args.prompt_repr == REPR_TYPE.MMConfidence  or args.prompt_repr == REPR_TYPE.MMConfidence2 or args.prompt_repr == REPR_TYPE.MM_Evaluator:
        if args.prompt_repr != REPR_TYPE.MM_MCQ_Formatter and args.prompt_repr != REPR_TYPE.MMConfidence and args.prompt_repr != REPR_TYPE.MMConfidence2 and args.prompt_repr != REPR_TYPE.MM_Evaluator:
            data = load_predicted_matches(args.num_samples, args.ensemble_mode, args.folder_name, args.dataset)
        else:
            with open(args.folder_name, 'r') as f:
                data = json.load(f)
                data = data["eval"]


        if "one_to_n" in args.folder_name:
            mode = "one_to_n"
        else:
            mode = "n_to_one"

        for question_json in tqdm(data):
            if args.prompt_repr == REPR_TYPE.MM_MCQ_Formatter or args.prompt_repr == REPR_TYPE.MMConfidence  or args.prompt_repr == REPR_TYPE.MMConfidence2 or args.prompt_repr == REPR_TYPE.MM_Evaluator:
                question_format = prompt.format(target=question_json,
                                                max_seq_len=max_seq_len,
                                                max_ans_len=args.max_ans_len)
            else:
                question_format = prompt.format(target=question_json, mode= mode)

            questions.append(question_format)

            token_cnt += question_format["prompt_tokens"]

        # cost estimated
        token_cnt = float(token_cnt) / len(questions)
        print(
            f"Total {len(questions)} questions, {token_cnt} tokens per prompt, {token_cnt / len(questions)} tokens per question")
        model_name = args.tokenizer
        if "/" in model_name:
            model_name = model_name.split("/")[1]

        if args.prompt_repr == REPR_TYPE.MM_Evaluator:
            seed_ = get_seed(args.folder_name)
            if "-processed" in seed_:
                seed_ = seed_.replace("-processed", "")
            path_generate = os.path.normpath(
                f"{os.path.dirname(args.folder_name)}/{args.prompt_repr}")
            questions_filename = f"questions-s-{seed_}.json"
        elif args.prompt_repr == REPR_TYPE.MM_MCQ_Formatter  or args.prompt_repr == REPR_TYPE.MMConfidence or args.prompt_repr == REPR_TYPE.MMConfidence2 or args.prompt_repr == REPR_TYPE.MM_Evaluator:
            seed_ = get_seed(args.folder_name)
            path_generate = os.path.normpath(
                f"{OUTPUT_PATH}baseline/{args.dataset}/{args.prompt_repr}/{model_name}-"
                f"{prompt.name}-{args.max_ans_len}")
            questions_filename = f"questions-s-{seed_}.json"
        else:
            questions_filename = f"{args.ensemble_mode}-{'_'.join(args.col_info)}-questions.json"
            if args.prompt_repr == REPR_TYPE.LogitsConfidenceScoringPrompt:
                path_generate = os.path.normpath(f"{OUTPUT_PATH}process/{args.dataset}/{args.folder_name}/{args.num_samples}/{args.prompt_repr}")
            else:
                path_generate = os.path.normpath(f"{OUTPUT_PATH}process/{args.dataset}/{args.folder_name}/confidence")
        os.makedirs(path_generate, exist_ok=True)

        try:
            with open(os.path.join(path_generate, questions_filename), "w") as file:
                json.dump(questions, file, indent=4)
            print(f"File saved successfully at {os.path.join(path_generate, questions_filename)}")
        except Exception as e:
            print(f"Failed to save file: {e}")

    elif args.prompt_repr == REPR_TYPE.TaDa:
        data = load_data(args.dataset, PATH_DATA, args.seed)

        for question_json in tqdm(getattr(data, func_name)(swap_tables=args.swap_tables)):
            question_format = prompt.format(target=question_json)

            questions.append(question_format)

            token_cnt += question_format["prompt_tokens"]

        # cost estimated
        token_cnt = float(token_cnt) / len(questions)
        print(
            f"Total {len(questions)} questions, {token_cnt} tokens per prompt, {token_cnt / len(questions)} tokens per question")

        if args.k_shot > 0:
            exp_count = [0] * (args.k_shot + 1)
            for q in questions:
                exp_count[q['n_examples']] += 1
            print('Example count per prompt: ')
            print('Examp: ' + ' '.join(['{:10d}'] * len(exp_count)).format(*[i for i in range(len(exp_count))]))
            print('Count: ' + ' '.join(['{:10d}'] * len(exp_count)).format(*[count for count in exp_count]))

        n_total_tokens = len(questions) * args.max_ans_len + token_cnt

        # save questions

        model_name = args.tokenizer
        if "/" in model_name:
            model_name = model_name.split("/")[1]


        path_generate = os.path.normpath(
                f"{OUTPUT_PATH}baseline/{args.dataset}/{model_name}-"
                f"{prompt.name}-{args.max_ans_len}")
        os.makedirs(path_generate, exist_ok=True)
        questions_filename = f"questions.json"
        try:
            with open(os.path.join(path_generate, questions_filename), "w") as file:
                json.dump(questions, file, indent=4)
            print(f"File saved successfully at {os.path.join(path_generate, questions_filename)}")
        except Exception as e:
            print(f"Failed to save file: {e}")

    elif args.prompt_repr == REPR_TYPE.MMCandidate:

        data = load_data(args.dataset, PATH_DATA, args.seed)

        for question_json in tqdm(getattr(data, func_name)(swap_tables=args.swap_tables)):
            question_format = prompt.format(target=question_json,
                                            max_seq_len=max_seq_len,
                                            max_ans_len=args.max_ans_len)

            questions.append(question_format)

            token_cnt += question_format["prompt_tokens"]

        # cost estimated
        token_cnt = float(token_cnt) / len(questions)
        print(
            f"Total {len(questions)} questions, {token_cnt} tokens per prompt, {token_cnt / len(questions)} tokens per question")


        # save questions

        model_name = args.tokenizer
        if "/" in model_name:
            model_name = model_name.split("3.1-")[1].split("B-")[0]

        path_generate = os.path.normpath(
                f"{OUTPUT_PATH}baseline/{args.dataset}/{args.prompt_repr}/{model_name}-"
                f"{prompt.name}-{args.max_ans_len}")
        os.makedirs(path_generate, exist_ok=True)
        questions_filename = f"questions-s-{args.seed}.json"
        try:
            with open(os.path.join(path_generate, questions_filename), "w") as file:
                json.dump(questions, file, indent=4)
            print(f"File saved successfully at {os.path.join(path_generate, questions_filename)}")
        except Exception as e:
            print(f"Failed to save file: {e}")

    else:
        data = load_data(args.dataset, PATH_DATA, args.seed)

        for question_json in tqdm(getattr(data, func_name)(swap_tables=args.swap_tables)):
            question_format = prompt.format(target=question_json,
                                            max_seq_len=max_seq_len,
                                            max_ans_len=args.max_ans_len)

            questions.append(question_format)

            token_cnt += question_format["prompt_tokens"]

        # cost estimated
        token_cnt = float(token_cnt) / len(questions)
        print(
            f"Total {len(questions)} questions, {token_cnt} tokens per prompt, {token_cnt / len(questions)} tokens per question")

        if args.k_shot > 0:
            exp_count = [0] * (args.k_shot + 1)
            for q in questions:
                exp_count[q['n_examples']] += 1
            print('Example count per prompt: ')
            print('Examp: ' + ' '.join(['{:10d}'] * len(exp_count)).format(*[i for i in range(len(exp_count))]))
            print('Count: ' + ' '.join(['{:10d}'] * len(exp_count)).format(*[count for count in exp_count]))

        n_total_tokens = len(questions) * args.max_ans_len + token_cnt

        # save questions

        model_name = args.tokenizer
        if "/" in model_name:
            model_name = model_name.split("3.1-")[1].split("B-")[0]

        extra_info = f'swap_{args.swap_tables}-'
        if args.n_rows is not None:
            extra_info += str(args.n_rows) + 'ROWS-'

        if args.n_col_ex is not None:
            extra_info += str(args.n_col_ex) + '-' + args.data_instance_selector + '-'

        if args.reasoning:
            extra_info += "reasoning-"

        if "dynamic" in args.prompt_repr :
            path_generate = os.path.normpath(
                f"{OUTPUT_PATH}process/{args.dataset}/{model_name}-{extra_info}"
                f"{prompt.name}-{'_'.join(args.col_info)}-{args.max_ans_len}")
        else:
            path_generate = os.path.normpath(
                f"{OUTPUT_PATH}process/{args.dataset}/{model_name}-{extra_info}"
                f"{prompt.name}-{args.max_ans_len}")
        os.makedirs(path_generate, exist_ok=True)
        questions_filename = f"questions-s-{args.seed}.json"
        try:
            with open(os.path.join(path_generate, questions_filename), "w") as file:
                json.dump(questions, file, indent=4)
            print(f"File saved successfully at {os.path.join(path_generate, questions_filename)}")
        except Exception as e:
            print(f"Failed to save file: {e}")
