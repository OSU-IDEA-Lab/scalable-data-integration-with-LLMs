import sys
import os



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from prompt.MatchMaker import get_formatted_options
from eval.eval_dataset import categorize_valentine, categorize_ehr, categorize_bird, print_aggregates
from eval.eval_utils import compute_prf1e, validate_MM_candidates

import argparse
import ast
import glob
import json
import re

from utils.data_builder import get_schemas_for_id


# Custom Exception
class FileNotFoundErrorCustom(Exception):
    pass


def find_result_files(folder, seed=7564, is_eval=False):
    if is_eval:
        pattern = os.path.join(folder, f"RESULTS-*-s-{seed}-processed.json")
    else:
        pattern = os.path.join(folder, f"RESULTS-*-{seed}.json")
    files = [os.path.basename(f) for f in glob.glob(pattern)]
    if len(files) == 0:
        raise FileNotFoundErrorCustom(f"Pattern {pattern} Not Found")
    return files[0]


def check_dict_values_match(dictionary, candidates):
    values = [v.strip() for v in dictionary.values() if
              "No Match".strip().lower() not in v.strip().lower()]  # Normalize spaces
    candidates = [c.strip() for c in candidates]  # Normalize spaces
    return sorted(values) == sorted(candidates)  # Order-independent check


def extract_mcq_options(text, candidates):
    # print('//////\ntext')
    # print(text)
    # print('candidates')
    # print(candidates)

    # Updated regex to handle spaces/newlines before and after (A), (B), etc.
    pattern = r"\(\s*([A-F])\s*\)\s*([^\n]+)"
    # pattern = r"\(\s*(A|B|C|D|E|F)\s*\)\s*([\s\S]*?)(?=\(\s*[A-F]\s*\)|$)"
    # pattern = r"^\s*\(?['\"]?A['\"]?\)?\s*([\s\S]*?)(?=\(\s*[B-F]\s*\)|$)"

    if "(A)" in text:
        text = '(A)'+text.split("(A)")[1]
    matches = re.findall(pattern, text)

    if len(matches) > 6 or len(matches) < 2:
        # print('\nlen problem\n')# Ensure exactly 5 options are present
        # print(text)# Ensure exactly 5 options are present
        # print(matches)# Ensure exactly 5 options are present
        return None

    options = {key: value.strip() for key, value in matches}  # Normalize spaces

    return options if check_dict_values_match(options, candidates) else None


def extract_mcq_scores(text):
    if text is None:
        return None
    # Regex pattern to match (A) 85, (B) 70, etc., allowing spaces/newlines
    pattern = r'\(?\s*["]?\(?([A-F])\)?["]?\s*\)?\s*:\s*(\d{1,3})'

    matches = re.findall(pattern, text)

    # Convert scores to integers and ensure they are within [0, 100]
    scores = {key: int(value) for key, value in matches if 0 <= int(value) <= 100}
    # print("..............................",scores)

    # Ensure all six options (A-F) are present
    return scores if len(scores) == 6 else None


def save_parsable_results(candidate_result, formatter_result, confidence_result, dataset_name, output_path, isConfidence2 = 'False'):
    with open(candidate_result, 'r') as f:
        candidate_data = json.load(f)
    if args.isConfidence2 != "True":
        with open(formatter_result, 'r') as f:
            formatter_data = json.load(f)
    with open(confidence_result, 'r') as f:
        confidence_data = json.load(f)

    test_cases = candidate_data['eval']
    config_ = candidate_data["config"]
    for i, test_case in enumerate(test_cases):
        print("---------------------",i)
        # check if all candidates in refined list exist in source schema
        if args.isConfidence2 != "True":
            formatter_answer = formatter_data["eval"][i]["predicted_mapping"]
        confidence_answer = confidence_data["eval"][i]["predicted_mapping"]
        source_schema, target_schema = get_schemas_for_id(test_case, dataset_name)
        valid_options = [source_schema['name'] + '-' + col['name'] + '(' + col['type'] + ')' for col in
                         source_schema['columns']]
        for attr, answer in test_case["predicted_mapping"].items():
            print(' => attr', attr)
            candidates = validate_MM_candidates(answer, valid_options)
            print('candidates', candidates)
            if candidates is None:

                # print("\n==========================\ncandidates None",answer)
                test_case["predicted_mapping"][attr] = []
                continue

            # check if fomatter kept the names correct and generate a dictionary of a:candidate1 etc
            if args.isConfidence2 != "True":
                options = extract_mcq_options(formatter_answer[attr], candidates)
            else:
                formatted_options = get_formatted_options(candidates)
                options = extract_mcq_options(formatted_options, candidates)


            print('options', options)
            if options is None:

                # print("\n=================================\noptions None")
                # if args.isConfidence2 != "True":
                #     print(formatter_answer[attr])
                # else:
                #     formatted_options = get_formatted_options(candidates)
                #     print(formatted_options)
                # print("Invalid formatter: ", formatter_answer[attr], "\ncandidates\n:", candidates)
                test_case["predicted_mapping"][attr] = []
                continue


            # retreive scores and generate final prediction . if no match:100 skip, else sort.
            scores = extract_mcq_scores(confidence_answer.get(attr, None))


            print('scores', scores)
            if scores is None:
                # print("Invalid Scores format: ", confidence_answer[attr])
                test_case["predicted_mapping"][attr] = []
                continue

            no_match_key = None
            for key, value in options.items():
                if "No Match".strip().lower() in value.strip().lower():
                    no_match_key = key
                    break

            if no_match_key:
                if scores[no_match_key] == 100:
                    # print(" No match scored 100 ")
                    test_case["predicted_mapping"][attr] = []
                    continue

                options.pop(no_match_key)
                scores.pop(no_match_key)

            # Sort options based on descending scores
            sorted_options = sorted(options.items(), key=lambda x: scores.get(x[0], 0), reverse=True)
            sorted_candidates = [[x[1], attr] for x in sorted_options]
            # Update predicted mapping with sorted options
            test_case["predicted_mapping"][attr] = sorted_candidates

    with open(output_path, "w") as file:
        json.dump({'config': config_, 'eval': test_cases}, file, indent=4)
        print(f"Results and outputs saved to {output_path}")


def get_last_char_as_int(s):
    if not s:  # Check if the string is empty
        print("Error: Empty string")
        return None
    last_char = s[-1]  # Get the last character
    try:
        rate = int(last_char)
        return rate if rate <= 5 or rate >= 0 else None
    except ValueError:
        print(f"\n\nError: '{last_char}' is not a number")
        print(s)
        return None


def get_demo(path, s):
    with open(path, 'r') as f:
        eval_data = json.load(f)

    test_cases = eval_data['eval']
    sub_demo = []
    for i, test_case in enumerate(test_cases):
        for attr, eval_ in test_case["predicted_mapping"].items():
            rating = get_last_char_as_int(eval_)
            if rating is None or rating < 4:
                continue
            entry = {
                "idx": i,
                "id": test_case["id"],
                "attr": attr,
                "seed": s,
                "rating": rating
            }
            sub_demo.append(entry)

    return sub_demo


def format_alignments(processed_ali):
    ali = []
    for src_trgt in processed_ali:
        target = src_trgt[1].lower()
        if "-" in src_trgt[0]:
            source = src_trgt[0].split("-")[1].split("(")[0].lower()
        else:
            source = src_trgt[0].lower()
        ali.append([source, target])
        # [
        #     "businesses-address(text)",
        #     "address"
        # ]
    return ali


def get_top_k(res_path, k):
    with open(res_path, 'r') as f:
        data = json.load(f)
    results = {"eval": [], 'config': data["config"]}
    for entry in data['eval']:
        top_k_alignments = []
        for attr in entry['predicted_mapping']:
            if len(entry['predicted_mapping'][attr]) < k:
                top_k_alignments.extend(format_alignments(entry['predicted_mapping'][attr]))
            else:
                top_k_alignments.extend(format_alignments(entry['predicted_mapping'][attr][:k]))
        results['eval'].append({
            "id": entry['id'],
            "gold_mapping": [[x[0].lower(), x[1].lower()] for x in entry['gold_mapping']],
            "predicted_mappings": top_k_alignments
        })

    return results


def eval_result(res, dataset_name, quest_path, output_path):
    with open(quest_path, 'r') as f:
        quest_data = json.load(f)

    annotated_eval = []

    column_dict = {'gold': [],
                   'pred': [],
                   'tp': [],
                   'fp': [],
                   'fn': [],
                   'precision': [],
                   'recall': [],
                   'f1': [],
                   'accuracy': [],
                   'accuracy2': [],
                   'effort': []
                   }

    # # Add columns based on dataset
    # if dataset_name == 'valentine':
    #     column_dict['dataset'] = []
    #     column_dict['problem_type'] = []
    #     column_dict['vertical_overlap'] = []
    #     column_dict['horizontal_overlap'] = []
    #     column_dict['schema_noise'] = []
    #     column_dict['data_noise'] = []
    # elif dataset_name == 'ehr' or dataset_name == 'synthea':
    #     column_dict['source_db'] = []
    #     column_dict['source_table'] = []
    #     column_dict['source_cols'] = []
    #     column_dict['target_db'] = []
    #     column_dict['target_table'] = []
    #     column_dict['target_cols'] = []
    #     column_dict['gt_size'] = []
    # elif dataset_name == 'bird':
    #     column_dict['domain'] = []
    #     column_dict['source_db'] = []
    #     column_dict['source_table'] = []
    #     column_dict['source_cols'] = []
    #     column_dict['target_db'] = []
    #     column_dict['target_table'] = []
    #     column_dict['target_cols'] = []
    #     column_dict['gt_size'] = []

    for i, test_case in enumerate(res['eval']):
        if test_case["predicted_mappings"] is []:
            print(f"{test_case['id']}: empty mappings")
            p, r, f1, accuracy, accuracy2, e = compute_prf1e(0, 0, 0, test_case, dataset_name)
            tp_alignments, fp_alignments, fn_alignments = 0, 0, 0
        else:
            predicted_alignments = set(tuple(pair) for pair in test_case["predicted_mappings"])
            gold_alignments = set(tuple(pair) for pair in test_case['gold_mapping'])

            tp_alignments = len(gold_alignments.intersection(predicted_alignments))
            fp_alignments = len(predicted_alignments.difference(gold_alignments))
            fn_alignments = len(gold_alignments.difference(predicted_alignments))

            p, r, f1, accuracy, accuracy2, e = compute_prf1e(tp=tp_alignments, fp=fp_alignments, fn=fn_alignments,
                                        test_case=test_case, dataset_name=dataset_name)

        test_case['precision'] = p
        test_case['recall'] = r
        test_case['f1'] = f1
        test_case['fp'] = fp_alignments
        test_case['fn'] = fn_alignments
        test_case['accuracy'] = accuracy
        test_case['accuracy2'] = accuracy2
        test_case['effort'] = e

        annotated_eval.append(test_case)

        # # Categorize based on the dataset
        # if dataset_name == 'valentine':
        #     categorize_valentine(test_case, column_dict, quest_data)
        # elif dataset_name == 'ehr' or dataset_name == 'synthea':
        #     categorize_ehr(test_case, column_dict, quest_data)
        # elif dataset_name == 'bird':
        #     categorize_bird(test_case, column_dict, quest_data)

        column_dict['gold'].append(gold_alignments)
        column_dict['pred'].append(predicted_alignments)
        column_dict['tp'].append(tp_alignments)
        column_dict['fp'].append(fp_alignments)
        column_dict['fn'].append(fn_alignments)
        column_dict['precision'].append(p)
        column_dict['recall'].append(r)
        column_dict['f1'].append(f1)
        column_dict['effort'].append(e)
        column_dict['accuracy'].append(accuracy)
        column_dict['accuracy2'].append(accuracy2)

    with open(output_path, 'w') as f_out:
        json.dump(annotated_eval, f_out, indent=4)  # Use annotated_eval to include "id" in results

    print(f'Results saved to {output_path}')

    return pd.DataFrame(column_dict)


def eval_results(result_dfs, output_txt_path, excel_file_path, dataset_name):
    original_stdout = sys.stdout
    with open(output_txt_path, 'w') as f_out:
        sys.stdout = f_out

        table_style = "tsv"  # See available styles here: https://pypi.org/project/tabulate/
        print('----- Dataset-wide:')
        eval_df = print_aggregates(result_dfs, grouping=None, table_style=table_style, ensemble=True)

        eval_df.to_excel(excel_file_path, index=False, engine='openpyxl')
        print()

        sys.stdout = original_stdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, choices=["bird", "ehr", "synthea"])
    parser.add_argument('--action', type=str, choices=["process_icl", "process", "icl", "eval"])
    parser.add_argument('--isConfidence2', type=str, choices=["True", "False"], default="False")
    parser.add_argument('--seeds', type=int, nargs='+', help="List of seed values")  # like --seeds 1 2 3 4 5
    parser.add_argument('--main_seed', type=int, choices=[7564, 87849, 268799, 4756, 98784, 62879],
                        default=7564)  # like --seeds 7564 87849 268799 4756 98784 62879

    args = parser.parse_args()
    dir_ = f"dataset/baseline/{args.dataset}"
    if args.action == "eval":
        for seed in args.seeds:
            candidate_path = os.path.join(dir_, "candidate_mm", "70-candidate_mm-1024")
            question_file_name = f"questions-s-{args.main_seed}.json"
            question_path = os.path.join(candidate_path, question_file_name)
            if args.isConfidence2 == "True":
                result_dir = os.path.join(dir_, "mm_icl2")
                print("2")
            else:
                result_dir = os.path.join(dir_, "mm_icl")
            result_json = find_result_files(result_dir, seed, is_eval=True)
            result_path = os.path.join(result_dir, result_json)
            for k in range(1, 6):
                output_path = os.path.join(result_dir, f'AN-{k}-' + result_json)
                result_k = get_top_k(result_path, k)

                result_dfs = [
                    eval_result(result_k, args.dataset, question_path, output_path)]

                output_txt_path = output_path.replace(".json", ".txt")
                excel_file_path = output_path.replace(".json", ".xlsx")

                eval_results(result_dfs, output_txt_path, excel_file_path, args.dataset)

                print(f"Results and outputs saved to {output_txt_path}")


    else:
        candidate_path = os.path.join(dir_, "candidate_mm", "70-candidate_mm-1024")
        question_file_name = f"questions-s-{args.main_seed}.json"
        question_path = os.path.join(candidate_path, question_file_name)
        formatter_path = os.path.join(dir_, "formatter_mm", "Meta-Llama-3.1-70B-Instruct-GPTQ-INT4-formatter_mm-1024")
        if args.isConfidence2 == "True":
            confidence_path = os.path.join(dir_, "confidence2_mm",
                                           "Meta-Llama-3.1-70B-Instruct-GPTQ-INT4-confidence2_mm-1024")
            output_path = os.path.join(dir_, "mm2")
        else:
            confidence_path = os.path.join(dir_, "confidence_mm",
                                           "Meta-Llama-3.1-70B-Instruct-GPTQ-INT4-confidence_mm-1024")
            output_path = os.path.join(dir_, "mm")
        if args.action == "process_icl":
            candidate_path = candidate_path.replace("-1024", "_ICL-1024")
            formatter_path = formatter_path.replace("-1024", "_ICL-1024")
            if args.isConfidence2 == "True":
                output_path = os.path.join(dir_, "mm_icl2")
                confidence_path = confidence_path.replace("-1024", "_ICL-1024")
            else:
                confidence_path = confidence_path.replace("-1024", "_ICL-1024")
                output_path = os.path.join(dir_, "mm_icl")

        os.makedirs(output_path, exist_ok=True)
        demo = []
        for seed in args.seeds:
            print("...........seed: ", seed)
            try:
                candidate_json = find_result_files(candidate_path, seed)
                candidate_result_path = os.path.join(candidate_path, candidate_json)
                formatter_json = find_result_files(formatter_path, seed)
                formatter_result_path = os.path.join(formatter_path, formatter_json)
                confidence_json = find_result_files(confidence_path, seed)
                confidence_result_path = os.path.join(confidence_path, confidence_json)
                if "process" in args.action:
                    output_name = candidate_json.replace(".json", "-processed.json").replace("-s-7564-", f"-s-{seed}-")
                    output_path = os.path.join(output_path, output_name)
                    save_parsable_results(candidate_result_path, formatter_result_path, confidence_result_path,
                                          args.dataset, output_path, args.isConfidence2)
                elif args.action == "icl":
                    if args.isConfidence2 == "True":
                        eval_dir = os.path.join(dir_, "mm2", "eval_mm")
                    else:
                        eval_dir = os.path.join(dir_, "mm", "eval_mm")
                    eval_json = find_result_files(eval_dir, seed)
                    eval_path = os.path.join(eval_dir, eval_json)
                    print(eval_path)
                    demo.extend(get_demo(eval_path, seed))
            except Exception as e:
                print(e)

        if args.action == "icl":
            sorted_demo = sorted(demo, key=lambda x: x["rating"], reverse=True)
            output_name = "icl.json"
            output_path = os.path.join(output_path, output_name)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sorted_demo, f, indent=4)
            print(f"Sorted entries written to {output_path}")
