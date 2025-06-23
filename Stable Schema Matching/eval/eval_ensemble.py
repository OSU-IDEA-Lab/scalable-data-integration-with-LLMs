import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import ast
import json
import pandas as pd
from utils.enums import REPR_TYPE

from eval.computational_eval import print_computational_cost
from eval.eval_dataset import OUTPUT_FORMAT, categorize_valentine, categorize_ehr, categorize_bird, print_aggregates
from eval.eval_utils import compute_prf1e

from prompt.InstanceDataHandler import get_dataset_name
from utils.data_builder import get_seed

from collections import Counter
import random

def get_union(valid_predicted_mappings):
    union_result = {}
    for attribute in valid_predicted_mappings[0].keys():
        all_values = set()
        for mapping in valid_predicted_mappings:
            for value in mapping.get(attribute, []):
                all_values.add(tuple(value))
        union_result[attribute] = [list(v) for v in all_values]
    return union_result


def get_intersection(valid_predicted_mappings):
    intersection_result = {}
    for attribute in valid_predicted_mappings[0].keys():
        common_values = set(tuple(x) for x in valid_predicted_mappings[0].get(attribute, []))
        for mapping in valid_predicted_mappings[1:]:
            common_values &= set(tuple(x) for x in mapping.get(attribute, []))
        intersection_result[attribute] = [list(v) for v in common_values]
    return intersection_result


def get_disjoint(valid_predicted_mappings):
    union = get_union(valid_predicted_mappings)
    intersection = get_intersection(valid_predicted_mappings)

    disjoint_result = {}
    for attribute in union.keys():
        union_values = set(tuple(x) for x in union[attribute])
        intersection_values = set(tuple(x) for x in intersection[attribute])
        disjoint_values = union_values - intersection_values
        disjoint_result[attribute] = [list(v) for v in disjoint_values]
    return disjoint_result


# def get_majority_vote(valid_predicted_mappings):
#     majority_result = {}
#     for attribute in valid_predicted_mappings[0].keys():
#         attribute_values = []
#         for mapping in valid_predicted_mappings:
#             attribute_values.extend(mapping.get(attribute, []))
#
#         counts = Counter(tuple(value) for value in attribute_values)
#
#         if not attribute_values:
#             majority_result[attribute] = []
#         else:
#             majority_value = max(counts, key=counts.get, default=[])
#             majority_result[attribute] = [list(majority_value)]
#     return majority_result

def get_majority_vote(valid_predicted_mappings):
    majority_result = {}
    for attribute in valid_predicted_mappings[0].keys():
        attribute_values = []
        for mapping in valid_predicted_mappings:
            attribute_values.extend(mapping.get(attribute, []))

        counts = Counter(tuple(value) for value in attribute_values)

        if not attribute_values:
            majority_result[attribute] = []
        else:
            # Find the max count
            max_count = max(counts.values(), default=0)
            # Get all elements with the max count
            majority_values = [list(key) for key, count in counts.items() if count == max_count]
            majority_result[attribute] = majority_values

    return majority_result



def get_majority_vote_tada(valid_predicted_mappings):
    majority_result = {}
    for attribute in valid_predicted_mappings["yes_list"][0].keys():
        all_yes = valid_predicted_mappings["yes_list"][0][attribute]
        all_yes += valid_predicted_mappings["yes_list"][1][attribute]
        all_yes += valid_predicted_mappings["yes_list"][2][attribute]

        all_no = valid_predicted_mappings["no_list"][0][attribute]
        all_no += valid_predicted_mappings["no_list"][1][attribute]
        all_no += valid_predicted_mappings["no_list"][2][attribute]

        # Count occurrences
        yes_counts = Counter(map(tuple, all_yes))
        no_counts = Counter(map(tuple, all_no))

        # Get the majority vote for each unique pair and filter classified as 'yes'
        classified_yes = [
            pair for pair in yes_counts.keys() | no_counts.keys()
            if yes_counts[pair] > no_counts[pair]
        ]

        majority_result[attribute] = classified_yes

    return majority_result



def get_match_gold_mappings(gold_mapping, predicted_mappings):
    gold_mapping_set = {tuple(gold_pair) for gold_pair in gold_mapping}

    matched_mappings = {}
    for attr, predicted_list in predicted_mappings.items():
        matched_mappings[attr] = [pair for pair in predicted_list if tuple(pair) in gold_mapping_set]

    return matched_mappings


def get_valid_predicted_mappings(result_paths):
    results = {}
    valid_predicted_mappings={}
    for path in result_paths:
        with open(path, 'r') as f:
            for entry in json.load(f)['eval']:
                entry_id = entry['id']
                entry_valid_predicted_mappings = entry['valid_predicted_mappings']
                if entry_id not in results:
                    results[entry_id] = {
                        "id": entry_id,
                        "gold_mapping": ast.literal_eval(entry['gold_mapping'].lower()),

                    }
                    valid_predicted_mappings[entry_id] = []

                valid_predicted_mappings[entry_id].append(entry_valid_predicted_mappings)

    return results, valid_predicted_mappings

def concatenate_dict_values(data):
    result = []
    for value in data.values():
        if isinstance(value, list):  # Ensure the value is a list
            result += value
    return result

def get_tada_valid_predicted_mappings(result_paths):
    results = {}
    valid_predicted_mappings={}
    for path in result_paths:
        with open(path, 'r') as f:

            for entry in json.load(f)['eval']:
                entry_id = entry['id']
                if entry_id not in results:
                    results[entry_id] = {
                        "id": entry_id,
                        "gold_mapping": ast.literal_eval(entry['gold_mapping'].lower()),

                    }
                    valid_predicted_mappings[entry_id] = {"yes_list": [], "no_list": []}

                valid_predicted_mappings[entry_id]["yes_list"].append(entry['valid_yes'])
                valid_predicted_mappings[entry_id]["no_list"].append(entry['valid_no'])

    return results, valid_predicted_mappings

def ensemble_results(result_paths, mode, isTada):

    if isTada:
        results, valid_predicted_mappings = get_tada_valid_predicted_mappings(result_paths)
    else:
        results, valid_predicted_mappings = get_valid_predicted_mappings(result_paths)
    for id in results:

        if mode == 'union':
            results[id]['valid_predicted_mappings'] = get_union(valid_predicted_mappings[id])

        elif mode == 'intersection':
            results[id]['valid_predicted_mappings'] = get_intersection(valid_predicted_mappings[id])

        elif mode == 'disjoint': # union - intersection
            results[id]['valid_predicted_mappings'] = get_disjoint(valid_predicted_mappings[id])

        elif mode == 'majority_vote':
            if isTaDa:
                results[id]['valid_predicted_mappings'] = get_majority_vote_tada(valid_predicted_mappings[id])
            else:
                results[id]['valid_predicted_mappings'] = get_majority_vote(valid_predicted_mappings[id])

        elif mode == 'match_gold_mappings':
            results[id]['valid_predicted_mappings'] = get_union(valid_predicted_mappings[id])
            results[id]['valid_predicted_mappings'] = get_match_gold_mappings(results[id]['gold_mapping'], results[id]['valid_predicted_mappings'])

        # if "movies" in id:
        #     print("\n\n".join([str(x) for x in valid_predicted_mappings[id]]))
        #
        #     print(results[id]['valid_predicted_mappings'])
        #     print("\n\n")
        #     print(results[id]['gold_mapping'])
        #     exit(0)
    return results


def get_ensemble_results_df(dataset_name,ensemble_dir, results, quest_file, mode):
    with open(quest_file, 'r') as f:
        quest_data = {entry['id']: entry for entry in json.load(f)}

    annotated_eval = []

    column_dict = {'gold': [],
                   'pred': [],
                   'tp': [],
                   'fp': [],
                   'fn': [],
                   'precision': [],
                   'recall': [],
                   'f1': [],
                   'effort': [],
                   'accuracy2': [],
                   'accuracy': []
                   }

    # Add columns based on dataset
    if dataset_name == 'valentine':
        column_dict['dataset'] = []
        column_dict['problem_type'] = []
        column_dict['vertical_overlap'] = []
        column_dict['horizontal_overlap'] = []
        column_dict['schema_noise'] = []
        column_dict['data_noise'] = []
    elif dataset_name == 'ehr' or dataset_name == 'synthea':
        column_dict['source_db'] = []
        column_dict['source_table'] = []
        column_dict['source_cols'] = []
        column_dict['target_db'] = []
        column_dict['target_table'] = []
        column_dict['target_cols'] = []
        column_dict['gt_size'] = []
    elif dataset_name == 'bird':
        column_dict['domain'] = []
        column_dict['source_db'] = []
        column_dict['source_table'] = []
        column_dict['source_cols'] = []
        column_dict['target_db'] = []
        column_dict['target_table'] = []
        column_dict['target_cols'] = []
        column_dict['gt_size'] = []

    # Iterate through the pairs and process the results
    for i, (test_id, test_case) in enumerate(results.items()):
        if (i + 1) % 100 == 0:
            print(f'Evaluating {i + 1}th prediction')

        # Add 'id' key to each test case
        test_case['id'] = test_id

        '''
            Get mappings/alignments
        '''
        if "predicted_mappings" not in test_case: #for both way ensemble it already has the predicted mappings
            all_mappings = [item for sublist in test_case['valid_predicted_mappings'].values() for item in sublist]
            test_case["predicted_mappings"] = all_mappings


        if test_case["predicted_mappings"] is None:
            print(f"{test_case['id']}: empty mappings")
            p, r, f1, accuracy, accuracy2, e = compute_prf1e(0, 0, 0, test_case, dataset_name)

            tp_alignments, fp_alignments, fn_alignments = 0, 0, 0
        else:

            # print('gold_alignments')
            # print(gold_alignments)
            predicted_alignments = set(tuple(pair) for pair in test_case["predicted_mappings"])
            gold_alignments = set(tuple(pair) for pair in test_case['gold_mapping'])

            tp_alignments = len(gold_alignments.intersection(predicted_alignments))
            fp_alignments = len(predicted_alignments.difference(gold_alignments))
            fn_alignments = len(gold_alignments.difference(predicted_alignments))

            p, r, f1, accuracy, accuracy2, e = compute_prf1e(tp=tp_alignments, fp=fp_alignments, fn=fn_alignments, test_case=test_case, dataset_name=dataset_name)

        test_case['precision'] = p
        test_case['recall'] = r
        test_case['f1'] = f1
        test_case['fp'] = fp_alignments
        test_case['fn'] = fn_alignments
        test_case['effort'] = e
        test_case['accuracy'] = accuracy
        test_case['accuracy2'] = accuracy2

        annotated_eval.append(test_case)

        # Categorize based on the dataset
        if dataset_name == 'valentine':
            categorize_valentine(test_case, column_dict, quest_data)
        elif dataset_name == 'ehr' or dataset_name == 'synthea':
            categorize_ehr(test_case, column_dict, quest_data)
        elif dataset_name == 'bird':
            categorize_bird(test_case, column_dict, quest_data)

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


    output_path = os.path.join(ensemble_dir, f'{mode}_ensemble_results.json')
    with open(output_path, 'w') as f_out:
        json.dump(annotated_eval, f_out, indent=4)  # Use annotated_eval to include "id" in results

    print(f'Results saved to {output_path}')

    return pd.DataFrame(column_dict)


def eval_results(result_dfs, output_txt_path, excel_file_path, dataset_name, base_dir=None):
    original_stdout = sys.stdout
    with open(output_txt_path, 'w') as f_out:
        sys.stdout = f_out
        if base_dir is not None:
            print(100 * '-')
            print(f'At {base_dir}...')
            print(' - Including the following files in evaluation:')
            for file in result_files:
                print(f'\t{file}')
            if len(files_excluded) > 0:
                print(' - EXCLUDING the following files from evaluation:')
                for file, reason in files_excluded:
                    print(f'\t{file}: {reason}')
            print(100 * '-')

        table_style = "tsv"  # See available styles here: https://pypi.org/project/tabulate/
        print('----- Dataset-wide:')
        eval_df = print_aggregates(result_dfs, grouping=None, table_style=table_style, ensemble=True)

        eval_df.to_excel(excel_file_path, index=False, engine='openpyxl')
        print()

        if dataset_name == 'valentine':
            for group in ['dataset', 'problem_type', 'schema_noise', 'data_noise']:
                print(f'----- {group}:')
                print_aggregates(result_dfs, grouping=[group], table_style=table_style, ensemble=True)
                print()

        elif dataset_name == 'ehr' or dataset_name == 'synthea':
            for group in [['source_db', 'target_db'],
                          ['source_db', 'source_table', 'source_cols',  # 'source_rows', 'source_cols',
                           'target_db', 'target_table', 'target_cols',  # 'target_rows', 'target_cols',
                           'gt_size']]:
                print(f'----- {group}:')
                print_aggregates(result_dfs, grouping=group, table_style=table_style, ensemble=True)
                print()

        elif dataset_name == 'bird':
            for group in [['domain'], ['domain', 'source_db', 'target_db'],
                          ['source_db', 'source_table', 'source_cols',  # 'source_rows', 'source_cols',
                           'target_db', 'target_table', 'target_cols',  # 'target_rows', 'target_cols',
                           'gt_size']]:
                print(f'----- {group}:')
                print_aggregates(result_dfs, grouping=group, table_style=table_style, ensemble=True)
                print()

        sys.stdout = original_stdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--res_path', dest='res_path', type=str,
                        help="the path to the RESULTS_MODEL- JSON file to evaluate OR the path to a directory "
                             "containing multiple RESULTS_MODEL- JSON files to evaluate")
    parser.add_argument('--mode', type=str, choices=["intersection", "union","disjoint","majority_vote","match_gold_mappings"], default="union",)
    parser.add_argument('--aggr_seeds', type=int, nargs='+', help="List of seed values")

    args = parser.parse_args()
    res_path = os.path.normpath(args.res_path)
    ensemble_mode = args.mode

    seeds = []
    result_files = []
    question_files = []

    files_excluded = []

    if REPR_TYPE.TaDa in res_path:
        isTaDa = True
        ensemble_mode = "majority_vote"
    else:
        isTaDa = False
    # --res_path is a directory
    if os.path.isdir(res_path):
        base_dir = res_path
        for file in os.listdir(res_path):

        # for file in os.listdir(args.res_path):
        #     evalTaDa = isTaDa and file.startswith("RESULTS")
            if not file.startswith("AN-RESULTS"):
                continue
            res_file = file

            seed = get_seed(res_file)
            if isTaDa:
                quest_file = 'questions.json'
                quest_path = os.path.join(args.res_path, f'questions.json')

            else:
                quest_file = f'questions-s-{seed}.json'
                quest_path = os.path.join(args.res_path, f'questions-s-{seed}.json')


                if seed in seeds:
                    files_excluded.append((f'{res_file}', 'File with duplicate seed found.'))
                    continue

                if not os.path.exists(quest_path):
                    files_excluded.append((f'{res_file}', f'"question-s-{seed}" not found.'))
                    continue

            if int(seed) in args.aggr_seeds:
                seeds.append(seed)
                result_files.append(os.path.join(base_dir, res_file))
                question_files.append(os.path.join(base_dir, quest_file))

    else:  # If --res_path is a file
        base_dir, res_file = os.path.split(res_path)
        seed = get_seed(res_file)
        if seed is None:
            quest_file = f'questions.json'
        else:
            quest_file = f'questions-s-{seed}.json'
        quest_path = os.path.join(base_dir, quest_file)

        if not os.path.exists(res_path):
            print(f'{res_path} not found')
        if not os.path.exists(quest_path):
            print(f'{quest_file} not found')

        result_files = [os.path.join(base_dir, res_file)]
        question_files = [os.path.join(base_dir, quest_file)]

    if len(args.aggr_seeds) == len(seeds):
        print("Length of aggr_seeds and seeds match.")
    else:
        print(seeds)
        raise ValueError(f"Length mismatch: aggr_seeds ({len(args.aggr_seeds)}) != seeds ({len(seeds)})")

    sample_num = len(args.aggr_seeds)
    ensemble_dir = os.path.join(args.res_path, f'{sample_num}ensemble')
    os.makedirs(ensemble_dir, exist_ok=True)

    dataset_name = get_dataset_name(res_path)
    results = ensemble_results(result_files, ensemble_mode,isTaDa)

    result_dfs=[get_ensemble_results_df(dataset_name, ensemble_dir, results, question_files[0], ensemble_mode)]


    output_txt_path = os.path.join(ensemble_dir, f'{ensemble_mode}_ensemble_results.txt')
    excel_file_path = os.path.join(ensemble_dir, f'{ensemble_mode}_dataset_wide_results.xlsx')

    eval_results(result_dfs, output_txt_path, excel_file_path, dataset_name, base_dir)

    print(f"Results and outputs saved to {output_txt_path}")
