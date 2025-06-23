import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.enums import REPR_TYPE
import argparse

# Add the root directory to the Python path so that the 'eval' module can be found

from scipy import stats

from utils.data_builder import get_schemas_for_id, get_seed

from prompt.InstanceDataHandler import *
from eval.eval_utils import *
from eval.eval_utils import _remove_nulls
from eval.computational_eval import print_computational_cost


def print_formated_s(row_name, l, element_format, header_format='{:15}'):
    template = header_format + " " + ' '.join([element_format] * len(l))
    print(template.format(row_name, *l))


class OUTPUT_FORMAT:
    VIEW = 0
    JSON = 1
    BASELINE = 2


def categorize_valentine(case, pred_dict, quest_dict):
    dataset, problem_type, sub_problem = case['id'].split('/')
    vertical_overlap = None
    horizontal_overlap = None
    schema_noise = None
    data_noise = None
    if not (dataset == 'Wikidata' or dataset == 'Magellan'):
        parts = sub_problem.split('_')[1:]
        allocation = parts.pop(0)
        if allocation == 'both':
            horizontal_overlap = int(parts.pop(0))
            vertical_overlap = int(parts.pop(0))
        elif allocation == 'vertical':
            vertical_overlap = int(parts.pop(0))
            horizontal_overlap = 100
        else:
            assert allocation == 'horizontal', 'Only option left is horizontal. Something wrong with parsing'
            vertical_overlap = 100
            horizontal_overlap = int(parts.pop(0))
        schema_noise = parts.pop(0)
        data_noise = parts.pop(0)

    pred_dict['dataset'].append(dataset)
    pred_dict['problem_type'].append(problem_type)
    pred_dict['vertical_overlap'].append(vertical_overlap)
    pred_dict['horizontal_overlap'].append(horizontal_overlap)
    pred_dict['schema_noise'].append(schema_noise)
    pred_dict['data_noise'].append(data_noise)


def categorize_ehr(case, pred_dict, quest_dict):
    source_db, source_table, target_db, target_table = parse_ehr_id_parts(case['id'])

    relevant_quest = quest_dict[case['id']]

    pred_dict['source_db'].append(source_db)
    pred_dict['source_table'].append(source_table)
    # pred_dict['source_rows'].append(case['source_rows'])
    pred_dict['source_cols'].append(len(relevant_quest['source_schema']['columns']))

    pred_dict['target_db'].append(target_db)
    pred_dict['target_table'].append(target_table)
    # pred_dict['target_rows'].append(case['target_rows'])
    pred_dict['target_cols'].append(len(relevant_quest['target_schema']['columns']))

    pred_dict['gt_size'].append(len(relevant_quest['gold_mapping']))


def categorize_bird(case, pred_dict, quest_dict):
    domain, source_db, source_table, target_db, target_table = parse_bird_id_parts(case['id'])

    relevant_quest = quest_dict[case['id']]

    pred_dict['domain'].append(domain)

    pred_dict['source_db'].append(source_db)
    pred_dict['source_table'].append(source_table)
    # pred_dict['source_rows'].append(case['source_rows'])
    pred_dict['source_cols'].append(len(relevant_quest['source_schema']['columns']))

    pred_dict['target_db'].append(target_db)
    pred_dict['target_table'].append(target_table)
    # pred_dict['target_rows'].append(case['target_rows'])
    pred_dict['target_cols'].append(len(relevant_quest['target_schema']['columns']))

    pred_dict['gt_size'].append(len(relevant_quest['gold_mapping']))


# def get_results_df(pred_file, quest_file):
def get_stats(stats):
    if stats is None:
        return {'same_schema_count': np.nan,
                'same_attribute_count': np.nan,
                'mapping_errors': np.nan,
                'mapping_errors_cnt': np.nan,
                'invalid_attribute_count': np.nan}
    return stats


def gather_n_prompts(isTaDa, test_case, source_schema, target_schema):
    if isTaDa:
        test_case['valid_yes'] = {}
        test_case['valid_no'] = {}
    else:
        test_case['valid_predicted_mappings'] = {}
    predicted_errors, predicted_alignments, stats = [], [], {
        'mapping_errors': [],
        'mapping_errors_cnt': 0,
        'same_schema_count': 0,
        'same_attribute_count': 0,
        'invalid_attribute_count': 0
    }

    for attribute in test_case['predicted_mapping'].keys():
        predicted_errors_i, predicted_alignments_i, stats_i = parse_alignments_from_JSON(
            test_case['predicted_mapping'][attribute], source_schema, target_schema, nth_attr=nth_attr,
            attr=attribute.lower(), isTaDa=isTaDa)

        predicted_errors.extend(predicted_errors_i)

        # if isTaDa:
        #     test_case['valid_yes'][attribute] = predicted_alignments_i["yes"]
        #     test_case['valid_no'][attribute] = predicted_alignments_i["no"]

        # Extend lists for errors and alignments
        if predicted_alignments_i is not None:
            if isTaDa:
                test_case['valid_yes'][attribute] = predicted_alignments_i["yes"]
                test_case['valid_no'][attribute] = predicted_alignments_i["no"]
                predicted_alignments.extend(predicted_alignments_i["yes"])

            else:
                predicted_alignments.extend(predicted_alignments_i)
                test_case['valid_predicted_mappings'][attribute] = predicted_alignments_i  # dictionaru for tada
        else:
            if isTaDa:
                test_case['valid_yes'][attribute] = []
                test_case['valid_no'][attribute] = []
            else:
                test_case['valid_predicted_mappings'][attribute] = []

        if stats_i is not None:
            # Filter out 'None' entries in mapping errors
            filtered_mapping_errors = [error for error in stats_i['mapping_errors'] if 'none' not in error]

            # Aggregate stats
            stats['mapping_errors'].extend(filtered_mapping_errors)
            stats['mapping_errors_cnt'] += len(filtered_mapping_errors)
            stats['same_schema_count'] += stats_i['same_schema_count']
            stats['same_attribute_count'] += stats_i['same_attribute_count']
            stats['invalid_attribute_count'] += stats_i['invalid_attribute_count']

    return test_case, predicted_errors, predicted_alignments, stats


def get_results_df(isTaDa, pred_file, dataset_name, quest_file=None, nth_attr=0):
    """
        Load gold and predicted files
    """
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)

    '''
        Determine how to parse the output of LLM based on directory name
    '''
    dir_path, _ = os.path.split(pred_file)
    _, dir = os.path.split(dir_path)
    if 'source_target' in dir or 'view_null' in dir:
        output_format = OUTPUT_FORMAT.VIEW
    # elif 'json' in dir or 'NL' in dir:
    #     output_format = OUTPUT_FORMAT.JSON
    elif 'baseline' in dir:

        if isTaDa:
            output_format = OUTPUT_FORMAT.JSON
        else:
            output_format = OUTPUT_FORMAT.BASELINE
    else:
        output_format = OUTPUT_FORMAT.JSON

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
                   'accuracy': [],
                   'accuracy2': [],
                   'error': [],
                   'mapping_errors_cnt': [],
                   'mapping_errors': [],
                   'same_schema_count': [],
                   'same_attribute_count': [],
                   'invalid_attribute_count': []}

    if dataset_name == 'valentine':
        column_dict['dataset'] = []
        column_dict['problem_type'] = []
        column_dict['vertical_overlap'] = []
        column_dict['horizontal_overlap'] = []
        column_dict['schema_noise'] = []
        column_dict['data_noise'] = []
    elif dataset_name == 'ehr' or dataset_name == "synthea":
        column_dict['source_db'] = []
        column_dict['source_table'] = []
        # column_dict['source_rows'] = []
        column_dict['source_cols'] = []
        column_dict['target_db'] = []
        column_dict['target_table'] = []
        # column_dict['target_rows'] = []
        column_dict['target_cols'] = []
        column_dict['gt_size'] = []
    elif dataset_name == 'bird':
        column_dict['domain'] = []
        column_dict['source_db'] = []
        column_dict['source_table'] = []
        # column_dict['source_rows'] = []
        column_dict['source_cols'] = []
        column_dict['target_db'] = []
        column_dict['target_table'] = []
        # column_dict['target_rows'] = []
        column_dict['target_cols'] = []
        column_dict['gt_size'] = []

    """
        Iterate through pairs
    """
    n_prompts, swapped, stable_matching = False, False, False
    if output_format != OUTPUT_FORMAT.BASELINE:
        test_cases = pred_data['eval']
        config_ = pred_data["config"]
        try:
            n_prompts = pred_data["config"]["n_prompts"]
        except (AttributeError, TypeError):
            n_prompts = False

        if "swap_T" in pred_data["config"]["question"]:
            swapped = True

        if "stable-matching" in pred_data["config"]:
            stable_matching = True
    else:
        test_cases = pred_data
        config_ = []

    for i, test_case in enumerate(test_cases):
        if (i + 1) % 100 == 0:
            print('Evaluating %dth prediction' % (i + 1))

        '''
            Get mappings/alignments
        '''
        source_schema, target_schema = get_schemas_for_id(test_case, dataset_name)
        if swapped:
            source_schema, target_schema = target_schema, source_schema
        if 'gold_mapping' in test_case:
            if 'matches' in test_case['gold_mapping']:
                gold_alignments = test_case['gold_mapping']['matches']
            else:
                gold_alignments = test_case['gold_mapping']
            try:
                gold_alignments = [(s.lower(), t.lower()) for s, t in gold_alignments]
            except:
                print()
        else:
            # For old experiment files
            gold_errors, gold_alignments, gold_stats = parse_alignments_from_JSON(test_case['gold_sql'], source_schema,
                                                                                  target_schema)
            assert len(gold_errors) == 0, f"{test_case['id']}: gold (ground truth) is not valid"
            assert not all(value == 0 for value in
                           gold_stats.values()), f"{test_case['id']}: stats show gold (ground truth) is not valid : {gold_stats} "

        if output_format == OUTPUT_FORMAT.VIEW:
            stats = None  # not supported
            predicted_errors, predicted_alignments = parse_alignments_from_view(test_case['predicted_mapping'])
            if len(predicted_errors) == 0:
                predicted_alignments = _remove_nulls(predicted_alignments)
        elif output_format == OUTPUT_FORMAT.JSON:
            if not n_prompts:
                if stable_matching:
                    stats = None  # not supported
                    predicted_errors = []
                    predicted_alignments = set(tuple(pair) for pair in test_case['predicted_mapping'])

                else:
                    predicted_errors, predicted_alignments, stats = parse_alignments_from_JSON(
                        test_case['predicted_mapping'],
                        source_schema, target_schema)
                    if predicted_alignments is not None:
                        test_case['valid_predicted_mappings'] = list(predicted_alignments)
                    else:
                        test_case['valid_predicted_mappings'] = []

            else:
                test_case, predicted_errors, predicted_alignments, stats = gather_n_prompts(isTaDa, test_case,
                                                                                            source_schema,
                                                                                            target_schema)
                print(predicted_errors)



        elif output_format == OUTPUT_FORMAT.BASELINE:
            stats = None  # not supported
            predicted_errors = []
            predicted_alignments = ((alignment['source'].lower(), alignment['target'].lower())
                                    for alignment in test_case['predicted_mapping'])

        '''
            Score mappings/alignments
        '''

        if stats is not None and len(stats['mapping_errors']) > 0:
            print(f"mapping_errors for {test_case['id']}: {stats['mapping_errors']}\n")
        if predicted_alignments is None:
            # if len(predicted_errors) > 0:
            print(f"{test_case['id']}: {predicted_errors}")
            tp_alignments, fp_alignments, fn_alignments = 0, 0, 0
        else:

            # Test for the correct alignment
            predicted_alignments = set(predicted_alignments)
            gold_alignments = set(gold_alignments)

            tp_alignments = len(gold_alignments.intersection(predicted_alignments))
            fp_alignments = len(predicted_alignments.difference(gold_alignments))
            fn_alignments = len(gold_alignments.difference(predicted_alignments))

        p, r, f1, accuracy, accuracy2, e = compute_prf1e(tp=tp_alignments, fp=fp_alignments, fn=fn_alignments, test_case=test_case, dataset_name=dataset_name)

        mappings_stats = get_stats(stats)

        test_case['precision'] = p
        test_case['recall'] = r
        test_case['f1'] = f1
        test_case['effort'] = e
        test_case['accuracy'] = accuracy
        test_case['accuracy2'] = accuracy2
        test_case['fp'] = fp_alignments
        test_case['fn'] = fn_alignments
        test_case['errors'] = predicted_errors
        test_case['gold_mapping'] = str(test_case['gold_mapping'])

        test_case.update(mappings_stats)
        annotated_eval.append(test_case)

        if output_format != OUTPUT_FORMAT.BASELINE:
            with open(quest_file, 'r') as f:
                quest_data = {entry['id']: entry for entry in json.load(f)}
            # Categorize this test case
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
        column_dict['error'].append(len(predicted_errors))

        column_dict['same_schema_count'].append(mappings_stats['same_schema_count'])
        column_dict['same_attribute_count'].append(mappings_stats['same_attribute_count'])
        column_dict['invalid_attribute_count'].append(mappings_stats['invalid_attribute_count'])
        column_dict['mapping_errors'].append(mappings_stats['mapping_errors'])
        column_dict['mapping_errors_cnt'].append(mappings_stats['mapping_errors_cnt'])

    path, ann_file_name = os.path.split(pred_file)
    with open(os.path.join(path, 'AN-' + ann_file_name), "w") as file:
        json.dump({'config': config_, 'eval': annotated_eval}, file, indent=4)

    return pd.DataFrame(column_dict)


def print_aggregates(dataframe_list, grouping=None, table_style='plain', ensemble=False):
    '''
        Takes a list of columns to group by (grouping). If no grouping is provided, will just aggregate over all rows.
        Otherwise, produces aggregates for that group in dataframe_list. If data_frame list contains multiple dateframes
        and confidence_interval is True (default), then also print the confidence interval across dataframes.
    '''

    def _get_agg(df, ensemble=False):
        if ensemble:
            return df.agg(
                precision=('precision', 'mean'),
                recall=('recall', 'mean'),
                f1=('f1', 'mean'),
                # fp=('fp', 'sum'),
                # fn=('fn', 'sum'),
                # accuracy2=('accuracy2', 'mean'),
                accuracy=('accuracy2', 'mean'),
                # accuracy=('accuracy', 'mean')
                # effort=('effort', 'mean')
            )
        return df.agg(
            precision=('precision', 'mean'),
            recall=('recall', 'mean'),
            f1=('f1', 'mean'),
            # fp=('fp', 'sum'),
            # fn=('fn', 'sum'),
            # effort=('effort', 'mean'),
            # accuracy=('accuracy', 'mean'),
            accuracy=('accuracy2', 'mean'),
            # accuracy2=('accuracy2', 'mean'),
            errors_total=('error', 'sum'),
            errors_pct=('error', 'mean'),
            mapping_errors_total=('mapping_errors_cnt', lambda x: x.dropna().sum())
        )

    agg_list = []
    for df in dataframe_list:
        if grouping is None:
            df = df.groupby(lambda x: True)
        else:
            df = df.groupby(grouping)
        agg_list.append(_get_agg(df, ensemble))

    if len(agg_list) == 1:
        print(agg_list[0].to_markdown(tablefmt=table_style))
        if grouping is None:
            return agg_list[0]
    else:
        # ToDo: add confidence interval calculations
        # print('Aggregating multiple files is not yet supported...')
        # pd.concat(agg_list)
        # Stack the dataframes into a 3D numpy array (rows x cols x dataframes)
        data_array = np.array([df.values for df in agg_list])

        # Calculate the mean and standard deviation for each cell (over the last axis)
        means = np.mean(data_array, axis=0)
        stds = np.std(data_array, axis=0, ddof=1)  # Using ddof=1 for sample std dev
        n = len(agg_list)  # Number of dataframes

        # Set your desired confidence level (e.g., 95%)
        confidence_level = 0.95
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)

        # Calculate the margin of error
        margin_of_error = z * (stds / np.sqrt(n))

        # Fill the dataframe with "mean ± margin_of_error" strings
        mean_error_df = pd.DataFrame(columns=agg_list[0].columns, index=agg_list[0].index)
        for row in range(means.shape[0]):
            for col in range(means.shape[1]):
                mean_value = means[row, col]
                margin_value = margin_of_error[row, col]
                mean_error_df.iloc[row, col] = f"{mean_value:.2f} ± {margin_value:.2f}"

        print(mean_error_df.to_markdown(tablefmt=table_style))
        if grouping is None:
            return mean_error_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--res_path', dest='res_path', type=str,
                        help="the path to the RESULTS_MODEL- JSON file to evaluate OR the path to a directory "
                             "containing multiple RESULTS_MODEL- JSON files to evaluate")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Specify the dataset name. If not provided, it will be inferred."
    )
    args = parser.parse_args()
    res_path = os.path.normpath(args.res_path)

    seeds = []
    result_files = []
    question_files = []

    files_excluded = []
    if REPR_TYPE.TaDa in res_path:
        isTaDa = True
    else:
        isTaDa = False

    # --res_path is a directory
    if os.path.isdir(res_path):
        for file in os.listdir(args.res_path):
            if not file.startswith("RESULTS"):
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

            seeds.append(seed)
            result_files.append(res_file)
            question_files.append(quest_file)

        base_dir = res_path
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

        result_files = [res_file]
        question_files = [quest_file]

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

    result_dfs = []
    if args.dataset_name is not None:
        dataset_name = args.dataset_name
    else:
        dataset_name = get_dataset_name(res_path)
    nth_attr = 0
    for idx, res_file in enumerate(result_files):
        if "one_to_n" in base_dir:
            nth_attr = 1  # source attribute should match with the key
        if "n_to_one" in base_dir or isTaDa:
            nth_attr = 2  # target attribute should match with the key

        result_dfs.append(get_results_df(isTaDa, os.path.join(base_dir, res_file), dataset_name,
                                         os.path.join(base_dir, question_files[idx]), nth_attr=nth_attr))
        # result_dfs.append(get_results_df(os.path.join(base_dir, res_file),
        #          os.path.join(base_dir, question_files[idx])))

    '''
        Print results here
    '''
    table_style = "tsv"  # See available styles here: https://pypi.org/project/tabulate/
    print('----- Dataset-wide:')
    eval_df = print_aggregates(result_dfs, grouping=None, table_style=table_style)
    excel_file_path = os.path.join(base_dir, 'dataset_wide_results.xlsx')
    eval_df.to_excel(excel_file_path, index=False)
    print()

    if dataset_name == 'valentine':
        for group in ['dataset', 'problem_type', 'schema_noise', 'data_noise']:
            print(f'----- {group}:')
            print_aggregates(result_dfs, grouping=[group], table_style=table_style)
            print()

        # print_computational_cost(result_files, question_files, seeds, base_dir)

    elif dataset_name == 'ehr' or dataset_name == 'synthea':
        for group in [['source_db', 'target_db'],
                      ['source_db', 'source_table', 'source_cols',  # 'source_rows', 'source_cols',
                       'target_db', 'target_table', 'target_cols',  # 'target_rows', 'target_cols',
                       'gt_size']]:
            print(f'----- {group}:')
            print_aggregates(result_dfs, grouping=group, table_style=table_style)
            print()

    elif dataset_name == 'bird':
        for group in [['domain'], ['domain', 'source_db', 'target_db'],
                      ['source_db', 'source_table', 'source_cols',  # 'source_rows', 'source_cols',
                       'target_db', 'target_table', 'target_cols',
                       # 'target_rows', 'target_cols',
                       'gt_size']]:
            print(f'----- {group}:')
            print_aggregates(result_dfs, grouping=group, table_style=table_style)
            print()
