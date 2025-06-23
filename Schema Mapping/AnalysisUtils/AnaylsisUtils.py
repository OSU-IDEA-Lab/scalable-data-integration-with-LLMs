import numpy as np
import pandas as pd
from scipy import stats

from AnalysisUtils.Experiment import OverlapResults
from AnalysisUtils.ExperimentCollection import ExperimentCollection
from DatabaseUtils.ArgHelper import ArgHelper

def compute_prf1(tp, fn, fp):

    if tp + fn + fp == 0:
        return 1.0, 1.0, 1.0

    if tp + fn == 0:
        recall = 0.
    else:
        recall = tp / (tp + fn)
    if tp + fp == 0:
        precision = 0.
    else:
        precision = tp / (tp + fp)

    if recall + precision == 0:
        f1 = 0.
    else:
        f1 = (2. * recall * precision) / (recall + precision)

    return precision, recall, f1

def get_dfs(experiment_collection: ExperimentCollection, drop_static_arg_columns: bool = True,
            ignore_paths_containing: str = None) \
        -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    def _get_rows(static_cols: dict, key_to_overlap_results: dict[str, OverlapResults]) -> list[dict]:
        rows = []

        for test_key, overlap in key_to_overlap_results.items():
            prec, rec, f1 = compute_prf1(tp=overlap.overlap_count.TP, fn=overlap.overlap_count.FN,
                                         fp=overlap.overlap_count.FP)
            test_level_info = {"Test": test_key, "Precision": prec, "Recall": rec, "F1": f1}

            rows.append(static_cols | test_level_info)

        return rows

    def _get_static_cols(df: pd.DataFrame) -> list[str]:
        static_cols = []
        for col in arg_helper.get_option_names():
            if df[col].nunique(dropna=False) == 1:
                static_cols.append(col)
        return static_cols

    def _to_dataframe(records: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame.from_records(records)
        df["hints"] = df["hints"].apply(str)
        return df

    # Generate dataframes where each row corresponds to the result of an overlap test.

    arg_helper = ArgHelper()
    collection = experiment_collection.filepath_to_experiment

    '''
        Calculate Overlap
    '''
    table_overlap_rows = []
    join_overlap_rows = []
    rule_table_overlap_rows = []
    rule_join_overlap_rows = []
    for filepath in collection.keys():

        if ignore_paths_containing is not None and ignore_paths_containing in filepath:
            continue

        args = arg_helper.parse_args_from_filepath(filepath)
        experiment_level_info = args

        table_overlap_rows += _get_rows(experiment_level_info, collection[filepath].mapping_results.table_overlap)
        join_overlap_rows += _get_rows(experiment_level_info, collection[filepath].mapping_results.join_overlap)

        if collection[filepath].prompt_config["clusterings"] == "ground_truth":
            for rule_id, results in collection[filepath].rule_results_by_id.items():

                rule_level_info = experiment_level_info | {"rule": rule_id}

                rule_table_overlap_rows += _get_rows(rule_level_info, results.table_overlap)
                rule_join_overlap_rows += _get_rows(rule_level_info, results.join_overlap)

    table_overlap_df = _to_dataframe(table_overlap_rows)
    join_overlap_df = _to_dataframe(join_overlap_rows)
    if len(rule_table_overlap_rows) > 0:
        rule_table_overlap_df = _to_dataframe(rule_table_overlap_rows)
        rule_join_overlap_df = _to_dataframe(rule_join_overlap_rows)

    if drop_static_arg_columns:
        static_cols = _get_static_cols(table_overlap_df)
        table_overlap_df.drop(static_cols, axis=1, inplace=True)
        join_overlap_df.drop(static_cols, axis=1, inplace=True)

        if len(rule_table_overlap_rows) > 0:
            static_cols = _get_static_cols(rule_table_overlap_df)
            rule_table_overlap_df.drop(static_cols, axis=1, inplace=True)
            rule_join_overlap_df.drop(static_cols, axis=1, inplace=True)

    return table_overlap_df, join_overlap_df, rule_table_overlap_df, rule_join_overlap_df

def ci_lower(data, confidence=0.95):
    """Calculate confidence interval for a dataset."""
    mean = np.mean(data)
    sem = stats.sem(data)
    interval = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return max(mean - interval, 0.0)

def ci_upper(data, confidence=0.95):
    """Calculate confidence interval for a dataset."""
    mean = np.mean(data)
    sem = stats.sem(data)
    interval = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return min(mean + interval, 1.0)

def tests_passed(column: pd.Series):
    return len(column[column>=1]) / len(column)

def get_max_pass_rate_mapping(df: pd.DataFrame):
    group_by_args = get_arg_grouping(df)

    # Get tests_passed [agg over args and seeds]
    avg_each_exp_df = df.groupby(group_by_args, dropna=False).agg(avg_Prec=("Precision", "mean"),
                                                    avg_Recall=("Recall", "mean"),
                                                    avg_F1=("F1", "mean"),
                                                    frac_perfect_precision=("Precision", tests_passed),
                                                    frac_perfect_recall=("Recall", tests_passed))
    group_by_args.remove("seed")
    avg_each_exp_df= avg_each_exp_df.groupby(group_by_args, dropna=False).agg(max_avg_F1=("avg_F1", "max"),
                                                                max_frac_perfect_precision=("frac_perfect_precision", "max"),
                                                                max_frac_perfect_recall=("frac_perfect_recall", "max"))

    avg_each_exp_df = avg_each_exp_df.round(2)

    # print((avg_each_exp_df).to_csv(sep="\t"))

    return avg_each_exp_df

def get_max_pass_rate_rule(df: pd.DataFrame):
    group_by_args = get_arg_grouping(df)
    group_by_args.remove("seed")
    group_by_args.append("rule")

    # Get tests_passed [agg over args and seeds]
    avg_each_exp_df = df.groupby(group_by_args, dropna=False).agg(avg_Prec=("Precision", "mean"),
                                                    avg_Recall=("Recall", "mean"),
                                                    avg_F1=("F1", "mean"),
                                                    frac_perfect_precision=("Precision", tests_passed),
                                                    frac_perfect_recall=("Recall", tests_passed))

    group_by_args.remove("rule")

    avg_each_exp_df= avg_each_exp_df.groupby(group_by_args, dropna=False).agg(max_avg_F1=("avg_F1", "max"),
                                                                max_frac_perfect_precision=("frac_perfect_precision", "max"),
                                                                max_frac_perfect_recall=("frac_perfect_recall", "max"))

    avg_each_exp_df = avg_each_exp_df.round(2)

    # print((avg_each_exp_df).to_csv(sep="\t"))

    return avg_each_exp_df

def get_arg_grouping(df: pd.DataFrame) -> list[str]:
    arg_helper = ArgHelper()
    return [arg for arg in arg_helper.get_option_names() if arg in df.columns]

def agg_by_args(df: pd.DataFrame) -> pd.DataFrame:
    group_by_args = get_arg_grouping(df)

    avg_each_exp_df = df.groupby(group_by_args, dropna=False).agg({"Precision": ["mean", ci_lower, ci_upper, "max"],
                                   "Recall": ["mean", ci_lower, ci_upper, "max"],
                                   "F1": ["mean", ci_lower, ci_upper, "max"]})

    group_by_args.remove("seed")
    avg_all_exp_df = avg_each_exp_df.copy()
    avg_all_exp_df.columns = ["_".join(col) for col in avg_each_exp_df.columns]
    avg_all_exp_df = avg_all_exp_df.reset_index()

    avg_all_exp_df = avg_all_exp_df.groupby(group_by_args, dropna=False).agg(
        {"seed": ["nunique"], "Precision_mean": ["mean", ci_lower, ci_upper],
         "Recall_mean": ["mean", ci_lower, ci_upper],
         "F1_mean": ["mean", ci_lower, ci_upper, "max"]})

    return avg_each_exp_df.round(2), avg_all_exp_df.round(2)

def agg_by_args(df: pd.DataFrame) -> pd.DataFrame:
    group_by_args = get_arg_grouping(df)

    avg_each_exp_df = df.groupby(group_by_args, dropna=False).agg({"Precision": ["mean", ci_lower, ci_upper, "max"],
                                   "Recall": ["mean", ci_lower, ci_upper, "max"],
                                   "F1": ["mean", ci_lower, ci_upper, "max"]})

    group_by_args.remove("seed")
    avg_all_exp_df = avg_each_exp_df.copy()
    avg_all_exp_df.columns = ["_".join(col) for col in avg_each_exp_df.columns]
    avg_all_exp_df = avg_all_exp_df.reset_index()

    avg_all_exp_df = avg_all_exp_df.groupby(group_by_args, dropna=False).agg(
        {"seed": ["nunique"], "Precision_mean": ["mean", ci_lower, ci_upper],
         "Recall_mean": ["mean", ci_lower, ci_upper],
         "F1_mean": ["mean", ci_lower, ci_upper, "max"]})

    return avg_each_exp_df.round(2), avg_all_exp_df.round(2)


