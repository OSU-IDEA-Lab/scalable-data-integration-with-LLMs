import json
import os

import pandas as pd


def is_in_the_options(gold_matches, options, attr):
    match_options = [[o, attr] for o, _ in options]
    return bool(set(map(tuple, gold_matches)).intersection(set(map(tuple, match_options))))


def evaluate_none_of_options(prefs, score_0, score_100, gold_mapping):
    tp, fp = 0, 0
    fn, tn = 0, 0

    gold_attributes = {mapping[1]: mapping[0] for mapping in gold_mapping}

    for prediction, attribute in score_100:
        if prediction == "None of the options":
            if attribute not in gold_attributes:
                tp += 1
            elif attribute in gold_attributes:
                if is_in_the_options(gold_mapping, prefs[attribute], attribute):
                    fp += 1
                else:
                    tp += 1

    for prediction, attribute in score_0:
        if prediction == "None of the options":
            if attribute in gold_attributes:
                if is_in_the_options(gold_mapping, prefs[attribute], attribute):
                    tn += 1
                else:
                    fn += 1

            elif attribute not in gold_attributes:
                fn += 1

    total_cases = tp + fp + fn + tn

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision_ = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy_ = (tp + tn) / total_cases if total_cases > 0 else 0.0
    f1_score = (2 * precision_ * recall) / (precision_ + recall) if (precision_ + recall) > 0 else 0.0

    return recall, precision_, accuracy_, f1_score


# def evaluate_confidence_score(prefs, score_0, score_100, gold_mapping):
#     tp, fp = 0, 0
#     fn, tn = 0, 0
#     gold_m = {tuple(mapping) for mapping in gold_mapping}
#     gold_attributes = {mapping[1]: mapping[0] for mapping in gold_mapping}
#
#     for prediction, attribute in score_0:
#         if prediction == "None of the options":
#             if attribute in gold_attributes:
#                 if is_in_the_options(gold_mapping, prefs[attribute], attribute):
#                     tn += 1
#                 else:
#                     fn += 1
#
#             elif attribute not in gold_attributes:
#                 fn += 1
#
#         elif (prediction, attribute) in gold_m:
#             fn += 1
#         else:
#             tn += 1
#
#     for prediction, attribute in score_100:
#         if prediction == "None of the options":
#             if attribute not in gold_attributes:
#                 tp += 1
#             elif attribute in gold_attributes:
#                 if is_in_the_options(gold_mapping, prefs[attribute], attribute):
#                     fp += 1
#                 else:
#                     tp += 1
#         elif (prediction, attribute) in gold_m:
#             tp += 1  # True Positive: correctly predicted match in gold mapping
#         else:
#             fp += 1  # False Positive: incorrect match in gold mapping
#
#     total_cases = tp + fp + fn + tn
#
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     precision_ = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     accuracy_ = (tp + tn) / total_cases if total_cases > 0 else 0.0
#     f1_score = (2 * precision_ * recall) / (precision_ + recall) if (precision_ + recall) > 0 else 0.0
#
#     return recall, precision_, accuracy_, f1_score
#

def evaluate_confidence_score(prefs, score_0, score_100, gold_mapping):
    tp, fp = 0, 0
    fn, tn = 0, 0
    gold_m = {tuple(mapping) for mapping in gold_mapping}
    gold_attributes = {mapping[1]: mapping[0] for mapping in gold_mapping}

    for prediction, attribute in score_0:
        if prediction == "None of the options":
            if attribute in gold_attributes:
                if is_in_the_options(gold_mapping, prefs[attribute], attribute):
                    tn += 1
                else:
                    fn += 1

            elif attribute not in gold_attributes:
                fn += 1

        elif (prediction, attribute) in gold_m:
            fn += 1
        else:
            tn += 1

    for prediction, attribute in score_100:
        if prediction == "None of the options":
            if attribute not in gold_attributes:
                tp += 1
            elif attribute in gold_attributes:
                if is_in_the_options(gold_mapping, prefs[attribute], attribute):
                    fp += 1
                else:
                    tp += 1
        elif (prediction, attribute) in gold_m:
            tp += 1  # True Positive: correctly predicted match in gold mapping
        else:
            fp += 1  # False Positive: incorrect match in gold mapping

    total_cases = tp + fp + fn + tn

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision_ = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy_ = (tp + tn) / total_cases if total_cases > 0 else 0.0
    f1_score = (2 * precision_ * recall) / (precision_ + recall) if (precision_ + recall) > 0 else 0.0

    return recall, precision_, accuracy_, f1_score

def evaluate_attr_options(score_0, score_100, gold_mapping):
    tp, fp = 0, 0
    fn, tn = 0, 0
    gold_m = {tuple(mapping) for mapping in gold_mapping}

    for prediction, attribute in score_0:
        if prediction == "None of the options":
            continue

        elif (prediction, attribute) in gold_m:
            fn += 1
        else:
            tn += 1

    for prediction, attribute in score_100:
        if prediction == "None of the options":
            continue
        elif (prediction, attribute) in gold_m:
            tp += 1  # True Positive: correctly predicted match in gold mapping
        else:
            fp += 1  # False Positive: incorrect match in gold mapping

    total_cases = tp + fp + fn + tn

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision_ = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy_ = (tp + tn) / total_cases if total_cases > 0 else 0.0
    f1_score = (2 * precision_ * recall) / (precision_ + recall) if (precision_ + recall) > 0 else 0.0

    return recall, precision_, accuracy_, f1_score


def evaluate_test_cases(test_cases, score_filter):
    results = []

    for test_case in test_cases["eval"]:
        score_0 = test_case['score_0']
        score_100 = test_case[score_filter]
        gold_mapping = test_case['gold_mapping']
        preferences = test_case['preferences']

        # Calculate each precision metric
        recall_none, precision_none, accuracy_none, f1_none = evaluate_none_of_options(preferences, score_0, score_100, gold_mapping)
        recall_attr, precision_attr, accuracy_attr, f1_attr = evaluate_attr_options(score_0, score_100, gold_mapping)
        recall_0_100, precision_0_100, accuracy_0_100, f1_0_100 = evaluate_confidence_score(preferences, score_0, score_100, gold_mapping)

        # Store results
        _stat = {
            "precision_none": precision_none,
            "recall_none": recall_none,
            "f1_none": f1_none,
            "accuracy_none": accuracy_none,
            "precision_attr": precision_attr,
            "recall_attr": recall_attr,
            "f1_attr": f1_attr,
            "accuracy_attr": accuracy_attr,
            "precision_0_100": precision_0_100,
            "recall_0_100": recall_0_100,
            "f1_0_100": f1_0_100,
            "accuracy_0_100": accuracy_0_100
        }
        test_case["score_stats"] = _stat
        results.append(_stat)

    eval_df = pd.DataFrame(results)

    # Calculate the mean values for each group (_none, _attr, _0_100)
    mean_none = eval_df.filter(regex='_none$').mean()
    mean_attr = eval_df.filter(regex='_attr$').mean()
    mean_0_100 = eval_df.filter(regex='_0_100$').mean()

    # Create a DataFrame with separate rows for each metric group
    mean_metrics = pd.DataFrame({
        "Metric Group": ["None of the options", "(attr, query)", "All"],
        "Precision": [mean_none['precision_none'], mean_attr['precision_attr'], mean_0_100['precision_0_100']],
        "Recall": [mean_none['recall_none'], mean_attr['recall_attr'], mean_0_100['recall_0_100']],
        "Accuracy": [mean_none['accuracy_none'], mean_attr['accuracy_attr'], mean_0_100['accuracy_0_100']],
        "F1 Score": [mean_none['f1_none'], mean_attr['f1_attr'], mean_0_100['f1_0_100']]
    })

    # Print and save the mean metrics as a Markdown table
    mean_metrics_markdown = mean_metrics.to_markdown(index=False, tablefmt="github")
    print(mean_metrics_markdown)

    with open("mean_metrics_report.txt", "w") as f:
        f.write(mean_metrics_markdown)

    return test_cases

    # mean_precision = eval_df.mean().to_frame(name="Mean Precision").T
    #
    # # Print and save the mean precision as a Markdown table
    # mean_precision_markdown = mean_precision.to_markdown(tablefmt="github")
    # print(mean_precision_markdown)
    #
    # with open("mean_precision_report.txt", "w") as f:
    #     f.write(mean_precision_markdown)
    #
    # return test_cases


if __name__ == "__main__":
    models = ["70"]
    ensemble_mode = "union"
    positive_100 = "score_100"
    # positive_100 = "score_positive"
    # scope = "name_type"
    scope = "cdesc_name_type"

    paths = {"ehr": f"model-swap-10-random_unique-reasoning-n_to_one_json_dynamic-cdesc_data_name_tdesc_type-800"
        # ,
        #      "bird": f"model-swap-10-random_unique-reasoning-n_to_one_json_dynamic-cdesc_data_name_type-800"
             }

    for dataset in paths:
        basedir = f"dataset/process/{dataset}"
        swap_state = ['swap_False', 'swap_True']
        for model in models:
            for state in swap_state:
                print('\n\n------------dataset: ', dataset, ", model : Llama3.1 ", model, ", ", state)
                path = paths[dataset].replace("model", model)
                task_folder = path.replace("swap", state)
                conf_score_path = os.path.join(basedir, task_folder, "confidence",
                                               f"{ensemble_mode}-{scope}-scores.json")
                conf_score_processed_path = os.path.join(basedir, task_folder, "confidence",
                                                         f"processed_{ensemble_mode}-{scope}-scores.json")
                with open(conf_score_processed_path, 'r') as f:
                    data = json.load(f)

                updated_data = evaluate_test_cases(data, positive_100)

                # Save updated JSON data back to file
                with open(conf_score_processed_path, 'w') as f:
                    json.dump(updated_data, f, indent=4)
