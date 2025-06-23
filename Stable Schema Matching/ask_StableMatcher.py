import copy
import os
import subprocess
import json
from copy import deepcopy
import ntpath
ntpath.realpath = ntpath.abspath

from eval.eval_MatchMaker import get_top_k, eval_result, eval_results

from stable_match import ManyToManyStableMatcher, lowercase_keys_and_values


def write_n_eval(folder, res, ques, res_path, ques_path, dataset_na, ensemble_mode, top_r=None):
    with open(res_path, 'w') as f:
        json.dump(res, f, indent=4)

    with open(ques_path, 'w') as f:
        json.dump(ques, f, indent=4)

    if top_r is None:
        out_path = f"{folder}/{ensemble_mode}-eval_results.txt"
    else:
        out_path = res_path.replace(".json",".txt").replace("avg-logits", "EV-avg-logits")

    # Create the output file
    with open(out_path, 'w') as f:
        # Run the evaluation script and capture the output
        terminal_result = subprocess.run(
            ["python3", "eval/eval_dataset.py", "--res_path", res_path, "--dataset_name", dataset_na],
            capture_output=True,
            text=True
        )
        print(terminal_result.stdout)
        f.write(terminal_result.stdout)

    # Read and return the JSON data from processed_score_path




def get_pref(score_path, processed_score_path, dataset_name, re_gen=False, is_logits=False):

    questions_ = get_question_file(score_path, is_logits)
    if is_logits or 'logits' in score_path:
        with open(score_path, 'r') as f:
            data = json.load(f)
        return data, questions_

    if os.path.exists(processed_score_path) and not re_gen:
        with open(processed_score_path, 'r') as f:
            data = json.load(f)
        return data, questions_

    try:
        result = subprocess.run(
            ["python3", "postprocess_confidence_scores.py", "--confidence_path", score_path, "--output_path", processed_score_path],
            capture_output=True,
            text=True,
            check=True  # This raises an error if the subprocess fails
        )
        print("Subprocess completed successfully.")
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Subprocess failed with error:", e.stderr)

    with open(processed_score_path, 'r') as f:
        data = json.load(f)

    scores_folder = os.path.dirname(score_path)
    scores_file_name = os.path.basename(score_path)
    score_results_path = os.path.join(scores_folder, f"Results-{scores_file_name}")
    score_questions_path = os.path.join(scores_folder, "questions.json")

    score_results = {"config": data["config"], "eval": []}
    score_results["config"]["n_prompts"] = False  # swap is false as well
    score_results["config"]["stable-matching"] = False  # swap is false as well

    n = len(data["eval"])
    for i in range(n):
        # print(f"test case {i}/{n}")
        res_entry = {
            "id": data["eval"][i]["id"],
            "gold_mapping": data["eval"][i]["gold_mapping"]
        }
        predicted_mapping = []

        for attr, preference_list in data["eval"][i]["filtered_preferences"].items():
            for option_value, score in preference_list:
                if option_value == "None of the options" and score == 100:
                    break
                predicted_mapping.append([option_value,attr])

        res_entry["predicted_mapping"] = predicted_mapping

        score_results["eval"].append(res_entry)

    write_n_eval(scores_folder, score_results, questions_, score_results_path, score_questions_path, dataset_name, ensemble_mode)
    print(f" >>> Confidence Score Evaluation Done.")



    return data, questions_


def get_question_file(score_path, is_logits=False):
    if 'cot_logits' in score_path:
        question_dir, _ = os.path.split(score_path.split("ensemble")[0])
    elif is_logits:
        question_dir, _ = os.path.split(score_path.split("logits")[0])
    else:
        question_dir, _ = os.path.split(score_path.split("confidence")[0])
    question_path = os.path.join(question_dir, "questions-s-7564.json")

    with open(question_path, 'r') as f:
        question = json.load(f)

    return question


def get_target_score(pref_swap_true, s_attr, t_attr):
    if s_attr in pref_swap_true:
        for match_s in pref_swap_true[s_attr]:
            if match_s[0] == t_attr:
                return match_s[1]
    return 0

def lower_keys(d):
    return {k.lower(): v for k, v in d.items()}

def save_avg_logits(pref_key, output_dir, mode , dataset, questions_path , swap_t , swap_f, agg, gstat = False):
    results = {"config": target_pref["config"], "eval": []}
    results["config"]["n_prompts"] = False  # swap is false as well
    results["config"]["stable-matching"] = True  # swap is false as well



        # entry["predicted_mapping"] = round_results
        # # entry["predicted_mapping"] = matches
        # results["eval"].append(entry)

    for i, test_case in enumerate(swap_f["eval"]):
        alignments = []
        avg_logits = []
        entry = {
            "id": test_case["id"],
            "gold_mapping": test_case["gold_mapping"],
            "predicted_mapping": {},
            pref_key: {}
        }
        test_case[pref_key] = lowercase_keys_and_values(test_case[pref_key])
        for t_attr, matches_scores in test_case[pref_key].items():
            for match_s in matches_scores:
                s_attr, s_score = match_s[0], match_s[1]
                swap_t["eval"][i][pref_key] = lowercase_keys_and_values(swap_t["eval"][i][pref_key])
                s_attr, t_attr = s_attr.lower(), t_attr.lower()
                t_score = get_target_score(swap_t["eval"][i][pref_key], s_attr, t_attr)
                if [s_attr, t_attr] not in alignments:
                    if agg == 'multiply':
                        s = s_score*t_score
                    else:
                        s = (s_score+t_score)/2
                    if s==0:
                        # print("ss ", t_attr, s_score, t_score, s)
                        continue
                    alignments.append([s_attr, t_attr])
                    avg_logits.append(s)

                    if t_attr not in entry[pref_key]:
                        entry[pref_key][t_attr] = []
                    entry[pref_key][t_attr].append([s_attr, s])


        swap_t["eval"][i][pref_key] = lowercase_keys_and_values(swap_t["eval"][i][pref_key])
        for s_attr, matches_scores in swap_t["eval"][i][pref_key].items():
            for match_s in matches_scores:
                t_attr, t_score = match_s[0], match_s[1]
                s_attr, t_attr = s_attr.lower(), t_attr.lower()
                if [s_attr, t_attr] not in alignments:
                    alignments.append([s_attr, t_attr])
                    s_score = 0

                    if agg == 'multiply':
                        s = s_score*t_score
                    else:
                        s = (s_score+t_score)/2
                    if s==0:
                        # print("ss ", t_attr, s_score, t_score, s)
                        continue
                    avg_logits.append(s)

                    if t_attr not in entry[pref_key]:
                        entry[pref_key][t_attr] = []
                    entry[pref_key][t_attr].append([s_attr, s])

        for t_attr in entry[pref_key]:
            entry[pref_key][t_attr].sort(key=lambda x: x[1], reverse=True)
            entry["predicted_mapping"][t_attr] = [[s_attr, t_attr] for s_attr, s in entry[pref_key][t_attr]]

        # print(entry[pref_key])
        # exit()
        results["eval"].append(entry)

    # if agg == 'multiply':
    #     result_json = f"multiply-{mode}-Results.json"
    # else:
    #     result_json = f"avg-{mode}-Results.json"
    result_json = f"{mode}-greedy-{gstat}.json"
    result_path = os.path.join(output_dir,result_json)

    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)

    for k in range(1, 6):
        output_path = os.path.join(output_dir, f'AN-{k}-' + result_json)
        result_k = get_top_k(result_path, k)

        result_dfs = [
            eval_result(result_k, dataset, questions_path, output_path)]

        output_txt_path = output_path.replace(".json", ".txt")
        excel_file_path = output_path.replace(".json", ".xlsx")

        eval_results(result_dfs, output_txt_path, excel_file_path, dataset)

        print(f"Results and outputs saved to {output_txt_path}")


if "__main__" == __name__:
    model = "70"
    # models = ["8", "70"]

    # 70 - swap_False - 10 - random_unique - reasoning - n_to_one_json_dynamic - cdesc_data_name_type - 800
    # 70 - swap_False - reasoning - n_to_one_json_dynamic - cdesc_name_type - 800
    # 70 - swap_True - 10 - random_unique - reasoning - n_to_one_json_dynamic - cdesc_data_name_type - 800
    # 70 - swap_True - reasoning - n_to_one_json_dynamic - cdesc_name_type - 800

    ensemble_modes = ["avg-logits"]
    # ensemble_modes = ["majority_vote"]
    reasoning_ = ["reasoning-"]
    # reasoning_ = ["", "reasoning-"]
    # prompt_repr = "n_to_one_json_dynamic"
    prompt_repr = "cot_logits_dynamic"
    # isDynamic = True
    isDynamic = False


    # ensemble_modes = ["majority_vote","intersection"]
    # sample_nums = [1]
    sample_nums = [4]
    greedy_stats = ["True"]
    # greedy_stats = ["True","False"]
    # greedy_stat = "False"
    seeds = [7564, 268799, 87849, 333]  # done
    # sample_nums = [3,4,5,6,7]
    # ensemble_modes = ["union","majority_vote","intersection"]
    # ensemble_modes = ["union"]
    re_gen = False
    k = float('inf')
    no_filter = False
    pref_key = "filtered_preferences"
    if no_filter:
        pref_key = "preferences"

    logits = False
    avg_logits = False
    # avg_logits = True
    aggr = 'multiply'
    # aggr = 'avg'
    if logits:
        pref_key = "preferences"
    elif prompt_repr=="cot_logits_dynamic":
        pref_key = "logits"

    for rn in reasoning_:
        for greedy_stat in greedy_stats:

            for ensemble_mode in ensemble_modes:
                paths = {
                        "synthea":
                            [f"{model}-swap-10-random_unique-{rn}{prompt_repr}-cdesc_data_name_tdesc_type-800"],
                                # [f"{model}-swap-reasoning-{prompt_repr}-cdesc_name_tdesc_type-800"],
                        "ehr":
                            [f"{model}-swap-10-random_unique-{rn}{prompt_repr}-cdesc_data_name_tdesc_type-800"],
                                # [f"{model}-swap-reasoning-{prompt_repr}-cdesc_name_tdesc_type-800"],
                        # "bird":
                        #     [f"{model}-swap-10-random_unique-{rn}{prompt_repr}-cdesc_data_name_type-800"]
                            #
                            # [f"{model}-swap-reasoning-{prompt_repr}-cdesc_name_type-800"]

                    # ,
                                # f"{model}-swap-reasoning-{prompt_repr}-cdesc_name_type-800"]
                         }
                # del paths["ehr"]
                for dataset in paths:
                    swapf = "swap_False"
                    swapt = 'swap_True'
                    for path in paths[dataset]:
                        basedir = f"dataset/process/{dataset}"

                        confidence_file = f"{ensemble_mode}-cdesc_name_type-scores.json"
                        logits_file = f"{ensemble_mode}-cdesc_name_type-logits.json"
                        #################################################################################################
                        # confidence_file = confidence_file.replace("cdesc_","")
                        processed_confidence_file = "processed_"+confidence_file
                        # n-to-one (source, target) prompt > one = target query pref

                        for sn in sample_nums:

                            sorted_seeds = [str(x) for x in sorted(seeds[:sn])]
                            seeds_str = '-s-' + '-'.join(sorted_seeds)
                            if prompt_repr == 'cot_logits_dynamic':
                                if sn!=1:
                                    if isDynamic:
                                        target_conf_score_path = os.path.join(basedir, path.replace("swap", swapf), "dynamic_ensemble", f"AN-greedy_{greedy_stat}_cot_logits{seeds_str}.json")
                                    else:
                                        target_conf_score_path = os.path.join(basedir, path.replace("swap", swapf), "ensemble", f"AN-greedy_{greedy_stat}_cot_logits{seeds_str}.json")
                                else:
                                    target_conf_score_path = os.path.join(basedir, path.replace("swap", swapf), f"AN-greedy_{greedy_stat}_cot_logits{seeds_str}.json")
                                target_conf_score_processed_path = None
                            elif not logits:
                                target_conf_score_path = os.path.join(basedir,path.replace("swap", swapf),"confidence",confidence_file)
                                target_conf_score_processed_path = os.path.join(basedir,path.replace("swap", swapf),"confidence",processed_confidence_file)
                            else:
                                target_conf_score_path = os.path.join(basedir,path.replace("swap", swapf),f"{sn}logits",logits_file)
                                target_conf_score_processed_path = None


                            target_pref, questions = get_pref(target_conf_score_path, target_conf_score_processed_path, dataset, re_gen, logits)


                            #n-to-one (target, source) prompt > one = source query pref
                            if prompt_repr == 'cot_logits_dynamic':
                                if sn != 1:
                                    if isDynamic:
                                        source_conf_score_path = os.path.join(basedir, path.replace("swap", swapt), "dynamic_ensemble", f"AN-greedy_{greedy_stat}_cot_logits{seeds_str}.json")
                                    else:
                                        source_conf_score_path = os.path.join(basedir, path.replace("swap", swapt), "ensemble", f"AN-greedy_{greedy_stat}_cot_logits{seeds_str}.json")
                                else:
                                    source_conf_score_path = os.path.join(basedir, path.replace("swap", swapt), f"AN-greedy_{greedy_stat}_cot_logits{seeds_str}.json")
                                source_conf_score_processed_path = None
                            elif not logits:
                                source_conf_score_path = os.path.join(basedir,path.replace("swap", swapt) ,"confidence",confidence_file)
                                source_conf_score_processed_path = os.path.join(basedir,path.replace("swap", swapt) ,"confidence",processed_confidence_file)
                            else:
                                source_conf_score_path = os.path.join(basedir,path.replace("swap", swapt) ,f"{sn}logits",logits_file)
                                source_conf_score_processed_path = None

                            source_pref, _ = get_pref(source_conf_score_path, source_conf_score_processed_path, dataset, re_gen, logits)

                            results = {"config":target_pref["config"], "eval":[]}
                            if avg_logits:
                                output_folder = os.path.join(basedir, "stable-match", path.replace("-swap", ""))
                                # if logits:
                                #     output_folder = os.path.join(output_folder, "avg-logits")

                                if aggr == 'multiply':
                                    output_folder = os.path.join(output_folder, f"{sn}multiply")
                                else:
                                    output_folder = os.path.join(output_folder, f"{sn}average")
                                questions_path = os.path.join(output_folder, "questions.json")
                                os.makedirs(output_folder, exist_ok=True)
                                with open(questions_path, 'w') as f:
                                    json.dump(questions, f, indent=4)
                                save_avg_logits(pref_key=pref_key, output_dir = output_folder, mode = ensemble_mode, dataset=dataset, questions_path = questions_path, swap_t = source_pref, swap_f = target_pref, agg = aggr, gstat = greedy_stat)
                                continue
                                # exit()


                            results["config"]["n_prompts"]=False #swap is false as well
                            results["config"]["stable-matching"]=True #swap is false as well

                            n = len(target_pref["eval"])
                            max_r = 0
                            for i in range(n):
                                # print(f"test case {i}/{n}")
                                entry = {
                                    "id" : target_pref["eval"][i]["id"],
                                    "gold_mapping" : target_pref["eval"][i]["gold_mapping"],
                                    "predicted_mapping" : []
                                }

                                source_attr = source_pref["eval"][i][pref_key].keys()
                                target_attr = target_pref["eval"][i][pref_key].keys()

                                pref_source_attr = source_pref["eval"][i][pref_key]
                                pref_target_attr = target_pref["eval"][i][pref_key]


                                # Create the ManyToManyStableMatcher instance
                                matcher = ManyToManyStableMatcher(source_attr, target_attr, pref_source_attr, pref_target_attr, top_k=k, is_logits=logits)

                                # Perform the matching
                                matches, round_results = matcher.match()


                                top_r = len(round_results)
                                if len(round_results) > 0:
                                    if top_r > max_r:
                                        max_r = top_r
                                #     # print('//////////////////////////////////////////////////////////////')
                                #     print('---------------top_r')
                                #     print(top_r)
                                #     for r in round_results:
                                #         print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                                #         print(r)

                                entry["predicted_mapping"] = round_results
                                # entry["predicted_mapping"] = matches
                                results["eval"].append(entry)
                                # if entry['id']=="mimic-iii:CALLOUT|omop:VISIT_DETAIL":
                                    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                                    # print(entry['id'])
                                    # print('\npref_source_attr')
                                    # print(pref_source_attr)
                                    # print('\npref_target_attr')
                                    # print(pref_target_attr)
                                    # print('\nsource_attr')
                                    # print(source_attr)
                                    # print('\ntarget_attr')
                                    # print(target_attr)
                                    # print('\nmatches')
                                    # print(matches)
                                    # print(".................................")

                            print(f"top {max_r} stable match done.")
                            count_max = 0
                            for i in range(1,max_r+1):
                                result_r = {"config": target_pref["config"], "eval": []}
                                result_r["config"]["n_prompts"] = False  # swap is false as well
                                result_r["config"]["stable-matching"] = True  # swap is false as well

                                count_max = 0
                                for entry in results["eval"]:
                                    if len(entry["predicted_mapping"]) == max_r:
                                        count_max +=1
                                    if len(entry["predicted_mapping"])>=i:
                                        entry_r = copy.deepcopy(entry)
                                        entry_r['predicted_mapping']=entry['predicted_mapping'][i-1]
                                        result_r["eval"].append(entry_r)
                                    else:
                                        entry_r = copy.deepcopy(entry)
                                        if entry["predicted_mapping"]==[]:
                                            entry_r['predicted_mapping']=[]
                                        else:
                                            entry_r['predicted_mapping']=entry['predicted_mapping'][-1]
                                        result_r["eval"].append(entry_r)

                                print('here >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> max_r, count_max')
                                print(max_r, count_max)
                                output_folder = os.path.join(basedir, "stable-match", path.replace("-swap", ""))
                                if isDynamic:
                                    output_folder = os.path.join(output_folder, f"dynamic-{seeds_str}")
                                elif logits or "logits" in prompt_repr:
                                    output_folder = os.path.join(output_folder, f"{sn}logits")
                                results_path = os.path.join(output_folder, f"{ensemble_mode}-greedy-{greedy_stat}-top-{i}.json")
                                questions_path = os.path.join(output_folder, "questions.json")
                                os.makedirs(output_folder, exist_ok=True)

                                write_n_eval(output_folder, result_r, questions, results_path, questions_path, dataset, ensemble_mode, i)

                            print(f" >>> Stable Matching Evaluation Done.")





