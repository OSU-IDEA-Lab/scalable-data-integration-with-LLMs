import argparse
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader

import time
from LLM.llm_classes import *
from utils.data_builder import get_seed, extract_ensemble_mode

# run python ask_LLM.py --question dataset/process/ALL-source_target_0-SHOT-200-ANS-2048/

QUESTION_FILE = "questions.json"
PATH_DATA = "dataset/"


def get_model_class(model_name: str):
    if model_name == LLM.DeepSeekCoderLiteInstruct:
        return DeepSeekCoderLiteInstruct
    elif model_name == LLM.Llama3p1_8:
        return Llama3p1_8
    elif model_name == LLM.Phi3:
        return Phi3
    elif model_name == LLM.Phi3_5:
        return Phi3_5
    elif model_name == LLM.Qwen2_5_32B:
        return Qwen2_5_32B
    elif model_name == LLM.Llama3_1_GPTQ:
        return Llama3_1_GPTQ
    else:
        raise ValueError(f"{model_name} is not supported yet")


def get_candidates(schema):
    return [col["name"] for col in schema["columns"]]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, help="The questions file to run over (e.g., questions-s-0.json)")
    parser.add_argument("--model", type=str,
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
                            LLM.Phi3_5,
                            LLM.Phi3,
                            LLM.CodeLlama13GPTQ],
                        default='TheBloke/CodeLlama-13B-Instruct-GPTQ')
    parser.add_argument("--llm_seed", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=1000000)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, help="The maximal length that an answer takes")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--stop_token", type=str,
                        choices=[
                            STOP_TOKEN.SQL,
                            STOP_TOKEN.JSON,
                            STOP_TOKEN._None_],
                        default=';')


    parser.add_argument("--greedy", type=str2bool, default=False, help="Enable greedy mode (True/False)")

    args = parser.parse_args()

    questions_json = json.load(open(args.question, "r"))
    # questions = [{"id": _["id"], "prompt": _["prompt"], "answer_prefix": _["answer_prefix"], "gold_mapping": _["gold_mapping"], "gold_sql": _["gold_sql"]} for _ in questions_json]
    questions = [
        {"id": _["id"], "n_prompts": _["n_prompts"], "prompt": _["prompt"], "answer_prefix": _.get("answer_prefix", None),
         "gold_mapping": _["gold_mapping"], "mcq_dict": _.get("mcq_dict", {})} for _ in questions_json]
    config = {arg: getattr(args, arg) for arg in vars(args)}

    if args.greedy:
        config['temperature'] = 0
        config['top_p'] = 1
    config["n_prompts"] = questions[0]["n_prompts"]
    confidence_score_prompt = False
    if "confidence" in args.question:
        confidence_score_prompt = True
        config["ensemble_mode"] = extract_ensemble_mode(args.question)
    logits = False
    cot_logits = False
    if "cot_logits" in args.question:
        cot_logits = True
        candidate_dict = {}
        for _ in questions_json:
            candidate_dict[_["id"]] = get_candidates(_["source_schema"])

    elif "logits" in args.question:
        logits = True
        config["ensemble_mode"] = extract_ensemble_mode(args.question)

    if args.start_index == 0:
        mode = "w"
    else:
        mode = "a"
    model_name = args.model
    if "/" in model_name:
        model_name = model_name.split("/")[1]

    curr_timestamp = datetime.now().strftime("%d%b%y_%Hh%Mm")
    seed = get_seed(args.question)
    path_to_question, _ = os.path.split(args.question)

    greedy_stat = str(args.greedy)

    if args.llm_seed != 0:
        out_file = f"{path_to_question}/RESULTS-{curr_timestamp}-s-{args.llm_seed}.json"
    elif confidence_score_prompt:
        out_file = args.question.replace("questions.json", "scores.json")
    elif cot_logits:
        out_file = args.question.replace("questions", f"greedy_{greedy_stat}_cot_logits")
    elif logits:
        out_file = args.question.replace("questions.json", "logits.json")
    elif seed is None:
        out_file = f"{path_to_question}/RESULTS-greedy_{greedy_stat}-{curr_timestamp}.json"
    else:
        out_file = f"{path_to_question}/RESULTS-greedy_{greedy_stat}-{curr_timestamp}-s-{seed}.json"

    # Create the DataLoader for batching
    question_loader = DataLoader([q["prompt"] for q in questions], batch_size=args.batch_size, shuffle=False,
                                 drop_last=False)

    # Get the correct model class and instantiate it
    ModelClass = get_model_class(args.model)
    model_instance = ModelClass(temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens,
                                stop_token=args.stop_token, seed =args.llm_seed)

    results = []
    token_cnt = 0

    start_time = time.time()
    for i, batch in enumerate(tqdm(question_loader)):
        if i < args.start_index:
            continue
        if i >= args.end_index:
            break
        if not config["n_prompts"]:
            try:
                res = model_instance.attempt_llm_request(batch)
            except Exception as e:
                print(f"{e}!")
                res = {"response": [] * len(batch)}

            # Parse results
            for j, mapping in enumerate(res["response"]):

                q = questions[i * args.batch_size + j]
                if q["answer_prefix"] is not None:
                    mapping = q["answer_prefix"] + mapping

                results.append({
                    "id": q["id"],
                    "gold_mapping": q["gold_mapping"],
                    "predicted_mapping": mapping
                })
        else:
            n_mappings = {}
            n_reasons = {}

            q = questions[i * args.batch_size]

            if logits:
                for attribute in batch.keys():
                    try:
                        res = model_instance.attempt_llm_logits(batch[attribute], q["mcq_dict"][attribute])
                    except Exception as e:
                        print(f"{e}!")
                        res = []

                    print('-----------------\nres')
                    print(res)
                    # Parse results
                    for j, mapping in enumerate(res):
                        if q["answer_prefix"] is not None:
                            mapping = q["answer_prefix"] + mapping
                        n_mappings[attribute] = mapping
                    print(n_mappings[attribute])

                result_dict = {
                    "id": q["id"],
                    "gold_mapping": q["gold_mapping"],
                    "preferences": n_mappings
                }
                if config["n_prompts"]:
                    result_dict["mcq_dict"] = q["mcq_dict"]

                results.append(result_dict)
            elif cot_logits:
                # print("-----------------==================--------------------")

                for attribute in batch.keys():
                    try:
                        #suppose batch size is 1
                        # res, reason = model_instance.attempt_llm_logits_append_method(batch[attribute][0], candidate_dict[q["id"]]+["None"])
                        res, reason = model_instance.try_gen(batch[attribute][0], candidate_dict[q["id"]]+["None"], args.greedy)
                        # res = model_instance.attempt_llm_logits_append_method(batch[attribute][0], candidate_dict[q["id"]])
                    except Exception as e:
                        print(f"{e}!")
                        reason = str(e)
                        res = []

                    # Parse results
                    n_mappings[attribute] = res
                    n_reasons[attribute] = reason
                    # print("\n\nAttribute : ",attribute)
                    # print(n_reasons[attribute])
                    # print(n_mappings[attribute])
                    # print("\n\n")
                    # exit()

                result_dict = {
                    "id": q["id"],
                    "gold_mapping": q["gold_mapping"],
                    "preferences": n_mappings,
                    "reasons": n_reasons
                }
                if config["n_prompts"]:
                    result_dict["mcq_dict"] = q["mcq_dict"]

                results.append(result_dict)

            else:

                for attribute in batch.keys():
                    try:
                        res = model_instance.attempt_llm_request(batch[attribute], args.greedy)
                    except Exception as e:
                        print(f"{e}!")
                        res = {"response": []}

                    # Parse results
                    for j, mapping in enumerate(res["response"]):
                        if q["answer_prefix"] is not None:
                            mapping = q["answer_prefix"] + mapping
                        n_mappings[attribute] = mapping

                result_dict = {
                    "id": q["id"],
                    "gold_mapping": q["gold_mapping"],
                    "predicted_mapping": n_mappings
                }
                if config["n_prompts"]:
                    result_dict["mcq_dict"] = q["mcq_dict"]

                results.append(result_dict)

    end_time = time.time()
    generation_time = end_time - start_time

    # Save results and elapsed time to JSON file
    output_data = {
        'config': config,
        'eval': results,
        'generation_time': generation_time
    }

    # print(output_data)

    with open(out_file, mode) as f:
        json.dump(output_data, f, indent=4)