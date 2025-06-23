import argparse
import json
import os

from cmd_utils import get_class_dict
from LLM import llm_classes

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, required=True,
                        help="The file containing the prompts")

    llm_dict = get_class_dict(llm_classes)
    parser.add_argument("--model", type=str, required=True,
                        help="The LLM's model ID", choices=llm_dict.keys())
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top_p", type=float, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=4096)

    # Grab args, do any necessary preprocessing
    args = parser.parse_args()
    assert args.input_file[-5:].lower() == ".json", f"Expected JSON file, but got {args.input_file[-5:]}"
    args.input_file = os.path.normpath(args.input_file)

    with open(args.input_file, "r") as f:
        prompts = json.load(f)

    '''
        Output file handling
    '''
    base_path, output_file = os.path.split(args.input_file)
    output_file = os.path.join(base_path, f"OUTPUT-{output_file}")

    if os.path.exists(output_file):
        print(f"File '{output_file}' exists. Stopping.")
        exit(0)
    else:
        print(f"File '{output_file}' does not exist. Creating.")
        os.makedirs(base_path, exist_ok=True)

    model_instance = llm_dict[args.model](
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=prompts["prompt_config"]["seed"],
    )

    output_json = {
        "prompt_config": prompts["prompt_config"],
        "model_config": model_instance.get_config(),
        "prompts": [],
    }

    for p in prompts["prompts"]:

        input = model_instance.apply_template(p["content"])
        output = model_instance.prompt_model([input])

        output_json["prompts"].append(
            {"cluster": p["cluster"], "input": input, "output": output}
        )

    with open(output_file, "w") as out:
        json.dump(output_json, out, indent=4)
    ## Create the DataLoader for batching
    #question_loader = DataLoader([(model_instance.apply_template(p["content"]), p["cluster"]) for p in prompts["prompts"]], batch_size=1, shuffle=False,
    #                             drop_last=False)

    #for i, batch in enumerate(tqdm(question_loader)):

    #    pdb.set_trace()

    #    prompts, clusters = zip(*batch)

    #    responses = model_instance.prompt_model(prompts)

    #    for clust, p, res in zip(clusters, prompts, responses):
    #        output_json["prompts"].append(
    #            {"cluster": clust, "input": p, "output": res}
    #        )

