import json
import os
import random

from eval.eval_MatchMaker import find_result_files



class BasicExampleSelector(object):
    def __init__(self, data, *args, **kwargs):
        self.data = data
        self.train_json = self.data.get_dataset_json()
        self.db_ids = [d["id"] for d in self.train_json]

    def get_examples(self, question, num_example):
        pass

    def get_example_prefix(self):
        pass

    def domain_mask(self, candidates: list, target_id, different_schemas=False):
        # TODO: check the conditions
        if different_schemas:
            cross_domain_candidates = [candidates[i] for i in range(len(self.db_ids))
                                       if (self.db_ids[i] != target_id and self.train_json[i]['source_schema'] !=
                                           self.train_json[target_id]['source_schema'])]

        else:
            cross_domain_candidates = [candidates[i] for i in range(len(self.db_ids))
                                       if self.db_ids[i] != target_id]
        return cross_domain_candidates


class MMSelector:
    def __init__(self, *args, **kwargs):
        self.dataset = kwargs['dataset']
        self.dir_ = f"dataset/baseline/{self.dataset}"
        self.repr_type = kwargs['repr_type']
        self.demo = self.load_demo(kwargs['isConfidence2'])
        self.trace, self.data = self.load_trace()


    def load_demo(self, is_confidence2):
        if is_confidence2:
            demo_path = self.dir_+"/mm2/icl.json"
        else:
            demo_path = self.dir_+"/mm/icl.json"
        with open(demo_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_examples(self, target, num_example):
        examples = []
        for ex in self.demo:
            if ex["id"] == target["id"]:
                continue
            if len(examples)>num_example:
                break
            idx = ex["idx"]
            attr = ex["attr"]
            prompt = self.get_example_prompt(self.data[idx]["prompt"][attr])
            answer = self.trace["eval"][idx]["predicted_mapping"][attr]
            examples.append([prompt, answer])

        return examples

    def load_trace(self):
        seed = self.demo[0]["seed"]

        if self.repr_type == "candidate_mm":
            folder_path = os.path.join(self.dir_, "candidate_mm", "70-candidate_mm-1024")
            candidate_json = find_result_files(folder_path, seed)
            result_path = os.path.join(folder_path, candidate_json)
        elif self.repr_type == "formatter_mm":
            folder_path = os.path.join(self.dir_, "formatter_mm",
                                          "Meta-Llama-3.1-70B-Instruct-GPTQ-INT4-formatter_mm-1024")
            formatter_json = find_result_files(folder_path, seed)
            result_path = os.path.join(folder_path, formatter_json)
        else: # self.repr_type == "confidence_mm":
            folder_path = os.path.join(self.dir_, "confidence_mm",
                                           "Meta-Llama-3.1-70B-Instruct-GPTQ-INT4-confidence_mm-1024")
            confidence_json = find_result_files(folder_path, seed)
            result_path = os.path.join(folder_path, confidence_json)

        data_path = os.path.join(folder_path, "questions-s-7564.json")

        with open(result_path, 'r') as f:
            traces = json.load(f)
        with open(data_path, 'r') as f:
            data = json.load(f)

        return traces, data

    def get_example_prompt(self, prompt):
        if self.repr_type == 'candidate_mm':
            temp = prompt.split("Input Schema:")[1]
            return "Input Schema:" + temp.split("think step by step in order to")[0] + "think step by step in order to"
        if self.repr_type == 'formatter_mm':
            temp = prompt.split("Input:")[1]
            return "Input:" + temp.split("(F)")[0] + "(F) No Match"
        return prompt.split("}")[0] + "}"


class RandomExampleSelector(BasicExampleSelector):
    def __init__(self, data, seed, *args, **kwargs):
        super().__init__(data)
        self.rng = random.Random(seed)

    def get_examples(self, target, num_example):
        # Never sample target as a few-shot example
        indexes = [i for i in range(len(self.train_json))
                   if self.db_ids[i] != target["id"]]
        selected_indexes = self.rng.sample(indexes, num_example)
        return [self.train_json[index] for index in selected_indexes]



class SimilarNullCoverageExampleSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

    def calculate_similarity(self, target_coverage, candidate_coverage):
        # Use absolute difference to calculate similarity
        return abs(target_coverage - candidate_coverage)

    def get_examples(self, target, num_example):
        target_id = target["id"]
        target_null_coverage = target.get("coverage_ratio", 0)  # Use 0 if there's no coverage_ratio

        if target_null_coverage == 0:
            print('Need to calculate coverage ratio from new JSON')

        # Calculate similarities for all candidates
        similarities = []
        for i in range(len(self.train_json)):
            if self.db_ids[i] != target_id:  # Ensure the example is not the same as the target
                candidate_null_coverage = self.train_json[i].get("coverage_ratio", 0)
                similarity = self.calculate_similarity(target_null_coverage, candidate_null_coverage)
                similarities.append((similarity, i))

        # Sort candidates by similarity (ascending order)
        similarities.sort(key=lambda x: x[0])

        # Select the top num_example most similar candidates
        selected_indexes = [idx for _, idx in similarities[:num_example]]

        return [self.train_json[index] for index in selected_indexes]

