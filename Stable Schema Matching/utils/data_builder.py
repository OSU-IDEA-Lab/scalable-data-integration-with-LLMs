import os
import random
import re
import json

PATH_DATA = "data/"

class BasicDataset(object):
    def __init__(self, path_data, dataset, seed):
        self.path_data = path_data
        self.dataset = dataset
        self.dataset_json = os.path.join(self.path_data, self.dataset)
        self.databases = dict()
        self.rng = random.Random(seed)

    def get_dataset_json(self, swap_tables):
        try:
            with open(self.dataset_json, 'r') as file:
                data = json.load(file)

                # Shuffle columns for each example
                for example in data:
                    self.rng.shuffle(example["source_schema"]["columns"])
                    self.rng.shuffle(example["target_schema"]["columns"])

                    if swap_tables:
                        # Swap ground truth pairs
                        example['gold_mapping'] = [[pair[1], pair[0]] for pair in example['gold_mapping']]

                        # Swap tables
                        source = example["source_schema"]
                        example["source_schema"] = example["target_schema"]
                        example["target_schema"] = source

                    example["swapped"] = swap_tables

            return data
        except FileNotFoundError:
            print(f"File not found: {self.dataset_json}")
            return []
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {self.dataset_json}")
            return []

    # TODO: change this
    def get_gold_sql(self):
        with open(self.dataset_json, "r") as file:
            answers = file.readlines()
            return answers

    # parser.add_argument("--data_type", type=str, choices=["assays", "miller2", "prospect"], default="all")

def get_schemas_for_id(example, dataset_name):
    if dataset_name == "valentine":
        file_path = "valentine/Valentine_dataset.json"
    elif dataset_name == "ehr":
        file_path = "ehr/ehr_dataset.json"
    elif dataset_name == "bird":
        file_path = "bird/bird_dataset.json"
    elif dataset_name == "synthea":
        file_path = "synthea/synthea_dataset.json"
    elif dataset_name == "gdc":
        file_path = "gdc/gdc_dataset.json"

    with open(os.path.join('data', file_path), 'r') as f:
        data = json.load(f)

    # Find the instance with the same id as example['id']
    for instance in data:
        if instance['id'] == example['id']:
            source_schema = instance['source_schema']
            target_schema = instance['target_schema']
            return source_schema, target_schema

    # If no matching instance is found, return None
    return None, None



def load_data(dataset, path_data, seed):
    if dataset == "valentine":
        return BasicDataset(path_data, "valentine/Valentine_dataset.json", seed)
    elif dataset == "ehr":
        return BasicDataset(path_data, "ehr/ehr_dataset.json", seed)
    elif dataset == "bird":
        return BasicDataset(path_data, "bird/bird_dataset.json", seed)
    elif dataset == "synthea":
        return BasicDataset(path_data, "synthea/synthea_dataset.json", seed)
    elif dataset == "gdc":
        return BasicDataset(path_data, "gdc/gdc_dataset.json", seed)


def load_predicted_matches(num_samples, ensemble_mode, folder_path, dataset):
    json_file = f"{ensemble_mode}_ensemble_results.json"
    file_path = os.path.join("dataset/process", dataset,folder_path,f"{num_samples}ensemble",json_file)
    with open(file_path, 'r') as f:
        return json.load(f)

def load_raw_confidence_scores(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_column_names(gold_sql):
    pattern = re.compile(r'CREATE VIEW [^()]+ \(([^)]+)\)')
    match = pattern.search(gold_sql)

    if match:
        columns = match.group(1).split(', ')
        return columns
    else:
        return []


def extract_ensemble_mode(path):
    question_file = path.split("/")[-1]
    ensemble_mode = question_file.split("-questions.json")[0]
    return ensemble_mode

def get_seed(file_name):
    sub_start = file_name.rfind('-s-')
    if sub_start == -1:
        return None
    return file_name[sub_start + 3:-5]

def selective_format(template, **kwargs):
    """
    Formats the template using provided arguments, replacing only the placeholders
    that exist in the template. Extra arguments are ignored.
    """
    return template.format(**{k: v for k, v in kwargs.items() if '{' + k + '}' in template})


def load_schemas(dataset_name, swapped=False):
    dataset_json = {
        "ehr": "ehr/ehr_dataset.json",
        "synthea": "synthea/synthea_dataset.json",
        "bird": "bird/bird_dataset.json",
        "valentine": "valentine/Valentine_dataset.json"
    }
    path = os.path.join(PATH_DATA, dataset_json[dataset_name])

    with open(path, 'r') as file:
        dataset = json.load(file)

    transformed_dataset = {}

    if not swapped:
        source_key = "source_schema"
        target_key = "target_schema"
    else:
        source_key = "target_schema"
        target_key = "source_schema"

    for entry in dataset:
        transformed_entry = {
            source_key : {
                "name": entry["source_schema"]["name"],
                **({"description": entry["target_schema"]["description"]} if "description" in entry["target_schema"] else {}),
                # "description": entry["source_schema"].get("description", ""),
                "columns": {
                    col["name"].lower(): {
                        "type": col["type"],
                        "column_description": col["column_description"],
                        **({"is_pk": col["is_pk"]} if "is_pk" in col else {})
                    }
                    for col in entry["source_schema"]["columns"]
                }
            },
            target_key : {
                "name": entry["target_schema"]["name"],
                **({"description": entry["target_schema"]["description"]} if "description" in entry["target_schema"] else {}),
                # "description": entry["target_schema"].get("description", ""),
                "columns": {
                    col["name"].lower(): {
                        "type": col["type"],
                        "column_description": col["column_description"],
                        **({"is_pk": col["is_pk"]} if "is_pk" in col else {})
                    }
                    for col in entry["target_schema"]["columns"]
                }
            }
        }
        transformed_dataset[entry["id"]] = transformed_entry

    return transformed_dataset


# from torch.utils.data import Dataset, DataLoader
#
# class NPromptDataset(Dataset):
#     def __init__(self, questions):
#         # Flatten the data into a list of (source_key, id, prompt) tuples
#         self.data = []
#         for i in range(len(questions)):
#             for attr, prompt in questions[i]["prompt"].items():
#                 self.data.append((i, attr, prompt))
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx]
#
# def collate_fn(batch):
#     # Batch contains tuples of (source_key, id, prompt)
#     indices = [item[0] for item in batch]
#     attributes = [item[1] for item in batch]
#     prompts = [item[2] for item in batch]
#     return indices, attributes, prompts
#
# # prompt_data = {
# #     "prompts1": {"id1": "What is AI?", "id2": "Explain ML.", "id3": "Define DL."},
# #     "prompts2": {"id4": "What is physics?"}
# # }
