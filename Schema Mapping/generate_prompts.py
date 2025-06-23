import os

from DatabaseUtils.ArgHelper import ArgHelper
from DatabaseUtils.DatabaseManager import DatabaseManager
from DatabaseUtils.SchemaLoader import load_schema
from DatabaseUtils.MappingLoader import load_gt_clusters, load_gav_clusters, load_full_cluster, load_multimap_clusters

import json

from DatabaseUtils.PromptBuilder import PromptBuilder

if __name__ == '__main__':

    # Grab args, do any necessary preprocessing
    arg_helper = ArgHelper()
    arg_helper.parse_args_from_cmd_line()

    clusters = None
    if arg_helper.args.clusters == "ground_truth":
        clusters = load_gt_clusters(
            dataset_name=arg_helper.args.DATASET, source_name=arg_helper.args.source, target_name=arg_helper.args.target
        )
    elif arg_helper.args.clusters.startswith("multimap"):
        assert arg_helper.args.clusters[8] == "_", "For multimap, must specify number of mappings per group with _n (e.g., 'multimap_2')."
        n = arg_helper.args.clusters[9:]
        assert n.isdigit(), "Assert expected integer n for 'multimap_n' but got '" + n + "'"
        clusters = load_multimap_clusters(
            dataset_name=arg_helper.args.DATASET, source_name=arg_helper.args.source, target_name=arg_helper.args.target,
            mappings_per_cluster=int(n), seed=arg_helper.args.seed
        )
        arg_helper.args.clusters = f"multimap_{n}"
    elif arg_helper.args.clusters == "gav":
        clusters = load_gav_clusters(
            dataset_name=arg_helper.args.DATASET, source_name=arg_helper.args.source, target_name=arg_helper.args.target
        )
    elif arg_helper.args.clusters == "full":
        clusters = load_full_cluster(
            dataset_name=arg_helper.args.DATASET, source_name=arg_helper.args.source, target_name=arg_helper.args.target
        )
    else:
        clusters = None  # Load in the specified file and use those clusterings. Just assert that this does not work right now.

    source_schema = load_schema(dataset_name=arg_helper.args.DATASET, schema_name=arg_helper.args.source)
    target_schema = load_schema(dataset_name=arg_helper.args.DATASET, schema_name=arg_helper.args.target)

    # Test clusters
    for c in clusters:
        for rel in c["source_relations"]:
            assert rel in source_schema.relations, f"Source relation '{rel}' not found in source schema."
        for rel in c["target_relations"]:
            assert rel in target_schema.relations, f"Target relation '{rel}' not found in target schema."

    db_manager = DatabaseManager()
    db_manager.attach_database(source_schema, "source", with_constraints=True)
    db_manager.attach_database(target_schema, "target", with_constraints=True)

    prompt_template = arg_helper.get_template()
    serializer = arg_helper.get_serializer()
    sampler = arg_helper.get_sampler()

    prompt_builder = PromptBuilder(
        db_manager=db_manager, prompt_template=prompt_template, seed=arg_helper.args.seed, serializer=serializer, sampler=sampler,
        shuffle_relations=arg_helper.args.shuffle_relations, shuffle_attributes=arg_helper.args.shuffle_attributes
    )

    output_json = {
        "prompt_config": prompt_builder.get_config() | {"clusterings": arg_helper.args.clusters},
        "prompts": [],
    }

    '''
        Output file handling
    '''
    full_path = arg_helper.get_filepath()
    if os.path.exists(full_path):
        print(f"File '{full_path}' exists. Stopping.")
        exit(0)
    else:
        print(f"File '{full_path}' does not exist. Creating.")
        base_path, _ = os.path.split(full_path)
        os.makedirs(base_path, exist_ok=True)

    db_manager.import_data("source")
    db_manager.import_data("target")

    '''
        Create prompts
    '''
    # Prompt per cluster
    for c in clusters:

        output_json["prompts"].append(
            {
                "cluster": {
                    "source": c["source_relations"],
                    "target": c["target_relations"],
                },
                "content": prompt_builder.get_prompt_content(
                    source_relations=c["source_relations"],
                    target_relations=c["target_relations"],
                ),
            }
        )

    with open(full_path, "w") as out:
        json.dump(output_json, out, indent=4)

    print(f"Generated {full_path}.")
