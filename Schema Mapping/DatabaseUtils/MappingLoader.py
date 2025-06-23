import errno
import os
import random

from DatabaseUtils.ConfigReader import get_config

import xml.etree.ElementTree as ET

from DatabaseUtils.SchemaLoader import load_schema

CONFIG = get_config()

def _check_file_path(path):
    if not os.path.exists(path):
        print(f"{path} does not exist.")
    else:
        try:
            open(path)
        except IOError as e:
            if e.errno == errno.EACCES:
                return "some default data"
            # Not a permission error.
            raise

def _get_file_pre(dataset_name: str) -> str:
    return f"{CONFIG['paths']['datasets']}/{dataset_name}"

def _parse_atom(atom):
    return atom.attrib["tableref"]

def load_gt_clusters(dataset_name: str, source_name: str, target_name: str):
    """
        Load gold mapping
    """
    xml_file_path = f"{_get_file_pre(dataset_name)}/mappings/{source_name}-to-{target_name}.xml"

    _check_file_path(xml_file_path)

    root = ET.parse(xml_file_path).getroot()

    gold_mapping = []

    for gold_rule_node in root.findall("Mappings/Mapping"):

        parsed = {"source_relations": [], "target_relations": [], "id": gold_rule_node.attrib['id']}

        # Source relations
        for atom in gold_rule_node.findall("Raw/Foreach/Atom"):
            parsed["source_relations"].append(_parse_atom(atom))

        # Target relations
        for atom in gold_rule_node.findall("Raw/Exists/Atom"):
            parsed["target_relations"].append(_parse_atom(atom))

        parsed["gold_sql"] = gold_rule_node.find("Gold_SQL").text

        if len(parsed["target_relations"]) == 1:
            parsed["join_overlap_sql"] = None
        else:
            parsed["join_overlap_sql"] = gold_rule_node.find("Select_SQL").text

        gold_mapping.append(parsed)

    return gold_mapping

def load_multimap_clusters(dataset_name: str, source_name: str, target_name: str, mappings_per_cluster: int, seed: int):
    xml_file_path = f"{_get_file_pre(dataset_name)}/mappings/{source_name}-to-{target_name}.xml"

    _check_file_path(xml_file_path)

    root = ET.parse(xml_file_path).getroot()

    mappings = load_gt_clusters(dataset_name, source_name, target_name)

    # Remove subsumed mappings
    for m_this in mappings:
        for m_other in [m for m in mappings if m["id"] != m_this["id"]]:

            this_src = set(m_this["source_relations"])
            this_tgt = set(m_this["target_relations"])
            other_src = set(m_other["source_relations"])
            other_tgt = set(m_other["target_relations"])

            if len(this_src - other_src) == 0 and len(this_tgt - other_tgt) == 0:
                new_mappings = [m for m in mappings if m["id"] != m_this["id"]]
                assert len(new_mappings) == len(mappings) - 1
                mappings = new_mappings
                break

    # Randomly partition mappings into groups of size n
    random.seed(seed)
    n = min(mappings_per_cluster, len(mappings))
    random.shuffle(mappings)
    final_groups = []
    while len(mappings) > 0:
        src_rels = set()
        tgt_rels = set()
        ids = []

        # Merge mappings into a group
        for _ in range(min(n, len(mappings))):
            next_to_add = mappings.pop(0)
            src_rels = src_rels.union(set(next_to_add["source_relations"]))
            tgt_rels = tgt_rels.union(set(next_to_add["target_relations"]))
            ids.append(next_to_add["id"])

        final_groups.append({"source_relations": list(src_rels), "target_relations": list(tgt_rels), "id": ids})

    # Return mappings. In calling class, make sure to get len() of mappings returned and modify the arg so that the directory isn't named
    # something crazy like multimap_9999999999
    return final_groups

def load_gav_clusters(dataset_name: str, source_name: str, target_name: str):
    """
        Load clusters with
            source: all_tables
            target: table_1
            ...
            source: all_tables
            target: table_n
    """

    source_relations = [rel.name for rel in load_schema(dataset_name, source_name).relations.values()]
    target_relations = [rel.name for rel in load_schema(dataset_name, target_name).relations.values()]

    clusters = []

    for target_rel in target_relations:
        clusters.append({"source_relations": source_relations, "target_relations": [target_rel]})

    return clusters

def load_full_cluster(dataset_name: str, source_name: str, target_name: str):
    """
        Load cluster with
            source: all_tables
            target: all_tables
    """

    source_relations = [rel.name for rel in load_schema(dataset_name, source_name).relations.values()]
    target_relations = [rel.name for rel in load_schema(dataset_name, target_name).relations.values()]

    clusters = []
    clusters.append({"source_relations": source_relations, "target_relations": target_relations})

    return clusters