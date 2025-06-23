import errno
import os

from DatabaseUtils.Schema import *
from DatabaseUtils.ConfigReader import get_config

import xml.etree.ElementTree as ET

from DatabaseUtils.enums import Dataset

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

def _get_file_pre(dataset_name, database_name) -> str:
    return f"{CONFIG['paths']['datasets']}/{dataset_name}/{database_name}"

def _str_to_bool(s):
    if s.lower() == 'true':
         return True
    elif s.lower() == 'false':
         return False
    else:
         raise ValueError

def _get_bool(s: Optional[ET.Element], default: bool) -> bool:
    if s is None:
        return default
    else:
        s = s.text

    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError(s)

def _get_schema_amalgam(dataset_name: str, database_name: str) -> Schema:
    """

    """

    xml_file_path = f"{_get_file_pre(dataset_name, database_name)}/schema.xml"
    _check_file_path(xml_file_path)

    root = ET.parse(xml_file_path).getroot()

    schema = Schema(dataset_name, database_name)
    schema_node = root.find("Schemas/SourceSchema")

    # Get all relations and their attributes
    for rel_node in schema_node.findall("Relation"):
        relation_name = rel_node.attrib["name"]

        # Add attributes
        for attr_node in rel_node.findall("Attr"):
            size = attr_node.find("DataType/Size")
            if size is not None:
                size = int(size.text)

            new_attribute = Attribute(
                name=attr_node.find("Name").text,
                data_type=attr_node.find("DataType/Type").text, data_size=size,
                nullable=_get_bool(attr_node.find("Nullable"), default=True),
                meaningful=_get_bool(attr_node.find("HasSemanticMeaning"), default=True)
            )
            schema.add_attribute(relation_name, new_attribute)

        # Table-level constraints

        #   Unique
        for unique_node in rel_node.findall("Unique"):
            schema.add_unique_constraint(relation_name,
                                         [attr.text for attr in unique_node.findall("Attr")])

        #   Primary key
        for attr_node in rel_node.findall("PrimaryKey/Attr"):
            schema.add_attribute_to_primary_key(relation_name, attr_node.text)

    # Add foreign keys
    for fk_node in schema_node.findall("ForeignKey"):

        schema.add_foreign_key(
            fk_node.find("From").attrib["tableref"],
            [attr_node.text for attr_node in fk_node.findall("From/Attr")],
            fk_node.find("To").attrib["tableref"],
            [attr_node.text for attr_node in fk_node.findall("To/Attr")],
        )

    return schema

def _get_schema_ehr(dataset_name: str, database_name: str) -> Schema:
    """

    """

    xml_file_path = f"{_get_file_pre(dataset_name, database_name)}/schema.xml"
    _check_file_path(xml_file_path)

    root = ET.parse(xml_file_path).getroot()

    schema = Schema(dataset_name, database_name)

    foreign_keys = []

    # Get all relations and their attributes
    for rel_node in root.findall("tables/table"):
        relation_name = rel_node.attrib["name"]

        if "chartevents_" in relation_name:
            continue

        # Add attributes
        for attr_node in rel_node.findall("column"):

            attribute_name = attr_node.attrib["name"]

            new_attribute = Attribute(
                name=attribute_name,
                data_type=attr_node.attrib["type"],
                size=int(attr_node.attrib["size"]),
                nullable=attr_node.attrib["nullable"],
                description=attr_node.attrib["remarks"]
            )
            schema.add_attribute(relation_name, new_attribute)

            # Track FKs to add at the end
            for fk_node in attr_node.findall("parent"):
                foreign_keys.append({
                    "from_relation_name": relation_name,
                    "from_attribute_names": [attribute_name],
                    "to_relation_name": fk_node.attrib["table"],
                    "to_attribute_names": [fk_node.attrib["column"]],
                    "name": fk_node.attrib["foreignKey"]
                })

        # Add primary key
        pk_node = rel_node.find("primaryKey")
        if pk_node is not None:
            schema.add_attribute_to_primary_key(relation_name, rel_node.find("primaryKey").attrib["column"])

    # Add foreign keys
    for fk in foreign_keys:
        schema.add_foreign_key(**fk)

    return schema

def load_schema(dataset_name: str, schema_name: str) -> Schema:
    dataset_to_method = {
        Dataset.AMALGAM.value: _get_schema_amalgam,
        Dataset.EHR.value: _get_schema_ehr
    }

    assert dataset_name in dataset_to_method, f"No script found for {dataset_name}"
    return dataset_to_method[dataset_name](dataset_name, schema_name)