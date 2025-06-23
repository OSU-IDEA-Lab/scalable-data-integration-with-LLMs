import copy
import json
import sqlite3
from abc import abstractmethod
import random
from collections import OrderedDict
from typing import NamedTuple

import numpy as np

from DatabaseUtils.DatabaseManager import DatabaseManager

from DatabaseUtils.Schema import Relation
from DatabaseUtils.enums import Hint

from DatabaseUtils.prompt_templates import AbstractPromptTemplate

STR_LIMIT = 100

class RowSample(NamedTuple):
    samples: dict[str, dict[str, str]]

class ColSample(NamedTuple):
    samples: dict[str, dict[str, list[str]]]

class Samplers(object):

    class AbstractSampler(object):
        def __init__(self, sample_size: int):
            assert sample_size is not None, "Must specify sample size (n)"
            assert sample_size > 0, "Sample size (n) must be greater than 0"
            self.sample_size = sample_size

        @abstractmethod
        def sample(self, connection: sqlite3.Connection, relations: list[Relation], namespace: str):
            pass

        def set_seed(self, seed: int):
            self.seed = seed
            self.rng = np.random.default_rng(seed=self.seed)

        def get_config(self) -> dict:
            return {"type": type(self).__name__, "sample_size": self.sample_size, "seed": self.seed}

    class RowWiseSampler(AbstractSampler):
        def __init__(self, sample_size: int):
            super().__init__(sample_size)

        def sample(self, connection: sqlite3.Connection, relations: list[Relation], namespace: str) -> RowSample:
            # Randomly sample n rows
            # For large tables, this is not efficient, but this should be fine for my purposes.

            sampled_rows = {}

            for relation in relations:
                sampled_rows[relation.name] = []

                cursor = connection.execute(f"SELECT * FROM {namespace}.{relation.name};")

                for row in self.rng.choice(cursor.fetchall(), size=self.sample_size, replace=False):
                    sampled_rows[relation.name].append({attr_name: attr_value
                                         for attr_name, attr_value in zip(cursor.description, row)})

            return RowSample(sampled_rows)

    class ColumnWiseSampler(AbstractSampler):
        def __init__(self, sample_size: int, weighted: bool):
            super().__init__(sample_size)
            self._weighted = weighted

        def sample(self, connection: sqlite3.Connection, relations: list[Relation], namespace: str) -> ColSample:

            sampled_cells = {}

            for relation in relations:
                sampled_cells[relation.name] = {}

                for attr in relation.attributes:
                    values = []
                    counts = []
                    for v, c in connection.execute(f"SELECT {attr.name}, count({attr.name}) as count "
                                                      f"FROM {namespace}.{relation.name} "
                                                      f"GROUP BY {attr.name} ORDER BY count, {attr.name} DESC;").fetchall():
                        # We get (None, 0) if Nulls exist in attribute
                        if not(v is None and c == 0):
                            values.append(v)
                            counts.append(c)

                    sampled_cells[relation.name][attr.name] = []
                    if len(values) > 0:
                        if self._weighted:
                            total = sum(counts)
                            p = [c/total for c in counts]
                        else:
                            p = [1/len(counts)]*len(counts)

                        n = self.sample_size
                        if len(values) < n:
                            n = len(values)

                        for val in self.rng.choice(values, size=n, replace=False, p=p, shuffle=False):
                            value = val.item()
                            if attr.get_supertype() == "TEXT" and len(value) > STR_LIMIT:
                                value = value[:STR_LIMIT] + "..."
                            sampled_cells[relation.name][attr.name].append(value)

            return ColSample(sampled_cells)

        def get_config(self) -> dict:
            return super().get_config() | {"weighted": self._weighted}

class Serializers(object):

    class AbstractSerializer(object):
        def __init__(self, hints_to_include: list[Hint]):
            self.hints_to_include = hints_to_include

        @abstractmethod
        def serialize(self, relations: list[Relation], data_sample = None):
            pass

        def set_seed(self, seed: int):
            self.seed = seed
            self.rng = np.random.default_rng(seed=self.seed)

        def get_config(self) -> dict:
            return {"type": type(self).__name__, "hints": [hint.name for hint in self.hints_to_include],
                    "seed": self.seed}

    class JSON(AbstractSerializer):
        def __init__(self, hints_to_include: list[Hint], pretty_print: bool):
            super().__init__(hints_to_include)
            self.pretty_print = pretty_print

        def serialize(self, relations: list[Relation], data_sample = None):

            schema_json = []
            for relation in relations:

                '''
                    Relation-level hints
                '''
                relation_json = OrderedDict({"relation": relation.name,
                                 "description": None,
                                 "attributes": [],
                                 "primary key": list(relation.primary_key),
                                 "unique constraints": list(relation.unique_constraints),
                                 "foreign keys": None})

                if Hint.TABLE_DESCRIPTION in self.hints_to_include and relation.description is not None:
                    relation_json["description"] = relation.description
                else:
                    del relation_json["description"]

                if Hint.CONSTRAINT_FK in self.hints_to_include and len(relation.foreign_keys) > 0:
                    relation_json["foreign keys"] = [{"from_attributes": fk.from_attributes,
                                                      "to_relation": fk.to_relation,
                                                      "to_attributes": fk.to_attributes} for fk in relation.foreign_keys]
                else:
                    del relation_json["foreign keys"]

                if len(relation.unique_constraints) == 0:
                    del relation_json["unique constraints"]

                '''
                    Attribute-level hints
                '''
                for attr in relation.attributes:
                    attr_json = {
                        "name": attr.name,
                        "type": attr.data_type
                    }

                    if attr.data_size is not None:
                        attr_json["type"] = f"{attr_json['type']}({attr.data_size})"

                    if Hint.CONSTRAINT_NULLABLE in self.hints_to_include:
                        attr_json["nullable"] = str(attr.nullable)
                    if Hint.ATTR_DESCRIPTION in self.hints_to_include and attr.description is not None:
                        attr_json["description"] = attr.description

                    # Add columnwise samples
                    if Hint.SAMPLE_DATA in self.hints_to_include and type(data_sample) is ColSample:
                        attr_json['sample data'] = data_sample.samples[relation.name][attr.name]

                    relation_json["attributes"].append(attr_json)

                # Add rowwise samples
                if Hint.SAMPLE_DATA in self.hints_to_include and type(data_sample) is RowSample:
                    relation_json['sample data'] = data_sample.samples[relation.name]

                schema_json.append(relation_json)

            if self.pretty_print:
                return json.dumps(schema_json, indent=4)
            else:
                return json.dumps(schema_json)

class PromptBuilder(object):
    def __init__(self, db_manager: DatabaseManager, prompt_template: AbstractPromptTemplate,
                 serializer: Serializers.AbstractSerializer,
                 seed: int, sampler: Samplers.AbstractSampler = None,
                 shuffle_relations: bool = True, shuffle_attributes: bool = True):
        self._db_manager = db_manager
        self._serializer = serializer
        self._sampler = sampler

        self._seed = seed
        self._serializer.set_seed(self._seed)
        if self._sampler is not None:
            self._sampler.set_seed(self._seed)

        self.shuffle_relations = shuffle_relations
        self.shuffle_attributes = shuffle_attributes

        self._prompt_template = prompt_template

    def _serialize_db(self, namespace: str, relations_to_include: list[str]) -> str:
        assert Hint.SAMPLE_DATA not in self._serializer.hints_to_include or self._sampler is not None, \
            "DATA specified as hint, but no Sampler was given"

        relations = copy.deepcopy(
            list(self._db_manager.namespace_to_schema[namespace].relations.values())) # DO A DEEP COPY FIRST

        # Shuffle relations and attributes
        if self.shuffle_relations:
            rng = random.Random(self._seed)
            rng.shuffle(relations)

        if self.shuffle_attributes:
            for r in relations:
                rng = random.Random(self._seed)
                rng.shuffle(r.attributes)

        # Drop relations
        relations = [r for r in relations if r.name in relations_to_include]

        # Sample
        if self._sampler is not None:
            data_samples = self._sampler.sample(self._db_manager.connection,
                                                relations, namespace)
        else:
            data_samples = None

        # Pass samples into the serialization method
        return self._serializer.serialize(relations, data_samples)

    def get_prompt_content(self, source_relations: list[str], target_relations: list[str]):

        serialized_source = self._serialize_db("source", relations_to_include=source_relations)
        serialized_target = self._serialize_db("target", relations_to_include=target_relations)

        return [{"role": "system", "content": self._prompt_template.SYSTEM},
                {"role": "user", "content":
                    self._prompt_template.USER.format(source_schema=serialized_source, target_schema=serialized_target)}]

    def get_config(self)  -> dict:
        config = {"seed": self._seed, "PromptTemplate": self._prompt_template.get_config(),
                  "DatabaseManager": self._db_manager.get_config(),
                  "Serializer": self._serializer.get_config(),
                  "shuffle_relations": self.shuffle_relations,
                  "shuffle_attributes": self.shuffle_attributes}
        if self._sampler is not None:
            config["Sampler"] = self._sampler.get_config()
        else:
            config["Sampler"] = None

        return config
