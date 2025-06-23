from typing import Optional

class Attribute(object):
    def __init__(self, name: str, data_type: str, data_size: int, description: str = None,
                 nullable: bool = True, meaningful: bool = True):
        self.name: str = name
        self.description : str = description

        # Constraints
        self.data_type: str = data_type
        self.data_size: int = data_size
        self.nullable: bool = nullable

        self.meaningful: bool = meaningful

    def get_supertype(self):
        return {"text": "TEXT",
                "varchar": "TEXT",
                "char": "TEXT",
                "int": "INTEGER",
                "float": "DOUBLE",
                "real": "DOUBLE"}[self.data_type.lower()]

    def as_sql(self, with_constraints: bool, as_sqlite: bool) -> str:

        if not with_constraints:
            dtype = "TEXT"
        elif as_sqlite:
            dtype = self.get_supertype()
        else:
            dtype = self.data_type

        base_str = f"{self.name} {dtype}"

        if with_constraints:
            if self.data_size is not None and not as_sqlite:
                base_str += f"({self.data_size})"
            if not self.nullable:
                base_str += f" NOT NULL"

        return base_str

class ForeignKey(object):
    def __init__(self, from_relation: str, from_attributes: list[str],
                 to_relation: str, to_attributes: list[str], name: str = None):
        self.from_relation: str = from_relation
        self.from_attributes: list[str] = from_attributes
        self.to_relation: str = to_relation
        self.to_attributes: list[str] = to_attributes
        self.name: str = name

    def as_sql(self, schema_namespace: str = None) -> str:

        if schema_namespace is not None:
            to_rel = f"{schema_namespace}.{self.to_relation}"
        else:
            to_rel = self.to_relation

        return f"FOREIGN KEY ({', '.join(self.from_attributes)}) REFERENCES {to_rel} ({', '.join(self.to_attributes)})"

class Relation(object):
    def __init__(self, name):
        self.name: str = name
        self.attributes: list[Attribute] = []
        self.primary_key: list[str] = list()
        self.foreign_keys: list[ForeignKey] = []
        self.description: str = None

        self.unique_constraints: list[list[str]] = []

    def contains_attribute_name(self, name: str) -> bool:
        return True if self.get_attribute(name) is not None else False

    def add_attribute(self, attribute: Attribute):
        assert not self.contains_attribute_name(attribute.name), \
            f"Attribute ({attribute.name}) already exists in relation ({self.name})"
        self.attributes.append(attribute)

    def add_attribute_to_primary_key(self, attribute_name: str):
        assert self.contains_attribute_name(attribute_name), \
            (f"Attribute ({attribute_name}) not found in relation ({self.name}) "
             f"(cannot be part of primary key for {self.name})")
        self.get_attribute(attribute_name).nullable = False
        self.primary_key.append(attribute_name)

    def add_description(self, description: str):
        assert self.description is None, f"{self.name} already has description ({self.description})"
        self.description = description

    def add_foreign_key(self, foreign_key: ForeignKey):
        self.foreign_keys.append(foreign_key)

    def add_unique_constraint(self, attribute_names: list[str]):
        for attr_name in attribute_names:
            assert self.contains_attribute_name(attr_name), \
                (f"Attribute ({attr_name}) not found in relation ({self.name}) "
                 f"(cannot be part of unique constraint for {self.name})")
        self.unique_constraints.append(attribute_names)

    def get_attribute(self, attribute_name: str) -> Optional[Attribute]:
        for attribute in self.attributes:
            if attribute.name == attribute_name:
                return attribute
        return None

    def as_sql(self, schema_namespace: str, with_constraints: bool, as_sqlite: bool) -> str:
        table_name = f"{schema_namespace}.{self.name}"

        # Add attributes
        parts = [f"\t{attr.as_sql(with_constraints, as_sqlite=as_sqlite)}" for attr in self.attributes]

        # Unique constraints
        if with_constraints:
            if len(self.unique_constraints) > 0:
                for constraint in self.unique_constraints:
                    parts += [f"\tUNIQUE ({', '.join(constraint)})"]

        # Add primary key
        if with_constraints:
            if len(self.primary_key) > 0:
                parts += [f"\tPRIMARY KEY ({', '.join(self.primary_key)})"]

        # Add foreign keys
        if len(self.foreign_keys) > 0:
            parts += [f"\t{fk.as_sql()}" for fk in self.foreign_keys]

        if with_constraints:
            eos = ") STRICT;"
        else:
            eos = ");"

        return "\n".join([f"CREATE TABLE {table_name} (",
                          ",\n".join(parts),
                          eos])

    def as_dict(self):
        # For sqlglot query rewriting. Output as "{db: {table: {col: type}}}"
        return {attr.name: attr.data_type for attr in self.attributes}

class Schema(object):
    def __init__(self, dataset_name: str, database_name: str):
        self.relations: dict[str, Relation] = dict()
        self.dataset_name: str = dataset_name
        self.database_name: str = database_name

    def add_attribute(self, relation_name: str, attribute: Attribute):
        if relation_name not in self.relations:
            self.relations[relation_name] = Relation(relation_name)
        self.relations[relation_name].add_attribute(attribute)

    def add_relation_description(self, relation_name: str, description: str):
        assert relation_name in self.relations, f"Relation ({relation_name}) not found in schema (must be one of {[rel.name for rel in self.relations]}"
        self.relations[relation_name].add_description(description)

    def add_unique_constraint(self, relation_name, attribute_names: list[str]):
        assert relation_name in self.relations, f"Relation ({relation_name}) not found in schema (must be one of {[rel.name for rel in self.relations]}"
        self.relations[relation_name].add_unique_constraint(attribute_names)

    def add_attribute_to_primary_key(self, relation_name: str, attribute_name: str):
        assert relation_name in self.relations, f"Relation ({relation_name}) not found in schema (must be one of {[rel.name for rel in self.relations]}"
        self.relations[relation_name].add_attribute_to_primary_key(attribute_name)

    def add_foreign_key(self, from_relation_name: str,
                        from_attribute_names: list[str],
                        to_relation_name: str,
                        to_attribute_names: list[str],
                        name: str = None):
        assert from_relation_name in self.relations, f"Relation ({from_relation_name}) not found in schema (must be one of {[rel for rel in self.relations]}"
        assert to_relation_name in self.relations, f"Relation ({to_relation_name}) not found in schema (must be one of {[rel for rel in self.relations]}"
        assert len(from_attribute_names) != 0, f"From (referee) must have at least one attribute for foreign key"
        assert len(to_attribute_names) != 0, f"To (refereed) must have at least one attribute for foreign key"
        assert len(from_attribute_names) == len(set(from_attribute_names)), "From (referee) must have unique attribute names for foreign key"
        assert len(to_attribute_names) == len(set(to_attribute_names)), "To (refereed) must have unique attribute names for foreign key"

        from_relation = self.relations[from_relation_name]
        to_relation = self.relations[to_relation_name]

        # Get attribute references
        from_attrs: list[str] = []
        to_attrs: list[str] = []
        for from_name, to_name in zip(from_attribute_names, to_attribute_names):
            from_attr = from_relation.get_attribute(from_name)
            to_attr = to_relation.get_attribute(to_name)

            assert from_attr is not None, f"From attribute cannot be None for FK ({from_relation_name}->{to_relation_name})"
            assert to_attr is not None, f"To attribute cannot be None for FK ({from_relation_name}->{to_relation_name})"
            assert from_attr.data_type == to_attr.data_type, (f"Expected {from_attr.name}({from_attr.data_type}) and "
                                                            f"{to_attr.name}({to_attr.data_type}) to be same datatype"
                                                            f"for FK ({from_relation_name}->{to_relation_name})")

            from_attrs.append(from_attr.name)
            to_attrs.append(to_attr.name)

        from_relation.add_foreign_key(ForeignKey(from_relation.name, from_attrs, to_relation.name, to_attrs, name))

    def as_sql(self, schema_namespace: str, with_constraints: bool, subset: list[str] = None,
               overwrite_tables: bool = False, as_sqlite: bool = True) -> str:
        return "\n\n".join([relation.as_sql(schema_namespace, as_sqlite=as_sqlite, with_constraints=with_constraints) for relation in self.relations.values()
                            if subset is None or relation in subset])

    def as_dict(self):
        # For sqlglot query rewriting. Output as "{db: {table: {col: type}}}"
        return {rel_name: rel.as_dict() for rel_name, rel in self.relations.items()}