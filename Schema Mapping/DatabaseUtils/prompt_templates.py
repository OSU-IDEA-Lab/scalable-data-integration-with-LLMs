from typing import NamedTuple

class AbstractPromptTemplate(NamedTuple):
    def get_config(self):
        return {"type": type(self).__name__}

class JSON_original(AbstractPromptTemplate):
    SYSTEM = ("Act as a schema mapper for relational schemas. Your task is to generate an SQL script that moves "
                   "data from the source database to the target database. "
                   "I will provide the information of tables in the source database and the target database.")

    USER = ("Source database schema:\n{source_schema}\n\n"
                 "Target database schema:\n{target_schema}\n\n"
                 "Let’s work this out step-by-step to make sure we get it correct. Note the following,\n"
                 " - Source tables must be specified using source.relation_name.\n"
                 " - Target tables must be specified using target.relation_name.\n"
                 " - Some attributes in the source database may not have any corresponding attributes in the target database. "
                 "In this case, data from these attributes should not be moved.")

class SQL(AbstractPromptTemplate):

    SYSTEM = (
        "Act as an expert schema mapper for relational databases. Your task is to generate a complete and correct SQL script "
        "that migrates data from the Source Database to the Target Database. "
        "You will be given details of the Source Tables and the Target Tables, including information about their attributes.\n"
        "Ensure that your script follows SQL best practices and that you explain your mapping choices step-by-step."
    )

    USER = (
        "Source Tables:\n{source_schema}\n\n"
        "Target Tables:\n{target_schema}\n\n"
        "Please generate an SQL script to transfer data from the Source Tables to the Target Tables. Work through each SQL statement step-by-step and follow these directives:\n"
        " 1. Write INSERT statements in the format: INSERT INTO target.target_table SELECT ...\n"
        " 2. Only migrate data from a source attribute to a target attribute if they are conceptually and semantically similar; ignore any extra source attributes.\n"
        " 3. Provide the final SQL code in markdown format using a code block labeled with ```sql.\n"
        "Provide a detailed explanation of your mapping logic followed by the final SQL code."
    )

class SQLite(AbstractPromptTemplate):

    SYSTEM = (
        "Act as an expert schema mapper for relational databases. Your task is to generate a complete and correct SQLite script "
        "that migrates data from the Source Database to the Target Database. "
        "You will be given details of the Source Tables and the Target Tables, including information about their attributes.\n"
        "Ensure that your script follows SQL best practices and that you explain your mapping choices step-by-step."
    )

    USER = (
        "Source Tables:\n{source_schema}\n\n"
        "Target Tables:\n{target_schema}\n\n"
        "Please generate an SQLite script to transfer data from the Source Tables to the Target Tables. Work through each SQL statement step-by-step and follow these directives:\n"
        " 1. Write INSERT statements in the format: INSERT INTO target.target_table SELECT ...\n"
        " 2. Only migrate data from a source attribute to a target attribute if they are conceptually and semantically similar; ignore any extra source attributes.\n"
        " 3. Provide the final SQLite code in markdown format using a code block labeled with ```sql. No other ```sql blocks should appear afterwards.\n"
        "Provide a detailed explanation of your mapping logic followed by the final SQLite code."
    )

class SQL_IgnorePKs(AbstractPromptTemplate):
    '''
        Adds extra rule for ignoring arbitrary primary keys. Alternatively... can just remove int PKs from target (preprocessing)?
    '''
    SYSTEM = (
        "Act as an expert schema mapper for relational databases. Your task is to generate a complete and correct SQL script "
        "that migrates data from the Source Database to the Target Database. "
        "You will be given details of the Source Tables and the Target Tables, including information about their attributes.\n"
        "Ensure that your script follows SQL best practices and that you explain your mapping choices step-by-step."
    )

    USER = (
        "Source Tables:\n{source_schema}\n\n"
        "Target Tables:\n{target_schema}\n\n"
        "Please generate an SQL script to transfer data from the Source Tables to the Target Tables. Work through each SQL statement step-by-step and follow these directives:\n"
        " 1. Write INSERT statements in the format: INSERT INTO target.target_table SELECT ...\n"
        " 2. Only migrate data from a source attribute to a target attribute if they are conceptually and semantically similar; ignore any extra source attributes.\n"
        " 3. Always follow rule #2, even if it leads to NULL primary keys in the target tables.\n" 
        " 4. Provide the final SQL code in markdown format using a code block labeled with ```sql.\n"
        "Provide a detailed explanation of your mapping logic followed by the final SQL code."
    )

class Datalog(AbstractPromptTemplate):
    SYSTEM = (
        "Act as an expert Datalog programmer. Your task is to generate a complete and correct set of Datalog rules "
        "that derive the Intensional Tables based on the Extensional Tables. "
        "You will be given details of the Intensional Tables and the Extensional Tables, including information about their attributes.\n"
        "Ensure that your Datalog rules are logically correct, follow standard Datalog syntax, and that you explain your logical choices step-by-step."
    )

    USER = (
        "Extensional Tables:\n{source_schema}\n\n"
        "Intensional Tables:\n{target_schema}\n\n"
        "Please generate Datalog rules to derive (deduce) the Intensional Tables given the Extensional Tables. Work through each rule step-by-step and follow these directives:\n"
        " 1. Write datalog rules in the format: intensional_relation(...) :- extensional_relation1(...), extensional_relation2(...), ...\n"
        " 2. Only map an extensional attribute to an intensional attribute if they are conceptually and semantically similar; ignore any extra extensional attributes.\n"
        " 3. Provide the final Datalog code in markdown format using a code block labeled with ```datalog.\n"
        "Provide a detailed explanation of your mapping logic followed by the final Datalog code."
    )