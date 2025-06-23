from enum import Enum


# What information is included in the prompt
class Hint(Enum):
    SAMPLE_DATA = "data"  # Need to specify loader here

    # Constraints
    CONSTRAINT_FK = "c_fk"
    CONSTRAINT_NULLABLE = "c_null"

    # Textual descriptions
    TABLE_DESCRIPTION = "tbl_desc"
    ATTR_DESCRIPTION = "attr_desc"

class Dataset(Enum):
    AMALGAM = "amalgam"
    EHR = "ehr"

class STOP_TOKEN:
    SQL = ";"
    JSON = "}"
    _None_ = "None"