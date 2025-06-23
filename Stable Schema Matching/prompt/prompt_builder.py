from prompt.MatchMaker import ConfidenceScoring, CandidateRefiner, MCQ_Formatter, Evaluator, ConfidenceScoring2
from utils.enums import REPR_TYPE
from utils.enums import SELECTOR_TYPE
from prompt.PromptReprTemplate import *
from prompt.ExampleSelectorTemplate import *
from prompt.PromptICLTemplate import BasicICLPrompt, BasicScoringPrompt, BaselinePrompt


def get_repr_class(repr_type: str):
    # if repr_type == REPR_TYPE.SOURCE_TARGET:
    #     repr_class = SourceTargetPrompt
    # elif repr_type == REPR_TYPE.VIEW_NULL:
    #     repr_class = view_NULL
    # elif repr_type == REPR_TYPE.VIEW_NULL_ROWS:
    #     repr_class = view_NULL_row_data_instance
    # elif repr_type == REPR_TYPE.VIEW_NULL_COL_EX:
    #     repr_class = view_NULL_Column_Example
    # elif repr_type == REPR_TYPE.SCHEMA_ALIGN_JSON_NO_SQL:
    #     repr_class = schema_align_no_sql
    # elif repr_type == REPR_TYPE.SQL:
    #     repr_class = SQLPrompt
    # elif repr_type == REPR_TYPE.SCHEMA_ALIGN_JSON:
    #     repr_class = schema_align_JSON
    # elif repr_type == REPR_TYPE.SCHEMA_ALIGN_JSON_NO_SQL_WITH_TYPE:
    #     repr_class = schema_align_no_sql_type
    # elif repr_type == REPR_TYPE.SCHEMA_ALIGN_JSON_NO_SQL_WITH_TYPE_VAL:
    #     repr_class = schema_align_no_sql_type_values
    # elif repr_type == REPR_TYPE.SCHEMA_ALIGN_JSON_COL_EX:
    #     repr_class = schema_align_JSON_Column_Example
    if repr_type == REPR_TYPE.ConfidenceScore:
        repr_class = ConfidenceScoringPrompt
    elif repr_type == REPR_TYPE.LogitsConfidenceScoringPrompt:
        repr_class = LogitsConfidenceScoringPrompt
    elif repr_type == REPR_TYPE.CoTLogitsPrompt:
        repr_class = CoTLogitsPrompt
    elif repr_type == REPR_TYPE.TaDa:
        repr_class = TaDa
    elif repr_type == REPR_TYPE.MMConfidence:
        repr_class = ConfidenceScoring
    elif repr_type == REPR_TYPE.MMConfidence2:
        repr_class = ConfidenceScoring2
    elif repr_type == REPR_TYPE.MMCandidate:
        repr_class = CandidateRefiner
    elif repr_type == REPR_TYPE.MM_MCQ_Formatter:
        repr_class = MCQ_Formatter
    elif repr_type == REPR_TYPE.MM_Evaluator:
        repr_class = Evaluator
    # elif repr_type == REPR_TYPE.N2M_NL:
    #     repr_class = N2M_NL
    # elif repr_type == REPR_TYPE.N2M_JSON:
    #     repr_class = N2M_JSON
    elif repr_type == REPR_TYPE.N2One_Json:
        repr_class = N2One_Json
    # elif repr_type == REPR_TYPE.One2N_Json:
    #     repr_class = One2N_Json
    # elif repr_type == REPR_TYPE.N2One_NL:
    #     repr_class = N2One_NL
    # elif repr_type == REPR_TYPE.One2N_NL:
    #     repr_class = One2N_NL
    else:
        raise ValueError(f"{repr_type} is not supported yet")
    return repr_class


def get_example_selector(selector_type: str):
    if selector_type == SELECTOR_TYPE.RANDOM:
        selector_cls = RandomExampleSelector
    elif selector_type == SELECTOR_TYPE.NULL_COVERAGE:
        selector_cls = SimilarNullCoverageExampleSelector
    elif selector_type == SELECTOR_TYPE.MatchMaker:
        selector_cls = MMSelector
    else:
        raise ValueError(f"{selector_type} is not supported yet!")
    return selector_cls

def prompt_factory(repr_type: str, k_shot: int, selector_type: str, isConfidence2:bool):
    repr_cls = get_repr_class(repr_type)

    if k_shot == 0:
        if repr_type == REPR_TYPE.TaDa:
            cls_name = f"{repr_type}"

            class PromptClass(repr_cls, BaselinePrompt):
                name = cls_name
                NUM_EXAMPLE = k_shot

                def __init__(self, *args, **kwargs):
                    repr_cls.__init__(self, *args, **kwargs)
                    BaselinePrompt.__init__(self, *args, **kwargs)


        elif repr_type == REPR_TYPE.ConfidenceScore or repr_type == REPR_TYPE.LogitsConfidenceScoringPrompt:
            cls_name = f"{repr_type}"

            class PromptClass(repr_cls, BasicScoringPrompt):
                name = cls_name
                NUM_EXAMPLE = k_shot

                def __init__(self, *args, **kwargs):
                    repr_cls.__init__(self, *args, **kwargs)
                    BasicScoringPrompt.__init__(self, *args, **kwargs)

        else:
            cls_name = f"{repr_type}"

            class PromptClass(repr_cls, BasicICLPrompt):
                name = cls_name
                NUM_EXAMPLE = k_shot

                def __init__(self, *args, **kwargs):
                    repr_cls.__init__(self, *args, **kwargs)
                    # init tokenizer
                    BasicICLPrompt.__init__(self, *args, **kwargs)
    else:
        selector_cls = get_example_selector(selector_type)

        if isConfidence2:
            cls_name = f"{repr_type}_ICL2"
        else:
            cls_name = f"{repr_type}_ICL"

        class PromptClass(repr_cls, BasicICLPrompt):
            name = cls_name
            NUM_EXAMPLE = k_shot

            def __init__(self, *args, **kwargs):
                repr_cls.__init__(self, *args, **kwargs)
                # init tokenizer
                BasicICLPrompt.__init__(self, example_selector=selector_cls(*args, **kwargs), *args, **kwargs)


    return PromptClass
