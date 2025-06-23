class REPR_TYPE:
    # SOURCE_TARGET = 'source_target'
    # VIEW_NULL = 'view_null'
    # VIEW_NULL_ROWS = 'view_null_rows'
    # VIEW_NULL_COL_EX = 'view_null_col_ex'
    # SQL = "sql"
    # SCHEMA_ALIGN_JSON = 'schema_align_json'
    # SCHEMA_ALIGN_JSON_COL_EX = 'schema_align_json_col_ex'

    N2M_NL = 'json_dynamic'
    N2M_JSON = 'json_schema_dynamic'
    N2One_Json = 'n_to_one_json_dynamic'
    N2One_NL = 'n_to_one_NL_dynamic'
    One2N_Json = 'one_to_n_json_dynamic'
    One2N_NL = 'one_to_n_NL_dynamic'
    ConfidenceScore = 'confidence_score'
    MMConfidence = 'confidence_mm'
    MMConfidence2 = 'confidence2_mm'
    MMCandidate = 'candidate_mm'
    MM_MCQ_Formatter = 'formatter_mm'
    MM_Evaluator = 'eval_mm'
    LogitsConfidenceScoringPrompt = 'logits'
    CoTLogitsPrompt = "cot_logits_dynamic"


    TaDa = 'tada'


class COL_INFO:
    NAME = 'name'
    TYPE = 'type'
    DATA = 'data'
    COL_DESC = 'cdesc'
    TABLE_DESC = 'tdesc'

class SELECTOR_TYPE:
    RANDOM = "random"
    NULL_COVERAGE = "null_coverage"
    MatchMaker = "mm"

class DataInstanceSelector:
    Random = 'random'
    Random_Unique = 'random_unique'
    Most_Frequent = 'most_frequent'
    Weighted_Sampler = 'weighted_sampler'


class STOP_TOKEN:
    SQL = ";"
    JSON = "}"
    _None_ = "None"

class GUIDELINE:
    COLUMN_TYPE = "type"
    DATA_VALUES = "null_coverage"

# https://chat.lmsys.org/?leaderboard
class LLM:
    #gpt4 score 1338 rank 2
    # DeepSeekv2_5 = "deepseek-ai/DeepSeek-V2.5"  # Model size 236B params rank 16 score 1256
    #Meta-Llama-3.1-405b-Instruct-fp8 score 1270 rank 9
    Gemma2 = "google/gemma-2-27b-it" #rank 35 score 1213
    Qwen2_5_32B = "Qwen/Qwen2.5-32B-Instruct"  #Qwen2.5-72b-Instruct ranked 11
    Llama3_1_GPTQ = "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4" #11.3B params
    Phi3_5_mini = "microsoft/Phi-3.5-mini-instruct" #says it do better than llama3.1 8B
    Mistralai3 = "mistralai/Ministral-8B-Instruct-2410"  #says it performs better than llama3.1 8B
    Phi3 = "microsoft/Phi-3-medium-128k-instruct" #14B params
    Phi3_5 = "microsoft/Phi-3.5-MoE-instruct" #41.9B params >>> probably doesn't fit
    DeepSeekCoderLiteInstruct = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" #15.7B params

    Llama3p1_8 = 'meta-llama/Meta-Llama-3.1-8B-Instruct'  #score 1137 rank 60




    # DeepSeekCoderInstruct = "deepseek-ai/DeepSeek-Coder-V2-Instruct"
    # DeepSeekCoderBase = "deepseek-ai/deepseek-llm-7b-base"
    #gated repo
    # CodeLlama7 = "meta-llama/CodeLlama-7b-Instruct-hf"
    CodeLlama13GPTQ = 'TheBloke/CodeLlama-13B-Instruct-GPTQ'
    # CodeLlama13 = "meta-llama/CodeLlama-13b-Instruct-hf"


    # Phi3_3p8 = "microsoft/Phi-3-mini-4k-instruct"
