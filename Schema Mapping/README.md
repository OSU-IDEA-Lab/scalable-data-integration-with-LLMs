# Schema Mapping

Before running any code, paths must be specified for your local machine. This is done by creating a `config.yaml` file
in the root directory. You can use this template to get started:
```YAML
---
paths:
  datasets: <root of datasets directory>
  temp_dbs: <where temporary DBs are saved>
  hugging_face_cache: <where transformer models should be cached>
  output: <root of output directory; prompts and experiments are placed here>
```

## Prompt Generation
Prompts are generated using generate_prompts.py. Here are the available options:

* --clusters {ground_truth, multimap_N, full}: Specify how relations should be split up into prompts
* --dataset {amalgam}: Currently, just amalgam 
* --source SOURCE {a1,a3,dblp}: For valid combinations of source and target, see the ground-truth mappings available
* --target TARGET {a1,a2,a3,a4}
* --seed SEED: Random seed for prompt generation (affects data value sampling and table/attribute reordering)
* --prompt TEMPLATE: See prompt_templates.py for available templates {Datalog,JSON_original,SQL,SQL_IgnorePKs,SQLite}
* --serializer {JSON}: How to serialize databases. Only JSON is currently supported
* --hints {data,c_fk,c_null,tbl_desc,attr_desc} \[{data,c_fk,c_null,tbl_desc,attr_desc} ...\]\]: Hints to include in the prompt
* \[--pp PP\]: {True,False} 
* \[--shuffle_relations SHUFFLE_RELATIONS\]: {True,False}
* \[--shuffle_attributes SHUFFLE_ATTRIBUTES\]: {True,False}
* \[--sampler {ColumnWiseSampler,RowWiseSampler}\]
* \[--sample_size SAMPLE_SIZE\]
* \[--weighted WEIGHTED\]: {True,False}

## Run LLM
After generating batches of prompts, you can feed them to an LLM using run_llm.py. Here are the available options:
--input_file INPUT_FILE: JSON file containing prompts generating from the step above.
--model {Llama3_1_70B_GPTQ,Llama3_1_8B_GPTQ}: more models can be added by modifying the llm_classes.py file.
--temperature TEMPERATURE: Sampling temperature
--top_p TOP_P: TOP_P for nucleus sampling
--max_new_tokens MAX_NEW_TOKENS: Cutoff point for number of tokens generated

## SQLite3 and STRICT Tables
Each version of Python is bundled with a specific version of sqlite3. However, the ["STRICT" keyword](https://sqlite.org/stricttables.html) (which enforces column typing) is only available in newer versions of sqlite (>=3.37.0).
You can check your version with print(sqlite3.sqlite_version).
You can upgrade your version of sqlite3 without fudging with your Python version by downloading the newest version of SQLite (https://www.sqlite.org/download.html) and then replacing sqlite3.dll (wherever your Python DDLs are kept) with the new version you just downloaded.