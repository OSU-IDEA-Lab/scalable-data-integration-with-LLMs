import os
import torch
import getpass
import warnings
from DatabaseUtils.ConfigReader import get_config
from timeit import default_timer as timer


# Set the HF_HOME environment variable
HF_CACHE = get_config()["paths"]["hugging_face_cache"]
os.environ['HF_HOME'] = HF_CACHE

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

try:
    from auto_gptq import exllama_set_max_input_length
except ImportError:
    warnings.warn('auto_gptq.exllama_set_max_input_length() not found. Ignore this issue if not running Llama3.1 .')

class AbstractModel(object):
    def __init__(self, model_id, temperature, top_p, max_new_tokens, seed):
        self.model_id = model_id
        self.temperature = temperature
        self.seed = seed
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        self._set_environment()
        self.model = self._get_model()
        self.tokenizer = self._get_tokenizer()
        self.pipeline = self._create_pipeline()

        # can check which models support flash attn : https://huggingface.co/docs/transformers/en/perf_infer_gpu_one
        self.decoding_strategy = {
            'do_sample': True,
            'top_p': self.top_p,
            'temperature': self.temperature,
            'max_new_tokens': self.max_new_tokens
        }

    def _set_environment(self):
        username = getpass.getuser()
        torch.random.manual_seed(self.seed)

    def _get_model(self):
        kwargs = {
            'device_map': "cuda",
            'torch_dtype': "auto",
            'trust_remote_code': True
        }

        # Add attn_implementation only if flash is True
        if self.flash:
            kwargs['attn_implementation'] = "flash_attention_2"

        return AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

    def _get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_id)

    def _create_pipeline(self):
        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def apply_template(self, prompt_content):

        if prompt_content[-1]["role"] != "assistant":
            prompt_content.append({"role": "assistant", "content": ""})

        return self.tokenizer.apply_chat_template(prompt_content, tokenize=False)

    def prompt_model(self, prompt):
        start_time = timer()
        full_output = self.pipeline(
            prompt,
            **self.decoding_strategy
        )
        end_time = timer()
        
        responses = []

        prompt_tokens = 0
        response_tokens = 0

        for i, prompt_text in enumerate(prompt):

            generated_text = full_output[i][0]['generated_text']
            resp = generated_text[len(prompt_text):]

            prompt_tokens += len(self.tokenizer.tokenize(prompt_text))
            response_tokens += len(self.tokenizer.tokenize(resp))

            responses.append(resp)

        return dict(response=responses, time=end_time-start_time, total_prompt_tokens=prompt_tokens, total_response_tokens=response_tokens)

    def get_config(self):

        return {"model_id": self.model_id,
                "seed": self.seed,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_new_tokens": self.max_new_tokens,
                "model": type(self.model).__name__,
                "tokenizer": type(self.tokenizer).__name__}

class Llama3_1_70B_GPTQ(AbstractModel):
    def __init__(self, **kwargs):
        self.flash = False
        super().__init__("hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4", **kwargs)
        # self.model = exllama_set_max_input_length(self.model, 4096)

class Llama3_1_8B_GPTQ(AbstractModel):
    def __init__(self, **kwargs):
        self.flash = False
        super().__init__("hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4", **kwargs)
        # self.model = exllama_set_max_input_length(self.model, 4096)
