import os
import torch
import getpass
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig
import json
import time
import warnings

try:
    from auto_gptq import exllama_set_max_input_length
except ImportError:
    warnings.warn('auto_gptq.exllama_set_max_input_length() not found. Ignore this issue if not running Llama3.1 .')
from utils.enums import LLM, STOP_TOKEN

import torch.nn.functional as F
from transformers import StoppingCriteria, StoppingCriteriaList


class SemicolonStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_token):
        self.tokenizer = tokenizer
        self.stop_token = stop_token

    def __call__(self, input_ids, scores, **kwargs):
        # Get the last token generated
        last_token_id = input_ids[0, -1].item()
        # Convert the token ID to the actual token (string)
        last_token = self.tokenizer.decode(last_token_id)
        # Stop if the token is a semicolon
        return last_token == self.stop_token


class BasicModel:
    def __init__(self, model_id, temperature, top_p, max_new_tokens, stop_token, seed=0, is_MM="False"):
        self.model_id = model_id
        self.temperature = temperature
        self.seed = seed
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.set_environment()
        self.model = self.get_model()
        self.tokenizer = self.get_tokenizer()
        self.do_sample = True
        # can check which models support flash attn : https://huggingface.co/docs/transformers/en/perf_infer_gpu_one

        if is_MM == "False":
            self.decoding_strategy = {
                'do_sample': self.do_sample,
                'top_p': self.top_p,
                'temperature': self.temperature,
                'max_new_tokens': self.max_new_tokens
            }
        else:
            self.decoding_strategy = {'temperature': 0.5, 'max_new_tokens': 1024, 'top_p': 1 }

        if stop_token != STOP_TOKEN._None_:
            semicolon_criteria = SemicolonStoppingCriteria(self.tokenizer, stop_token)
            stopping_criteria = StoppingCriteriaList([semicolon_criteria])
            self.decoding_strategy['stopping_criteria'] = stopping_criteria

    def set_environment(self):
        # Set the HF_HOME environment variable
        username = getpass.getuser()
        os.environ['HF_HOME'] = f'/nfs/stak/users/{username}/hpc-share/huggingface'
        torch.random.manual_seed(self.seed)

    def get_model(self):
        kwargs = {
            'device_map': "cuda",
            'torch_dtype': "auto",
            'trust_remote_code': True
        }

        # Add attn_implementation only if flash is True
        if self.flash:
            kwargs['attn_implementation'] = "flash_attention_2"

        print(f"Loading model: {self.model_id}")
        try:
            return AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_id)

    def create_pipeline(self):
        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def call_llm_with_prompt(self, my_pipeline, prompt, greedy = False):


        if greedy:
            self.do_sample = False
            self.temperature = 1
            self.top_p = 1
            self.decoding_strategy = {
                'do_sample': self.do_sample,
                'temperature': self.temperature,
                'top_p': self.top_p,
                'max_new_tokens': self.max_new_tokens
            }

        # print('-------------------------------self.do_sample :', self.do_sample)

        full_output = my_pipeline(
            prompt,
            **self.decoding_strategy
        )
        response_clean = []

        for i, prompt_text in enumerate(prompt):
            generated_text = full_output[i][0]['generated_text']
            clean_text = generated_text[len(prompt_text):]
            # print("-------------------prompt:")
            # print(generated_text[:len(prompt_text)])
            # print()
            # print("________________________answer:")
            # print(clean_text)
            # print()
            response_clean.append(clean_text)

        return dict(response=response_clean)

    def attempt_llm_request(self, batch, greedy = False):
        n_repeat = 0
        my_pipeline = self.create_pipeline()
        while True:
            try:
                response = self.call_llm_with_prompt(my_pipeline, batch, greedy)
                break
            except Exception as e:
                n_repeat += 1
                print(f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                time.sleep(1)
                continue

        return response


    def attempt_llm_logits(self, prompt, candidates):
        n_repeat = 0
        while True:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits

                probs = F.softmax(logits, dim=-1)
                candidate_scores = []

                for candidate in candidates:
                    candidate_ids = self.tokenizer.encode(candidate)

                    if len(candidate_ids) > logits.shape[1]:
                        print(f"Warning: Model did not generate enough tokens to match '{candidate}'")
                        continue

                    token_probs = [probs[0, i, token_id].item() for i, token_id in enumerate(candidate_ids)]

                    total_log_prob = sum(torch.log(torch.tensor(token_probs)))
                    probability_score = torch.exp(total_log_prob).item()

                    candidate_scores.append([candidate, probability_score])

                # Normalize scores to sum to 1
                total_sum = sum(prob for _, prob in candidate_scores)
                normalized_scores = [[candidate, prob / total_sum] for candidate, prob in candidate_scores]

                break
            except Exception as e:
                n_repeat += 1
                print(f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                time.sleep(1)
                continue

        return normalized_scores




    #
    #
    # def attempt_llm_logits_append_method(self, prompt, candidates):
    #     marker = "Best Match:"
    #
    #     generated_text = self.attempt_llm_request([prompt], greedy=True)["response"][0]
    #
    #
    #     marker_index = generated_text.find(marker)
    #     if marker_index == -1:
    #         print(f"Marker '{marker}' not found in generated text.")
    #         return [], generated_text
    #
    #     base_text = generated_text[:marker_index + len(marker)].strip()
    #
    #     scores = []
    #     self.model.eval()
    #     with torch.no_grad():
    #         for candidate in candidates:
    #             # full_input = prompt + base_text + " " + candidate
    #             full_input = base_text + " " + candidate
    #             inputs = self.tokenizer(full_input, return_tensors="pt")
    #             inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
    #
    #             outputs = self.model(**inputs)
    #             logits = outputs.logits[0]  # shape: (seq_len, vocab_size)
    #
    #             # Get candidate token IDs
    #             candidate_ids = self.tokenizer(candidate, add_special_tokens=False)["input_ids"]
    #             input_ids_tensor = inputs["input_ids"][0]
    #             start_idx = input_ids_tensor.shape[0] - len(candidate_ids)
    #
    #             log_probs = 0.0
    #             for i, token_id in enumerate(candidate_ids):
    #                 token_logits = logits[start_idx + i - 1]  # predicts token at i
    #                 token_log_probs = F.log_softmax(token_logits, dim=-1)
    #                 log_probs += token_log_probs[token_id].item()
    #
    #             scores.append(log_probs)
    #
    #     # with torch.no_grad():
    #     #     for candidate in candidates:
    #     #         full_input = prompt + base_text + " " + candidate
    #     #         tokenized = self.tokenizer(full_input, return_tensors="pt").to(self.model.device)
    #     #         input_ids = tokenized["input_ids"][0]
    #     #
    #     #         outputs = self.model(**tokenized)
    #     #         logits = outputs.logits[0]  # (seq_len, vocab_size)
    #     #
    #     #         # Get log-probabilities for the candidate tokens only
    #     #         candidate_ids = self.tokenizer(candidate, add_special_tokens=False)["input_ids"]
    #     #         start_idx = len(input_ids) - len(candidate_ids)
    #     #
    #     #         log_probs = 0.0
    #     #         for i, token_id in enumerate(candidate_ids):
    #     #             token_logits = logits[start_idx + i - 1]  # logit predicts next token
    #     #             token_log_probs = F.log_softmax(token_logits, dim=-1)
    #     #             log_probs += token_log_probs[token_id].item()
    #     #         print("\n\n\n")
    #     #         print(full_input)
    #     #         print(log_probs)
    #     #         scores.append(log_probs)
    #
    #     probs = F.softmax(torch.tensor(scores), dim=0).tolist()
    #     scored_candidates = list(zip(candidates, probs))
    #
    #     scored_candidates.sort(key=lambda x: x[1], reverse=True)
    #
    #     return scored_candidates, generated_text


    def try_gen(self, prompt, candidates, greedy = True):

        marker = "Best Match:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        if greedy:
            self.decoding_strategy = {
                'do_sample': False,
                'temperature': 1.0,
                'top_p': 1.0,
                'max_new_tokens': self.max_new_tokens
            }

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                return_dict_in_generate=True,
                output_scores=True,
                **self.decoding_strategy
            )

        generated_ids = output.sequences[0][inputs.input_ids.shape[-1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # print('------------prompt: \n', prompt)
        # print('------------generated_text: \n', generated_text)

        marker_token_count = len(
            self.tokenizer.encode(generated_text.split(marker)[0] + marker, add_special_tokens=False))

        logits_per_token = output.scores[marker_token_count:]  # list of tensors, one per generated token
        # generated_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)[marker_token_count:]

        probs = [torch.softmax(logit[0], dim=-1) for logit in logits_per_token]
        # eos_token_id = self.tokenizer.eos_token_id
        candidate_scores = []

        for candidate in candidates:
            candidate_ids = self.tokenizer.encode(" "+candidate, add_special_tokens=False)

            if len(candidate_ids) > len(probs):
                print(f"Warning: Model did not generate enough tokens to match '{candidate}'")
                continue

            token_probs = []
            for i, token_id in enumerate(candidate_ids):
                token_probs.append(probs[i][token_id].item())
            if not token_probs:
                continue

            total_log_prob = sum(torch.log(torch.tensor(token_probs)))
            probability_score = torch.exp(total_log_prob).item()
            candidate_scores.append((candidate, probability_score))


        try:
            total = sum(score for _, score in candidate_scores)
            normalized_scores = [[cand, score / total] for cand, score in candidate_scores if score != 0]
            scored_candidates = sorted(normalized_scores, key=lambda x: x[1], reverse=True)
            return scored_candidates, generated_text
        except Exception as e:
            print(f"{e}!")
            reason = str(e) + "-" + generated_text
            return [], reason

    # def try_gen(self, prompt, candidates, greedy = True):
    #
    #     marker = "Best Match:"
    #     inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    #
    #     if greedy:
    #         self.decoding_strategy = {
    #             'do_sample': False,
    #             'temperature': 1.0,
    #             'top_p': 1.0,
    #             'max_new_tokens': self.max_new_tokens
    #         }
    #
    #     with torch.no_grad():
    #         output = self.model.generate(
    #             **inputs,
    #             return_dict_in_generate=True,
    #             output_scores=True,
    #             **self.decoding_strategy
    #         )
    #
    #     generated_ids = output.sequences[0][inputs.input_ids.shape[-1]:]
    #     generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    #     print('------------generated_text: \n', generated_text)
    #
    #     marker_token_count = len(
    #         self.tokenizer.encode(generated_text.split(marker)[0] + marker, add_special_tokens=False))
    #
    #     logits_per_token = output.scores[marker_token_count:]  # list of tensors, one per generated token
    #     generated_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)[marker_token_count:]
    #
    #     # Get vocab probs for each generated token after the marker
    #     probs = [torch.softmax(logit[0], dim=-1) for logit in logits_per_token]
    #
    #     eos_token_id = self.tokenizer.eos_token_id
    #
    #     candidate_scores = []
    #
    #     for candidate in candidates:
    #         candidate_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
    #
    #         if len(candidate_ids) > len(probs):
    #             print(f"Warning: Model did not generate enough tokens to match '{candidate}'")
    #             continue
    #
    #         token_probs = []
    #
    #         for i, token_id in enumerate(candidate_ids):
    #             # Skip if it's an EOS token
    #             if token_id == eos_token_id:
    #                 continue
    #             token_probs.append(probs[i][token_id].item())
    #
    #         if not token_probs:
    #             continue  # skip empty
    #
    #         total_log_prob = sum(torch.log(torch.tensor(token_probs)))
    #         probability_score = torch.exp(total_log_prob).item()
    #         candidate_scores.append((candidate, probability_score))
    #
    #         print(f"\n\n\n\n\n------------{candidate}------------\nGenerated tokens:", self.tokenizer.convert_ids_to_tokens(generated_ids[marker_token_count:]))
    #         print(f"Candidate '{candidate}' tokens:",
    #               self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(candidate, add_special_tokens=False)))
    #         print("Length of logits_per_token:", len(logits_per_token))
    #         print(f"Length of candidate_ids for '{candidate}':",
    #               len(self.tokenizer.encode('ethnicity', add_special_tokens=False)))
    #
    #     # Normalize
    #     total = sum(score for _, score in candidate_scores)
    #     normalized_scores = [(cand, score / total) for cand, score in candidate_scores]
    #
    #     # Sort and return
    #     scored_candidates = sorted(normalized_scores, key=lambda x: x[1], reverse=True)
    #     print(scored_candidates)
    #     return scored_candidates, generated_text
    #
    # def try_gen(self, prompt, candidates):
    #
    #     inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    #
    #     with torch.no_grad():
    #         output = self.model.generate(
    #             **inputs,
    #             max_new_tokens=800,
    #             return_dict_in_generate=True,
    #             output_scores=True,
    #             do_sample=False  # greedy decoding
    #         )
    #
    #     # Extract the generated token IDs (excluding the input prompt)
    #     generated_ids = output.sequences[0][inputs.input_ids.shape[-1]:]
    #
    #     # Get logits for each generated token
    #     logits_per_token = output.scores  # list of tensors, one per generated token
    #     generated_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)
    #
    #     # Print generated text and top-1 predictions
    #     print("Generated text:", self.tokenizer.decode(generated_ids, skip_special_tokens=True))
    #     for i, (token, logits) in enumerate(zip(generated_tokens, logits_per_token)):
    #         top_token_id = torch.argmax(logits).item()
    #         top_token = self.tokenizer.decode([top_token_id])
    #         print(f"Step {i + 1}: Generated Token = '{token}', Top Logit Token = '{top_token}'")

    def attempt_llm_logits_append_method(self, prompt, candidates):
        generated_text = ''
        marker = "Best Match:"
        print(".................................?????????")

        try:

            generated_text = self.attempt_llm_request([prompt], greedy = True)["response"][0]
            marker_index = generated_text.find(marker)


            inputs = self.tokenizer(generated_text, return_tensors="pt").to(self.model.device)
            print("_?_?_?_0")

            # Forward pass to get logits
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # shape: (1, seq_len, vocab_size)

            print("_?_?_?_1")

            # Decode to find marker position
            marker = "Best Match:"
            marker_token_count = len(self.tokenizer.encode(generated_text.split(marker)[0] + marker, add_special_tokens=False))
            print("_?_?_?_2")
            # Iterate over tokens starting at marker_token_count
            print('marker_token_count: ', marker_token_count)
            print('logits.shape[1]: ', logits.shape[1])
            print(generated_text)
            for i in range(marker_token_count, logits.shape[1]):
                print(".....................................",i)
                token_id = torch.argmax(logits[0, i])  # Highest logit at position i
                token = self.tokenizer.decode(token_id)
                print("<",token,">")




            if marker_index == -1:
                print(f"Marker '{marker}' not found in LLM output.")
                return [], generated_text


            # full_text = prompt + generated_text
            full_text = generated_text
            tokens = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)


            pre_marker_text = full_text[:marker_index + len(marker)]
            marker_token_count = len(self.tokenizer(pre_marker_text, add_special_tokens=False)["input_ids"])


            with torch.no_grad():
                outputs = self.model(**tokens)
            logits = outputs.logits  # shape: (1, seq_len, vocab_size)

            scores = []
            for candidate in candidates:
                token_ids = self.tokenizer.encode(" "+candidate, add_special_tokens=False)
                if marker_token_count + len(token_ids) > logits.shape[1]:
                    print(f"Candidate '{candidate}' exceeds available logits. Skipping.")
                    scores.append(float("-inf"))
                    continue

                log_probs = F.log_softmax(
                    logits[0, marker_token_count:marker_token_count + len(token_ids), :], dim=-1
                )
                token_log_probs = [log_probs[i, token_ids[i]] for i in range(len(token_ids))]
                score = torch.sum(torch.stack(token_log_probs)).item()
                scores.append(score)


            max_score = max(scores)
            exp_scores = [torch.exp(torch.tensor(s - max_score)) for s in scores]
            total = sum(exp_scores)
            normalized_scores = [float(s / total) for s in exp_scores]
            scored_candidates = sorted(zip(candidates, normalized_scores), key=lambda x: x[1], reverse=True)
            return scored_candidates, generated_text

        except Exception as e:
            print(f"Exception during LLM scoring: {e}")
            return [], f"Error : {e}" + "reasoning : "+ generated_text

    # def attempt_llm_logits_append_method(self, prompt, candidates):
    #
    #     marker = "Best Match:"
    #     extra_tokens_after_marker = 30
    #     generated_tokens = []
    #     marker_token_ids = self.tokenizer.encode(marker, add_special_tokens=False)
    #     marker_found = False
    #     marker_index_in_tokens = None
    #     generated_text = ''
    #
    #     try:
    #         # Prepare input
    #         inputs = self.tokenizer(prompt, return_tensors="pt")
    #         input_ids = inputs["input_ids"].to(self.model.device)
    #         generated_input = input_ids
    #
    #         for _ in range(self.max_new_tokens):
    #             with torch.no_grad():
    #                 outputs = self.model(input_ids=generated_input)
    #             next_token_logits = outputs.logits[0, -1, :]
    #             next_token_id = torch.argmax(next_token_logits).unsqueeze(0)
    #
    #             # Append new token to sequence
    #             generated_input = torch.cat([generated_input, next_token_id.unsqueeze(0)], dim=1)
    #             generated_tokens.append(next_token_id.item())
    #
    #             # Check for marker in generated sequence
    #             decoded_so_far = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    #             # print(decoded_so_far)
    #             # print()
    #             if not marker_found and marker in decoded_so_far:
    #                 marker_found = True
    #                 marker_index_in_tokens = len(generated_tokens)
    #                 tokens_after_marker = 0
    #             elif marker_found:
    #                 tokens_after_marker += 1
    #                 if tokens_after_marker >= extra_tokens_after_marker:
    #                     break  # stop after extra N tokens
    #
    #         # Full generated text
    #         generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    #
    #         if not marker_found:
    #             print("Marker 'Best Match:' not found in generated output.")
    #             return [], generated_text
    #
    #         # Score candidates starting from after the marker
    #         full_ids = torch.cat([input_ids, torch.tensor([generated_tokens], device=self.model.device)], dim=1)
    #         with torch.no_grad():
    #             full_outputs = self.model(input_ids=full_ids)
    #         logits = full_outputs.logits  # shape: (1, total_len, vocab_size)
    #
    #         candidate_start_index = input_ids.shape[1] + marker_index_in_tokens
    #         scores = []
    #
    #         for candidate in candidates:
    #             token_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
    #             end_index = candidate_start_index + len(token_ids)
    #
    #             if end_index > logits.shape[1]:
    #                 print(f"Candidate '{candidate}' exceeds available logits. Skipping.")
    #                 scores.append(float('-inf'))
    #                 continue
    #
    #             candidate_logits = logits[0, candidate_start_index:end_index, :]
    #             probs = F.log_softmax(candidate_logits, dim=-1)
    #             token_log_probs = [probs[i, token_ids[i]] for i in range(len(token_ids))]
    #             score = torch.sum(torch.stack(token_log_probs)).item()
    #             scores.append(score)
    #
    #         # Normalize scores
    #         max_score = max(scores)
    #         exp_scores = [torch.exp(torch.tensor(s - max_score)) for s in scores]
    #         total = sum(exp_scores)
    #         normalized_scores = [float(s / total) for s in exp_scores]
    #
    #         return list(zip(candidates, normalized_scores)), generated_text
    #
    #     except Exception as e:
    #         print(f"Exception during LLM scoring: {e}")
    #         return [], f"Error : {e}" + "reasoning : "+ generated_text

class Llama3_1_GPTQ(BasicModel):
    def __init__(self, **kwargs):
        self.flash = True
        super().__init__(LLM.Llama3_1_GPTQ, **kwargs)
        self.model = exllama_set_max_input_length(self.model, 8192)


class Qwen2_5_32B(BasicModel):
    def __init__(self, **kwargs):
        self.flash = False
        super().__init__(LLM.Llama3_1_GPTQ, **kwargs)


class Llama3p1_8(BasicModel):
    def __init__(self, **kwargs):
        self.flash = True
        super().__init__(LLM.Llama3p1_8, **kwargs)


class DeepSeekCoderLiteInstruct(BasicModel):
    def __init__(self, **kwargs):
        self.flash = False
        super().__init__(LLM.DeepSeekCoderLiteInstruct, **kwargs)  # Use enum value


class CodeS(BasicModel):
    def __init__(self, **kwargs):
        self.flash = False
        super().__init__(LLM.CodeS, **kwargs)  # Use enum value


class CodeLlama13GPTQ(BasicModel):
    def __init__(self, **kwargs):
        self.flash = False
        super().__init__(LLM.CodeLlama13GPTQ, **kwargs)  # Use enum value


class Phi3(BasicModel):
    def __init__(self, **kwargs):
        self.flash = True
        super().__init__(LLM.Phi3, **kwargs)  # Use enum value


class Phi3_5(BasicModel):
    def __init__(self, **kwargs):
        self.flash = True
        super().__init__(LLM.Phi3_5, **kwargs)  # Use enum value
#
# class Phi3_3p8(BasicModel):
#     def __init__(self, **kwargs):
#         super().__init__(LLM.Phi3_3p8, **kwargs)  # Use enum value
#
# class StarCoder15(BasicModel):
#     def __init__(self, **kwargs):
#         super().__init__(LLM.StarCoder15, **kwargs)  # Use enum value

# class Phi3_3p8(BasicModel):
#     def __init__(self, **kwargs):
#         super().__init__(LLM.Phi3_3p8, **kwargs)  # Use enum value


# class DeepSeekCoderInstruct(BasicModel):
#     def __init__(self, **kwargs):
#         super().__init__(LLM.DeepSeekCoderInstruct, **kwargs)  # Use enum value
#
#
# class DeepSeekCoderBase(BasicModel):
#     def __init__(self, **kwargs):
#         super().__init__(LLM.DeepSeekCoderBase, **kwargs)  # Use enum value
#
#     def get_model(self):
#         model = AutoModelForCausalLM.from_pretrained(
#             self.model_id,
#             torch_dtype=torch.float16,  # Changed from bfloat16 to float16
#             device_map="auto",
#         )
#         model.generation_config = GenerationConfig.from_pretrained(self.model_id)
#         model.generation_config.pad_token_id = model.generation_config.eos_token_id
#         return model
#
#     def get_tokenizer(self):
#         return AutoTokenizer.from_pretrained(self.model_id)
#
#     def call_llm_with_prompt(self, prompt):
#         model = self.get_model()
#         tokenizer = self.get_tokenizer()
#
#         inputs = tokenizer(prompt, return_tensors="pt")
#         outputs = model.generate(**inputs.to(model.device), max_new_tokens=self.max_new_tokens)
#
#         response_clean = []
#         for i, prompt_text in enumerate(prompt):
#             generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
#             clean_text = generated_text[len(prompt_text):]
#             print("-------------------prompt:")
#             print(generated_text[:len(prompt_text)])
#             print()
#             print("________________________answer:")
#             print(clean_text)
#             print()
#             response_clean.append(clean_text)
#
#         return dict(response=response_clean)
#
#     def attempt_llm_request(self, batch):
#         n_repeat = 0
#         while True:
#             try:
#                 response = self.call_llm_with_prompt(batch)
#                 break
#             except json.decoder.JSONDecodeError:
#                 n_repeat += 1
#                 print(f"Repeat for the {n_repeat} times for JSONDecodeError", end="\n")
#                 time.sleep(1)
#                 continue
#             except Exception as e:
#                 n_repeat += 1
#                 print(f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
#                 time.sleep(1)
#                 continue
#
#         return response
#

#
# class Llama3p1_8Quantized(BasicModel):
#     def __init__(self, **kwargs):
#         super().__init__(LLM.Llama3p1_8Quantized, **kwargs)
#
#     def get_model(self):
#         model_id = self.model_id
#         compute_dtype = torch.float16  # Adjust if needed
#         cache_dir = '.'
#         model = AutoHQQHFModel.from_quantized(model_id, cache_dir=cache_dir, compute_dtype=compute_dtype)
#         quant_config = BaseQuantizeConfig(nbits=4, group_size=64, quant_scale=False, quant_zero=False, axis=1)
#         patch_linearlayers(model, patch_add_quant_config, quant_config)
#         HQQLinear.set_backend(HQQBackend.PYTORCH)
#         prepare_for_inference(model, backend="bitblas")  # Adjust backend if needed
#         return model
#
#
#     def get_tokenizer(self):
#         model_id = self.model_id
#         cache_dir = '.'
#         return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
#
#
#     def create_pipeline(self):
#         model = self.get_model()
#         tokenizer = self.get_tokenizer()
#         gen = HFGenerator(model, tokenizer, max_new_tokens=self.max_new_tokens, do_sample=True, compile="partial").warmup()
#         return gen
#
#
#     def call_llm_with_prompt(self, gen, prompt):
#         response_clean = []
#         for prompt_text in prompt:
#             generated_text = gen.generate(prompt_text, print_tokens=False)
#             clean_text = generated_text[len(prompt_text):]
#             print("-------------------prompt:")
#             print(generated_text[:len(prompt_text)])
#             print()
#             print("________________________answer:")
#             print(clean_text)
#             print()
#             response_clean.append(clean_text)
#
#         return dict(response=response_clean)
#
#
# class Llama3p1_8QuantizedCalib(Llama3p1_8Quantized):
#     def __init__(self, **kwargs):
#         self.model_id = LLM.Llama3p1_8QuantizedCalib
#         super().__init__(**kwargs)
