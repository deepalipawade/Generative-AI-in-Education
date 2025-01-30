import openai
import project_part1_prompts as prompts
import project_part1_repair as repair
import project_part1_utils as utils

import os
import json

import torch

if torch.cuda.is_available():
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

# Class for generating hints using language models
class Hint:
    # Initialize the Hint class with model details
    def __init__(self, model_name, is_huggingface):
        self.model_name = model_name
        self.system_prompt = prompts.system_message_nus
        self.user_prompt = prompts.user_message_nus_hint_basic
        self.is_huggingface = is_huggingface
        self.user_prompt_with_repair = prompts.user_message_nus_hint_with_repair

        if self.is_huggingface:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name, 
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            self.tokenizer = get_chat_template(
                self.tokenizer,
                chat_template="chatml",
            )
            FastLanguageModel.for_inference(self.model)
        else:
            with open("176_Pawade_openai.txt") as f:
                content = f.read().strip()
            openai_key = content.split("\n")[2]
            openai.api_key = openai_key

    # Extract hint from the generated text
    def extract_hint(self, text):
        start_tag = "[HINT]"
        end_tag = "[/HINT]"
        
        start_index = text.find(start_tag)
        if start_index == -1:
            return ""
        
        start_index += len(start_tag)
        end_index = text.find(end_tag, start_index)
        if end_index == -1:
            end_index = len(text)
        
        extracted_text = text[start_index:end_index].strip()
        return extracted_text

    # Save the transcript to a JSON file at "project_part1_transcripts/transcript.json". This file contains all prompts and LLM responses which can be used for debugging.
    def save_transcript_to_json(self, transcript):
        os.makedirs("project_part1_transcripts", exist_ok=True)
        file_path = os.path.join("project_part1_transcripts", "transcript.json")
        
        # Read existing data
        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                try:
                    existing_data = json.load(json_file)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []
        
        # Append new transcript data
        existing_data.extend(transcript)
        
        # Write back to the file
        with open(file_path, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)

    # Call the OpenAI language model
    def call_llm_openai(self, system_prompt_formatted, user_prompt_formatted):
        response = openai.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt_formatted},
                {"role": "user", "content": user_prompt_formatted}
            ]
        )
        
        return response.choices[0].message.content

    # Call the Hugging Face language model
    def call_llm_huggingface(self, system_prompt_formatted, user_prompt_formatted):
        prompt_string = f"""<|system|>\n{system_prompt_formatted}<|end|>\n<|user|>\n{user_prompt_formatted}<|end|>\n<|assistant|>"""        
        inputs = self.tokenizer(prompt_string, return_tensors="pt", padding=True, truncation=True).to("cuda")
        
        outputs = self.model.generate(
            input_ids=inputs.input_ids, 
            attention_mask=inputs.attention_mask, 
            max_new_tokens=2048, 
            use_cache=True
        )
        
        response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        return response_text

    # Call the appropriate language model based on configuration        
    def call_llm(self, problem_data, buggy_program):
        system_prompt_formatted = self.system_prompt
        user_prompt_formatted = self.user_prompt.format(problem_data=problem_data, buggy_program=buggy_program)
        
        transcript = []
        
        if self.is_huggingface:
            generated_response = self.call_llm_huggingface(system_prompt_formatted, user_prompt_formatted)
        else:
            generated_response = self.call_llm_openai(system_prompt_formatted, user_prompt_formatted)
        
        transcript.append({
            "input_prompt": system_prompt_formatted + user_prompt_formatted,
            "output": generated_response
        })
            
        self.save_transcript_to_json(transcript)
        
        return generated_response

    # Generate a hint for the given problem and program
    # def generate_hint(self, problem_data, buggy_program, testcases):  
        # generated_response = self.call_llm(problem_data, buggy_program)
        # hint = self.extract_hint(generated_response)
        # return hint

    def generate_hint(self, problem_data, buggy_program, testcases):
        repair_agent = repair.Repair(model_name=self.model_name, is_huggingface=self.is_huggingface)

        # Step 1: Generate the best repair
        best_repair = repair_agent.generate_repair(problem_data, buggy_program, testcases)
        
        if not best_repair:
            print("[DEBUG] No valid repair found. Unable to generate hint.")
            return None
        
        # Step 2: Generate hint with explanation using the best repair
        user_prompt_formatted = self.user_prompt_with_repair.format(
            problem_data=problem_data, 
            buggy_program=buggy_program, 
            repair=best_repair
        )
        
        if self.is_huggingface:
            generated_response = self.call_llm_huggingface(self.system_prompt, user_prompt_formatted)
        else:
            generated_response = self.call_llm_openai(self.system_prompt, user_prompt_formatted)
        
        # Extract explanation and hint
        explanation = self.extract_text(generated_response, "[EXPLANATION]", "[/EXPLANATION]")
        hint = self.extract_text(generated_response, "[HINT]", "[/HINT]")

        # Debugging/Logging
        print(f"[DEBUG] Explanation: {explanation}")
        print(f"[DEBUG] Hint: {hint}")

        return hint

    def extract_text(self, text, start_tag, end_tag):
        start_index = text.find(start_tag)
        if start_index == -1:
            return ""
        start_index += len(start_tag)
        end_index = text.find(end_tag, start_index)
        if end_index == -1:
            end_index = len(text)
        return text[start_index:end_index].strip()

