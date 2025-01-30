import openai
import project_part1_prompts as prompts
import project_part1_utils as utils
import project_part1_evaluate as evaluate
import project_part1_utils as utils



import os
import json

import torch

if torch.cuda.is_available():
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

# Class for handling program repair using language models
class Repair:
    # Initialize the Repair class with model details
    def __init__(self, model_name, is_huggingface):
        self.model_name = model_name
        self.system_prompt = prompts.system_message_nus
        self.user_prompt = prompts.user_message_nus_repair_basic
        self.is_huggingface = is_huggingface
        
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

    # Extract fixed code from the generated text
    def extract_fixed_code(self, text):
        start_tag = "[FIXED]"
        end_tag = "[/FIXED]"
        
        start_index = text.find(start_tag)
        if start_index == -1:
            return ""
        
        start_index += len(start_tag)
        end_index = text.find(end_tag, start_index)
        if end_index == -1:
            end_index = len(text)
        
        extracted_text = text[start_index:end_index].strip()
        
        if extracted_text.startswith("```"):
            extracted_text = extracted_text[3:]
        if extracted_text.startswith("python"):
            extracted_text = extracted_text[6:]
        if extracted_text.endswith("```"):
            extracted_text = extracted_text[:-3]
        
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
            ],
            temperature=0.7
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
            use_cache=True,
            # You can use the parameters below to set the temperature for sampling. To learn more about these parameters, you can refer to https://huggingface.co/blog/how-to-generate.
            do_sample=True,
            temperature=0.7
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
    
    # Generate a repair for the given problem and program
    # def generate_repair(self, problem_data, buggy_program, testcases):        
    #     generated_response = self.call_llm(problem_data, buggy_program)
    #     fixed_code = self.extract_fixed_code(generated_response)
        
    #     return fixed_code

    def generate_repair(self, problem_data, buggy_program, testcases):
        # Generate 3 repair candidates
        candidates = []
        print("[DEBUG] Starting repair generation")

        for i in range(3):
            print(f"[DEBUG] Generating candidate {i+1}")
            generated_response = self.call_llm(problem_data, buggy_program)
            fixed_code = self.extract_fixed_code(generated_response)
            
            if fixed_code is not None:  # Ensure only valid candidates are appended
                candidates.append(fixed_code)
            else:
                print(f"[DEBUG] Candidate {i+1} is None and will not be added")

        # If no valid candidates were generated, return None early
        if not candidates:
            print("[DEBUG] No valid candidates generated. Returning None.")
            return None

        print("[DEBUG] Evaluating candidates")

        # Logic to select the best candidate based on minimum edit distance
        best_candidate = None
        best_distance = -1

        # Initialize Compiler and Distance instances from utils
        compiler = utils.Compiler()
        distance_calculator = utils.Distance()

        for candidate in candidates:
            correct = True  # Assume the candidate is correct initially

            for testcase in testcases:
                testcase_input = testcase["input"]
                code_to_check = candidate + f"\nresult = {testcase_input}\nprint(result)\n"
                success, output = compiler.run_program(code_to_check)
                testcase_correct = success and output.strip() == testcase['output'].strip()
                correct = correct and testcase_correct

            if correct:
                edit_distance = distance_calculator.get_edit_distance(candidate, buggy_program)
                print(f"[DEBUG] Candidate passed all test cases with distance {edit_distance}")

                # Update the best candidate if it's closer
                if best_distance == -1 or edit_distance < best_distance:
                    best_candidate = candidate
                    best_distance = edit_distance
            else:
                print(f"[DEBUG] Candidate failed one or more test cases")

        # If no correct candidate is found, return None
        if best_candidate is None:
            print("[DEBUG] No correct candidates passed all test cases so returning first candidate")
            return candidates[0]

        print("[DEBUG] Best candidate selected")
        return best_candidate
