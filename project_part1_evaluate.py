import os
import json
from datetime import datetime

import project_part1_utils as utils
import project_part1_repair as repair
import project_part1_hint as hint

# Class for evaluating program repairs and generating hints
# You do not need to change this class
class ProgramEvaluation:
    def __init__(self, repair_agent, hint_agent, compiler, distance):
        self.repair_agent = repair_agent
        self.hint_agent = hint_agent
        self.compiler = compiler
        self.distance = distance

    # Evaluates a repair by generating a repair for a buggy program and testing it against test cases
    def get_and_evaluate_repair(self, problem_data, buggy_program, testcases):        
        extracted_repair = self.repair_agent.generate_repair(problem_data, buggy_program, testcases)
        
        distance = -1 # Default value
        correct = True
                
        for testcase in testcases:
            testcase_input = testcase["input"]
            code_to_check = extracted_repair + f"\nresult = {testcase_input}\nprint(result)\n"
            success, output = self.compiler.run_program(code_to_check)
            testcase_correct = success and output.strip() == testcase['output'].strip()
            correct = correct and testcase_correct
            
        if correct:
            distance = self.distance.get_edit_distance(extracted_repair, buggy_program)
            
        
        repair_results = {
            "correct": correct,
            "repair": extracted_repair,
            "distance": distance
        }
        
        return repair_results
    
    # Generates a hint for a buggy program
    def get_hint(self, problem_data, buggy_program, testcases):               
        extracted_hint = self.hint_agent.generate_hint(problem_data, buggy_program, testcases)        
        return extracted_hint
    
# Class for evaluating datasets of programs
# You do not need to change this class
class DatasetEvaluation:
    def __init__(self, programEvaluator, mode):
        self.programEvaluator = programEvaluator
        self.mode = mode

    # Fetches test cases for a given problem ID
    def fetch_testcases(self, problem_id):
        content_directory = self.get_content_directory()
        problem_file = self.find_problem_file(content_directory, problem_id)
        return self.load_testcases(problem_file)

    # Fetches problem data including test cases and constructs a problem description
    def fetch_problem_data(self, problem_id):
        content_directory = self.get_content_directory()
        problem_file = self.find_problem_file(content_directory, problem_id)
        testcases = self.load_testcases(problem_file)
        return self.construct_problem_data(problem_file, testcases, problem_id)

    # Returns the directory path where problem files are stored
    def get_content_directory(self):
        return os.path.join("./project_part1_datasets/problems/")

    # Finds the problem file for a given problem ID in the specified directory
    def find_problem_file(self, content_directory, problem_id):
        problem_file = next((f for f in os.listdir(content_directory) if f.startswith(f"{problem_id}_")), None)
        if not problem_file:
            raise FileNotFoundError(f"No file found for problem_id {problem_id}")
        return os.path.join(content_directory, problem_file)

    # Loads test cases from a problem file
    def load_testcases(self, problem_file):
        try:
            with open(problem_file, 'r') as file:
                data = json.load(file)
                return data["tests"]
        except FileNotFoundError:
            raise
        except json.JSONDecodeError:
            raise

    # Constructs problem data by combining the title, description, and sample test cases
    def construct_problem_data(self, problem_file, testcases, problem_id):
        with open(problem_file, 'r') as file:
            data = json.load(file)
            title = data["title"]
            problem_description = data["description"]
            if data["additional_description"]:
                problem_description += "\n" + data["additional_description"]
            sample_testcase = self.create_sample_testcase(testcases, problem_id)
            problem_data = f"{title} - {problem_description}\n\n\nSample Testcase - \n{sample_testcase}\n"
            return problem_data

    # Creates a sample test case string for a given problem ID
    def create_sample_testcase(self, testcases, problem_id):
        sample_testcase = f"Input: \n{testcases[0]['input']}\nExpected Output: \n{testcases[0]['output']}\n"
        if problem_id == "2":
            sample_testcase += f"\nInput: \n{testcases[1]['input']}\nExpected Output: \n{testcases[1]['output']}\n"
            sample_testcase += f"\nInput: \n{testcases[3]['input']}\nExpected Output: \n{testcases[3]['output']}\n"
        return sample_testcase

    # Evaluates all programs in a specified folder for a given problem ID
    def evaluate_programs_in_folder(self, folder_path, problem_id, problem_data, evaluation_results):
        total_programs = 0
        correct_programs = 0

        for program_folder in os.listdir(folder_path):
            if "prog_" in program_folder:
                total_programs += 1
                buggy_code_path = os.path.join(folder_path, program_folder, "buggy.py")
                with open(buggy_code_path, "r") as file:
                    buggy_code = file.read()
                    testcases = self.fetch_testcases(problem_id)
                    
                    program_results = {}
                    
                    if self.mode == "repair":
                        repair_results = self.programEvaluator.get_and_evaluate_repair(problem_data, buggy_code, testcases)
                        if repair_results["correct"]:
                            correct_programs += 1
                        
                        program_results["repair"] = repair_results
                        
                    elif self.mode == "hint":
                        hint = self.programEvaluator.get_hint(problem_data, buggy_code, testcases)
                        program_results["hint"] = hint
                        
                    evaluation_results[f"problem_{problem_id}_{program_folder}"] = program_results
                    
    # Saves the evaluation results to a specified directory with a timestamp
    def save_evaluation_results(self, evaluation_results, save_directory, timestamp):
        os.makedirs(save_directory, exist_ok=True)
        
        try:
            overall_results = {}
            
            if self.mode == "repair":
                overall_results["repair_model"] = self.programEvaluator.repair_agent.model_name
                total_programs = len(evaluation_results)
                correct_programs = sum([1 for program in evaluation_results.values() if program["repair"]["correct"]])
                repair_rate = correct_programs / total_programs
                
                correct_distances = [program["repair"]["distance"] for program in evaluation_results.values() if program["repair"]["correct"]]
                
                if len(correct_distances) > 0:
                    average_distance = sum(correct_distances) / len(correct_distances)
                else:
                    average_distance = -1
                
                overall_results["RPass"] = repair_rate*100
                overall_results["REdit"] = average_distance
            elif self.mode == "hint":
                overall_results["hint_model"] = self.programEvaluator.hint_agent.model_name
            
            overall_results["timestamp"] = timestamp
            
            sorted_evaluation_results = dict(sorted(
                evaluation_results.items(),
                key=lambda item: tuple(map(int, item[0].split('_')[1::2]))
            ))
            
            with open(os.path.join(save_directory, f"results_{timestamp}.json"), 'w') as json_file:
                json.dump({
                    "summary": overall_results,
                    "results": sorted_evaluation_results
                }, json_file, indent=4)
                
        except IOError:
            raise
        
    # Evaluates the dataset by iterating over folders and saving results
    def evaluate_dataset(self, save_path=None):
        evaluation_data_path = os.path.join(f"./project_part1_datasets/evaluation_data/")
        evaluation_results = {}
        timestamp = datetime.now().isoformat()

        for folder in os.listdir(evaluation_data_path):
            folder_path = os.path.join(evaluation_data_path, folder)
            if os.path.isdir(folder_path):
                problem_id = folder.split("_")[0]
                problem_data = self.fetch_problem_data(problem_id)
                self.evaluate_programs_in_folder(folder_path, problem_id, problem_data, evaluation_results)

            if save_path:
                self.save_evaluation_results(evaluation_results, save_path, timestamp)

# Main function to execute the program evaluation
# You can change the model_selected variable to select the model and the mode variable to switch between repair and hint modes
if __name__ == "__main__":    
    model_dict = {
        "gpt-4o-mini": "gpt-4o-mini",
        "phi-3-mini": "unsloth/Phi-3-mini-4k-instruct"
    }
    
    model_selected = model_dict["gpt-4o-mini"] # Change this to select the model
    
    compiler = utils.Compiler()
    distance = utils.Distance()
        
    repair_agent = repair.Repair(model_name=model_selected, is_huggingface=(model_selected == "unsloth/Phi-3-mini-4k-instruct"))
    hint_agent = hint.Hint(model_name=model_selected, is_huggingface=(model_selected == "unsloth/Phi-3-mini-4k-instruct"))
    programEvaluator = ProgramEvaluation(repair_agent, hint_agent, compiler, distance)
    
    mode = "repair"  # Can take values "repair" or "hint"
    
    datasetEvaluator = DatasetEvaluation(programEvaluator, mode=mode)
    datasetEvaluator.evaluate_dataset(save_path="./ /")