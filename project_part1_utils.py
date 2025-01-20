import os
import subprocess
import tempfile
from Levenshtein import distance as levenshtein_distance
from pygments.lexers.python import PythonLexer
from typing import List
import pygments
import re

# Class for compiling and running Python programs
# You do not need to change this class
class Compiler:
    # A helper function to extract the error message from the error string
    def extract_error_message(self, error_string):
        lines = error_string.strip().split('\n')
        final_error_line = next((line for line in reversed(lines) if 'Error' in line), None)
        return final_error_line if final_error_line else 'Unknown Error'

    # Run a Python program and return the output. Error messages are returned if the program fails to run.
    def run_program(self, program):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(program.encode())
            temp_file_path = temp_file.name
        try:
            result = subprocess.run(['python', temp_file_path], capture_output=True, text=True, check=True, timeout=4)
            
            result_str = result.stdout.strip()
            
            # Convert the format to properly match the expected output for boolean values
            if result_str == "false":
                result_str = "False"
            elif result_str == "true":
                result_str = "True"
                
            return True, result_str
        
        except subprocess.CalledProcessError as e:
            error_message = self.extract_error_message(e.stderr.strip())
            return False, error_message
        except subprocess.TimeoutExpired:
            return False, "Execution timed out"
        except FileNotFoundError:
            return False, "Python interpreter not found"
        except PermissionError:
            return False, "Permission denied"
        finally:
            os.remove(temp_file_path)

    # Run a Python program with multiple test cases and return the results. Returns a boolean indicating if all test cases passed and the output for each test case.
    def run_program_with_testcases(self, program, testcases):
        results = []
        all_correct = True
        for testcase in testcases:
            testcase_input = testcase["input"]
            expected_output = testcase["output"]
            code_to_check = program + f"\nresult = {testcase_input}\nprint(result)\n"
            success, output = self.run_program(code_to_check)
            testcase_correct = success and output.strip() == expected_output.strip()
            results.append((testcase_correct, output))
            all_correct = all_correct and testcase_correct
        return all_correct, results

# Class for calculating edit distances between Python programs
# You do not need to change this class
class Distance:
    def __init__(self):
        self.lexer = PythonLexer()
    
    # Lex the program into tokens using the lexer
    def lex_program(self, program):
        return list(self.lexer.get_tokens(program))

    # Convert the program into a list of essential tokens, removing unnecessary spaces and comments
    def program_to_essential_tokens(self, program: str, strip_chars="\n\r\t\f ") -> List[str]:
        simplified_program_tokens = []

        if isinstance(program, float) or program is None or len(program) == 0:
            return [""]

        lines = program.split("\n")
        meaningful_lines = [line for line in lines if line.strip(strip_chars) != ""]
        program_without_blank_lines = "\n".join(meaningful_lines)

        lex_output = self.lex_program(program_without_blank_lines)
        is_start_of_line = True
        for component in lex_output:
            token_type, token_value = component
            if token_type == pygments.token.Whitespace and token_value == "\n":
                is_start_of_line = True
                if len(simplified_program_tokens) == 0 or simplified_program_tokens[-1] != "\n":
                    simplified_program_tokens.append(token_value)
            elif not is_start_of_line and token_type == pygments.token.Text and re.match(r"^\s+$", token_value):
                pass  # drop all unnecessary spaces (i.e. all space tokens after the first non-space token in every line)
            elif token_type in pygments.token.Comment.subtypes:
                pass  # drop all comments
            else:
                simplified_program_tokens.append(token_value)
                is_start_of_line = False

        while len(simplified_program_tokens) > 0 and simplified_program_tokens[-1].strip(strip_chars) == "":
            del simplified_program_tokens[-1]

        return simplified_program_tokens

    # Calculate the edit distance between two sequences of tokens
    def token_edit_distance(self, seq1: List[str], seq2: List[str]) -> int:
        return levenshtein_distance(seq1, seq2)

    # Get the edit distance between two Python programs which are represented as strings
    def get_edit_distance(self, source1: str, source2: str) -> int:
        return self.token_edit_distance(self.program_to_essential_tokens(source1), self.program_to_essential_tokens(source2))