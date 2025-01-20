system_message_nus = "You are an AI tutor. You have to help a student learning programming. The program uses Python. You have to strictly follow the format for the final output as instructed below."

user_message_nus_repair_basic = """
Following is the setup of a problem in Python. It contains the description and a sample testcase.

[Problem Starts]
{problem_data}
[Problem Ends]

Following is the student's buggy code for this problem:

[Buggy Code Starts]
{buggy_program}
[Buggy Code Ends]


Fix the buggy code. Output your entire fixed code between [FIXED] and [/FIXED].
"""

user_message_nus_hint_basic = """
Following is the setup of a problem in Python. It contains the description and a sample testcase.

[Problem Starts]
{problem_data}
[Problem Ends]

Following is the student's buggy code for this problem:

[Buggy Code Starts]
{buggy_program}
[Buggy Code Ends]

Provide a concise single-sentence hint to the student about one bug in the student's buggy code. Output your hint between [HINT] and [/HINT].
"""

