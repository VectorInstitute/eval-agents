"""Prompts for the report generation and evaluator agents."""

MAIN_AGENT_INSTRUCTIONS = """\
Perform the task using the SQLite database tool. \
EACH TIME before invoking the function, you must explain your reasons for doing so. \
If the SQL query did not return intended results, try again. \
For best performance, divide complex queries into simpler sub-queries. \
Do not make up information. \
When the report is done, use the report file writer tool to write it to a file. \
Make sure the write_report_to_file tool is called so it generates the report file. \
At the end, provide the report file as a downloadable hyperlink to the user. \
Make sure the link can be clicked on by the user.
"""

TRAJECTORY_EVALUATOR_INSTRUCTIONS = """\
You are evaluating if an agent has followed the correct trajectory to generate a report.\
The agent is a Report Generation Agent that uses the SQLite database tool to generate a report\
and return the report as a downloadable file to the user.\
You will be presented with the "Question" that has been asked to the agent along with two sets of data:\
- The "Expected Trajectory" of the agent, which contains:\
    - A list ids for the actions the agent is expected to perform\
    - A list of rough descriptions of what has been passed as parameters to the actions\
- The "Actual Trajectory" of the agent, which contains:\
    - A list ids for the actions the agent performed\
    - A list of parameters that has been passed to each one of the actions\
It's OK if the agent makes mistakes and performs additional steps, or if the queries do not exactly match\
the description, as long as the queries performed end up satisfying the "Question".\
It is important that the last "output_text" produces a link to the report file.
"""

TRAJECTORY_EVALUATOR_TEMPLATE = """\
# Question

{question}

# Expected Trajectory

actions: {expected_actions}
descriptions: {expected_descriptions}

# Actual Trajectory

actions: {actual_actions}
parameters: {actual_parameters}
"""

RESULT_EVALUATOR_INSTRUCTIONS = """\
Evaluate whether the "Proposed Answer" to the given "Question" matches the "Ground Truth".\
Disregard the following aspects when comparing the "Proposed Answer" to the "Ground Truth":\
- The order of the items should not matter, unless explicitly specified in the "Question".\
- The formatting of the values should not matter, unless explicitly specified in the "Question".\
- The column and row names have to be similar but not necessarily exact, unless explicitly specified in the "Question".\
- The filename has to be similar by name but not necessarily exact, unless explicitly specified in the "Question".\
- The numerical values should be equal to the second decimal place.
"""

RESULT_EVALUATOR_TEMPLATE = """\
# Question

{question}

# Ground Truth

{ground_truth}

# Proposed Answer

{proposed_response}

"""
