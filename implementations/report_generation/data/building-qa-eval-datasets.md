# Building Ground-Truth QA Datasets for Agent Evaluation

This guide explains the methodology used to produce the [OnlineRetailReportEval.json](OnlineRetailReportEval.json)
ground truth dataset. The goal is to provide a guide to the techiniques used here in order to
reproduce it for your own datasets.

---

## Core Idea

An evaluation dataset is a list of `(question, ground_truth)` pairs where ground truth is
a verifiable correct response to the question. An evaluator, like an LLM judge, checks
whether the agent's response contains those facts, and from it other metrics like
precision, recall, F1 score and others can be measured.

In the `OnlineRetailReportEval.json` dataset, the question is contained in the
`input` field and the ground truth is cointained in the `expected_output` field.

---

## Dataset Structure

Each example needs four fields:

| Field | Type | Description |
|---|---|---|
| `id` | `int` | Unique identifier |
| `input` | `str` | The query given to the agent |
| `expected_output` | `dict[str, Any]` | The ground truth (see below) |

The ground truth is split into two sections, one for each evaluator:
- `final_report`: Contains the data that is expected to be sent to the function that
generates the report. Used by the
[final result evaluator](../../../aieng-eval-agents/aieng/agent_evals/report_generation/evaluation/offline.py#L193)
to check if the final result produced by the agent is correct.
- `trajectory`: Contains the tool calls the agent is expected to make in order to
produce the final result. Used by the
[trajectory evaluator](../../../aieng-eval-agents/aieng/agent_evals/report_generation/evaluation/offline.py#L258)
to check if the agent is following the required steps in order to produce a result.
The trajectory itself has two parts:
    - `actions`: the list of tool calls the agents are expected to perform.
    - `description`: the description of the parameters that are supposed to be
    sent to the actions. Descriptions are needed (as opposed to the actual parameter values)
    because the parameters to those actions can be slightly different between runs while
    still being correct.

---

## How the Evaluators Work

The evaluators (which are both LLM-as-judges) receive four pieces of information in order
to make their evaluation decision:
- The `input` sent to the agent.
- The `expected_output` (or ground truth)
- The actual agent output
- A set of instructions on how to evaluate the actual output against the
expected output and input.

To check those instructions in details, please see the
[prompts.py](../../../aieng-eval-agents/aieng/agent_evals/report_generation/prompts.py) file.

In the **Final Result Evaluator**, the instructions tell the actual agent output should
match the expected output with a few special criteria, like having a certain floating point
tolerance, disregarding the column names, disregarding the order (unless explicitly
specified in the input), etc.

In the **Trajectory Evaluator**, the actual agent output contains the list of tools called
by the agent and their parameters. The evaluator is instructed to check if the list of
expected tool calls match the list of instructions, and if the parameters passed to
those tools match the description provided in the ground truth. The evaluator is also
provided with evaluation criteria such as: disregard if the agent makes mistakes
and performs additional steps (as long as the steps expected are performed at some
point), to always check if the final response is produced, etc.

The evaluators will produce a 1 or 0 score telling if the actual output matches the expected
output within the given criteria, as well as an explanation of why did they give that score.

---

## Tips for Writing Good Ground Truth

### 1. Use questions with verifiable, known answers

To check the agent is outputing proper results, find questions that you know the answers
for, or questions that you can produce and verify the answers for, so you can compare
them with the agent answers.

Remember the agents can produce slightly different results that are still considered
correct for the same queries, so you will have to incorporate those fluctuations into
the instructions that is given to the evaluators.

### 2. Add questions with various levels of difficulty

This will tell you in what level os difficulty are the agents failing more often,
or are more likely to fail.

### 3. Collect a good agent run to use as ground truth

You can also run the queries through the agents a few times and choose goods responses
that you want to use as ground truth.

You can also examine those runs and check what are the patterns the agents are
exhibiting on the bad responses and incorporate those as instructions to the
evaluators in order for them to evaluate bad agent runs responses correctly.

### 4. Tell the evaluators what do you consider important

On the instructions to the evaluators, it is crucial to tell them what do you
consider important and not important when evaluating a response.

Being specific is better than being vague or ommiting information. You will
want to avoid judgment calls by the evaluators in order to have more stable
and trusting results.

It is important to specify in the instructions what the evaluators
should be looking for in both positive and negative responses.

---

## How to Construct Your Dataset

### Option A: Manual curation (highest quality)
1. Write questions that reflect real user queries in your domain.
2. Look up the answer from a trusted source, or produce a response from an agent run and
manually verify its correctness.
3. Format the answer into the dataset format.

### Option B: LLM-assisted generation + human review
Use an LLM to generate `(question, answer)` pairs from a reference document, then have a
human verify each answer against the source. Prompt pattern:

```
Given this document: <doc>
Generate 5 factual questions and their concise ground truth answers.
For each answer, decompose it into atomic facts if there are multiple.
Format: {"question": "...", "answer": "...", "answer_type": "Single Answer" | "Set Answer"}
```
