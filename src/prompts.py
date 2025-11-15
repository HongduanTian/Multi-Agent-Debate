__all__ = ["COT_PROMPT", "DEBATE_PROMPT", "PRUNE_PROMPT"]

COT_PROMPT = """
Please solve the problem step by step.
"""

DEBATE_PROMPT = """
These are the potential solutions to the problem:
{context}
Use the potential solutions as additional information for the following question.

{question}
Please think step by step and solve the problem.
"""

PRUNE_PROMPT = """
Evaluate the given solutions based on the question. ** Your reponse MUST end with the following format: <label>YES</label> or <label>NO</label> or <label>NOT SURE</label>. ** Return YES if the solution is completely correct, NO if any part of the solution is incorrect, and NOT SURE if you are unsure.\n
{question}
{solution}
"""

# Some other prompts can be proposed here.