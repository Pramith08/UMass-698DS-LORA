"""Canonical judge prompt template.

This is the *exact* prompt Geetanjali used with GPT-OSS 140B to label the 5k training set.
Both training and evaluation must format inputs through `format_prompt` so the inputs match
the labels.
"""

JUDGE_PROMPT_TEMPLATE = """You are an expert cybersecurity answer evaluator.

You will be given a cybersecurity question and three candidate answers.

Your task is to rank the three answers from best to worst based on:

Technical correctness

Completeness

Relevance to the question

Clarity and precision

Lack of hallucination or misleading information

Keywords match

Assign each rubric a score and then sum all 6 rubric to find total score for a response and then rank the responses.

Important:

R1, R2, and R3 are all candidate answers.

Do not assume the reference answer is always best.

Judge only based on answer quality.

Output ONLY the ranking.

The output format must be exactly like one of these:

R1>R2>R3

R2>R1>R3

R3>R2>R1

R2>R3>R1

R3>R1>R2

R1>R3>R2

Question:
{question}

R1:
{r1}

R2:
{r2}

R3:
{r3}

Ranking:"""


VALID_RANKINGS = frozenset({
    "R1>R2>R3",
    "R1>R3>R2",
    "R2>R1>R3",
    "R2>R3>R1",
    "R3>R1>R2",
    "R3>R2>R1",
})


def format_prompt(question: str, r1: str, r2: str, r3: str) -> str:
    """Format the judge prompt with three candidate answers in slots R1/R2/R3."""
    return JUDGE_PROMPT_TEMPLATE.format(question=question, r1=r1, r2=r2, r3=r3)
