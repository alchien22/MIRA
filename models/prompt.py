FACTUALITY_PROMPT = """
You are a medical expert.

You will be given a question, relevant retrieved information, and a generated answer.
Your task is to evaluate how factually correct the answer is with respect to the question and context.

Step 1: Write 2-4 sentences of reasoning explaining the factual strengths and weaknesses of the answer.

Step 2: Provide a factuality score from 1 to 100 based on the following scale:
- 100: The answer is completely correct.
- 50: The answer is partially correct, with important omissions or errors.
- 1: The answer is completely incorrect.

Your response must follow this format:
Reasoning: <your reasoning here>
Factuality Score: <integer from 1 to 100>

Now here is the input:
Question: {question}
Context: {retrieved_info}
Answer: {generated_answer}

Begin your answer below:
"""


CONSISTENCY_PROMPT = """
You are a medical expert.

You will be given some retrieved context and a generated answer.
Your task is to evaluate how consistent the answer is with the context.

Step 1: Write 2-4 sentences explaining whether the answer is well-supported by the context. Focus on whether each claim in the answer is grounded in the provided information.

Step 2: Provide a consistency score from 1 to 100 based on the following scale:
- 100: All claims in the answer are directly supported by the context.
- 50: Some claims are supported by the context, but others are missing or contradictory.
- 0: No claims in the answer are supported by the context.

Your response must follow this format:
Reasoning: <your reasoning here>
Consistency Score: <integer from 1 to 100>

Now here is the input:
Context: {retrieved_info}
Answer: {generated_answer}

Begin your answer below:
"""


