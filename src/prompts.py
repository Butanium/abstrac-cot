DECOMPOSITION_INSTRUCTIONS = """I'm going to ask you a question. I want you to decompose it into a series of subquestions which should be enough to answer the original question. Ideally, each subquestion should be much easier to answer than the original question. Ensure that each subquestion is self-contained, containing all necessary information. This is crucial because I'll be presenting each subquestion independently to someone else, excluding the original problem. They should be able to solve the subquestion based solely on the provided information and context. This is really important. For example, avoid ambiguous references such as \"the teacher\" or \"the above elements\" without providing sufficient context. If needed you may include the entire passage or situation being referenced.

Quote passages or text in their entirety to maintain clarity in the decomposition. Use <sub_q> tags for each subquestion, including corresponding numbers, e.g. <sub_q_1></sub_q_1>.

You are allowed and encouraged to ask subquestions that depend on previous answers. To do this, use placeholders to refer to previous answers. For example, if you need to refer to the answer to the first subquestion, use {sub_a_1} as a placeholder.

Ensure not to decompose more than necessary and to avoid subquestions irrelevant to the original question. You'll be evaluated on the simplicity, conciseness, and correctness of your decompositions.

After your initial decomposition, I'll give you the answer to the subquestion which doesn't depend on any previous answers - you should then output the subquestions that can now be answered, incorporating the answer to the first subquestion filled in and rephrased appropriately if necessary. Eventually, you'll have answers to all the subquestions, at which point you should output the sequence <FIN></FIN>.
"""

OLD_INSTRUCTION = """I'm going to ask you a question. I want you to decompose it into a series of subquestions. Each subquestion should be self-contained with all the information necessary to solve it. This is because I'll be showing someone else the subquestion without showing them the original problem and they need be able to solve the subquestion with only the information and context of the subquestion provided. This is really important - for example, you should never say things like \"the teacher\" or \"that player\" without giving more context as to who the teacher / that player is and possibly the entire passage or situation that is being referenced. You should quote passages or text from the questions in their entirety to accomplish this task in the right way.\n\nMake sure not to decompose more than necessary or have any trivial subquestions - you'll be evaluated on the simplicity, conciseness, and correctness of your decompositions as well as your final answer. Please put each subquestion in <sub_q> tags, but include the numbers corresponding to each in the tag, eg <sub_q_1></sub_q_1>. After your initial decomposition, I'll give you the answer to the first subquestion you asked - you should then output the remaining subquestions you need answered, with the answer to the first subquestion filled in and rephrased appropriately if necessary. Eventually you'll have answers to all the subquestions, at which point you should output the sequence <FIN></FIN>.\n\nLet's go through some examples together. Do you understand the instructions?"""

INSTRUCTION_REMINDER = f"""Great progress so far! Before we move on to the next question, let's quickly review your instructions.

{DECOMPOSITION_INSTRUCTIONS}


Now, let's proceed to the next question: """

# This prompt is given when the model is asked to answer a single subquestion without any context
SINGLE_QUESTION_PROMPT = """Please answer the question and keep your response **concise**. Do not repeat the question nor the instructions, do not talk. Just answer the question. Here's the question:\n"""

RECOMP_ANSWER_PROMPT = "Based on the above, the correct answer is choice ("

FEW_SHOT_ANSWER_PROMPT = "The correct answer is choice ("

RECOMPOSITION_INSTRUCTIONS = """I'm going to give you a question, answer choices for that question, and a series of subquestions and answers to those subquestions that should help you pick the correct answer choice. You should make sure to make your final answer based on the subquestions and their answer - these have been carefully selected for their correctness and accuracy, so you should defer to them on all matters."""


## Cot prompts
COT_START_PROMPT = "Let's think step by step:\n\n"
COT_QUESTION_PROMPT = (
    "Based on the above, what is the single, most likely answer choice?"
)

COT_ANSWER_PROMPT = "The correct answer is choice ("

COTD_QUESTION_PROMPT = "Based on the above, what's the correct answer to the question?"

COTD_ANSWER_PROMPT = "The correct answer to the question is choice ("

COTD_START_PROMPT = "<sub_q>"
