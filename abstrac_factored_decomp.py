from openai import OpenAI
from DecompositionFaithfulnessPaper.prompts.factored_decomposition_recomposition_few_shot_prompt import FD_RECOMPOSITION_FEW_SHOT
import os

client = OpenAI(api_key=os.environ["CHARBEL_OPEN_AI_KEY"])
# chat_completion = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": FD_DECOMPOSITION_FEW_SHOT},
#     ])

