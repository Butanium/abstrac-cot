from DecompositionFaithfulnessPaper.prompts.factored_decomposition_recomposition_few_shot_prompt import (
    FD_RECOMPOSITION_FEW_SHOT as recomp_promt,
)
from DecompositionFaithfulnessPaper.prompts.factored_decomposition_decomposition_few_shot_prompt import (
    FD_DECOMPOSITION_FEW_SHOT as decomp_promt,
)
import json


def few_shot_to_messages(few_shot_prompt: str) -> list[dict[str, str]]:
    """
    Convert a few-shot prompt to a list of messages. The format of the few-shot prompt is as follows:
    Human: ...
    Assistant: ...
    ...

    Args:
        few_shot_prompt (str): a few-shot prompt

    Returns:
        list[dict[str, str]]: a list of messages in openai api format
    """
    chat = []
    for line in few_shot_prompt.replace("Human:\n", "Human: ").split("Human: ")[1:]:
        human, assistant = line.replace("Assistant:\n", "Assistant: ").split(
            "Assistant: "
        )
        chat.append({"role": "user", "content": human.rstrip()})
        chat.append({"role": "assistant", "content": assistant.rstrip()})
    return chat


if __name__ == "__main__":
    decomp_promt = few_shot_to_messages(decomp_promt)
    recomp_promt = few_shot_to_messages(recomp_promt)
    with open("data/fd_few_shot_chat.json", "w") as f:
        json.dump({"decomp_promt": decomp_promt, "recomp_promt": recomp_promt}, f, indent=4)
