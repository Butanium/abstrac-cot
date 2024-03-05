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
    chat = [{"role": "system", "content": "You are a helpful assistant."}]
    for line in few_shot_prompt.replace("Human:\n", "Human: ").split("Human: ")[1:]:
        human, assistant = line.replace("Assistant:\n", "Assistant: ").split(
            "Assistant: "
        )
        chat.append({"role": "user", "content": human.rstrip()})
        chat.append({"role": "assistant", "content": assistant.rstrip()})
    return chat


if __name__ == "__main__":
    from DecompositionFaithfulnessPaper.prompts.factored_decomposition_recomposition_few_shot_prompt import (
        FD_RECOMPOSITION_FEW_SHOT as recomp_prompt,
    )
    from DecompositionFaithfulnessPaper.prompts.factored_decomposition_decomposition_few_shot_prompt import (
        FD_DECOMPOSITION_FEW_SHOT as decomp_prompt,
    )
    from DecompositionFaithfulnessPaper.prompts.few_shot_prompt import (
        FEW_SHOT as few_shot_prompt,
    )
    from DecompositionFaithfulnessPaper.prompts.chain_of_thought_few_shot_prompt import (
        COT_FEW_SHOT as cot_prompt,
    )
    from DecompositionFaithfulnessPaper.prompts.chain_of_thought_decomposition_few_shot_prompt import (
        COTD_FEW_SHOT as cotd_prompt,
    )
    import json

    decomp_prompt = few_shot_to_messages(decomp_prompt)
    recomp_prompt = few_shot_to_messages(recomp_prompt)
    with open("data/fd_few_shot_chat.json", "w") as f:
        json.dump(
            {"decomp_prompt": decomp_prompt, "recomp_prompt": recomp_prompt},
            f,
            indent=4,
        )
    with open("data/few_shot_chat.json", "w") as f:
        json.dump(few_shot_to_messages(few_shot_prompt), f, indent=4)
    with open("data/cot_few_shot_chat.json", "w") as f:
        json.dump(few_shot_to_messages(cot_prompt), f, indent=4)
    with open("data/cotd_few_shot_chat.json", "w") as f:
        json.dump(few_shot_to_messages(cotd_prompt), f, indent=4)
