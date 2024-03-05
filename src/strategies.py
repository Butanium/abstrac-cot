from .model_utils import (
    user_message,
    assistant_message,
    stop_token,
    query_model,
    question_message,
)
from .constants import SPLIT_STRING
from .prompts import (
    COT_START_PROMPT,
    COT_QUESTION_PROMPT,
    COT_ANSWER_PROMPT,
    COTD_START_PROMPT,
    COTD_QUESTION_PROMPT,
    COTD_ANSWER_PROMPT,
    FEW_SHOT_ANSWER_PROMPT,
)
from .factored_decomposition import decompose, recompose


def few_shot(model, tokenizer, question, choices, log_path, few_shot_chat):
    """
    Runs the few-shot method on a single question and set of choices.

    Args:
        model: The model to use.
        tokenizer: The tokenizer to use.
        question: The question to answer.
        choices: The list of choices for the question.
        log_path: The path to log the results.
        few_shot_chat: The chat prompt to use for the few-shot method.

    Returns:
        answer: The model's answer to the question.
    """
    prompt_chat = few_shot_chat.copy()
    prompt_chat.append(question_message(question, choices))
    prompt_chat.append(assistant_message(f"{FEW_SHOT_ANSWER_PROMPT}{SPLIT_STRING}"))
    prompt = tokenizer.apply_chat_template(prompt_chat, tokenize=False).split(
        SPLIT_STRING
    )[0]
    answer = query_model(
        prompt,
        model,
        stop_token(model),
        log_path=log_path,
        max_tokens=1,
        temperature=0,
    )
    return answer


def chain_of_thought(model, tokenizer, question, choices, log_path, cot_chat):
    """
    Runs the chain of thought method on a single question and set of choices.

    Args:
        model: The model to use.
        tokenizer: The tokenizer to use.
        question: The question to answer.
        choices: The list of choices for the question.
        log_path: The path to log the results.
        cot_chat: The chat prompt to use for the chain of thought method.

    Returns:
        answer: The model's answer to the question.
    """
    prompt_chat = cot_chat.copy()
    prompt_chat.append(question_message(question, choices))
    prompt_chat.append(assistant_message(COT_START_PROMPT + SPLIT_STRING))
    prompt = tokenizer.apply_chat_template(prompt_chat, tokenize=False).split(
        SPLIT_STRING
    )[0]
    cot = COT_START_PROMPT + query_model(
        prompt, model, stop_token(model), log_path=log_path, max_tokens=2048
    )
    prompt_chat[-1]["content"] = cot
    prompt_chat.append(user_message(COT_QUESTION_PROMPT))
    prompt_chat.append(assistant_message(COT_ANSWER_PROMPT + SPLIT_STRING))
    prompt = tokenizer.apply_chat_template(prompt_chat, tokenize=False).split(
        SPLIT_STRING
    )[0]
    answer = query_model(
        prompt, model, stop_token(model), log_path=log_path, max_tokens=1, temperature=0
    )
    return answer


def chain_of_thought_decomposition(
    model, tokenizer, question, choices, log_path, cotd_chat
):
    """
    Runs the chain of thought decomposition method on a single question and set of choices.

    Args:
        model: The model to use.
        tokenizer: The tokenizer to use.
        question: The question to answer.
        choices: The list of choices for the question.
        log_path: The path to log the results.
        cotd_chat: The chat prompt to use for the chain of thought decomposition method.

    Returns:
        answer: The model's answer to the question.
    """
    prompt_chat = cotd_chat.copy()
    prompt_chat.append(question_message(question, choices))
    prompt_chat.append(assistant_message(COTD_START_PROMPT + SPLIT_STRING))
    prompt = tokenizer.apply_chat_template(prompt_chat, tokenize=False).split(
        SPLIT_STRING
    )[0]
    cotd = COTD_START_PROMPT + query_model(
        prompt, model, stop_token(model), log_path=log_path, max_tokens=2048
    )
    prompt_chat[-1]["content"] = cotd
    prompt_chat.append(user_message(COTD_QUESTION_PROMPT))
    prompt_chat.append(assistant_message(COTD_ANSWER_PROMPT + SPLIT_STRING))
    prompt = tokenizer.apply_chat_template(prompt_chat, tokenize=False).split(
        SPLIT_STRING
    )[0]
    answer = query_model(
        prompt, model, stop_token(model), log_path=log_path, max_tokens=1, temperature=0
    )
    return answer


def factored_decomposition(
    model,
    tokenizer,
    question,
    choices,
    log_path,
    decomp_chat,
    recomp_chat,
    skip_system=True,
    skip_user_instruction=False,
):
    """
    Runs the factored decomposition method on a single question and set of choices.

    Args:
        model: The model to use.
        tokenizer: The tokenizer to use.
        question: The question to answer.
        choices: The list of choices for the question.
        log_path: The path to log the results.
        decomp_chat: The chat prompt to use for the decomposition step.
        recomp_chat: The chat prompt to use for the recomposition step.
        skip_system: Whether to skip the system message in the chat.
        skip_user_instruction: Whether to skip the user instruction in the chat.

    Returns:
        answer: The model's answer to the question.
    """
    decomp_chat = decomp_chat.copy()
    if skip_system and skip_user_instruction:
        decomp_chat = decomp_chat[3:]
        recomp_chat = recomp_chat[3:]
    elif skip_user_instruction:
        decomp_chat = [decomp_chat[0]] + decomp_chat[3:]
        recomp_chat = [recomp_chat[0]] + recomp_chat[3:]
    elif skip_system:
        decomp_chat = decomp_chat[1:]
        recomp_chat = recomp_chat[1:]
    qas = decompose(model, tokenizer, question, choices, decomp_chat, log_path)
    answer = recompose(model, tokenizer, question, choices, qas, recomp_chat, log_path)
    return answer
