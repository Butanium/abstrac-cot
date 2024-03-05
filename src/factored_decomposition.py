from .model_utils import (
    user_message,
    assistant_message,
    question_message,
    stop_token,
    query_model,
)
import json
import regex
from .constants import QUESTION_LABELS, SPLIT_STRING
from .prompts import (
    DECOMPOSITION_INSTRUCTIONS,
    RECOMPOSITION_INSTRUCTIONS,
    SINGLE_QUESTION_PROMPT,
    RECOMP_ANSWER_PROMPT,
)


def decompose(
    model,
    tokenizer,
    question,
    choice_list,
    decomp_chat,
    log_path,
    verbose=False,
    max_rounds=10,
):
    """
    Decompose a question into subquestions which are all answered independently by the model.
    """
    log_path = log_path / "decomp"
    log_path.mkdir(parents=True, exist_ok=True)
    decomp_chat = decomp_chat.copy()
    decomp_chat.append(question_message(question, choice_list))
    # Little hack to make sure the prompt format is correct for the assistant message format of the model
    subqa = []
    for round in range(1, max_rounds + 1):
        with open(log_path / f"decomp_chat_{round}.json", "w") as f:
            json.dump(decomp_chat, f, indent=4)
        prompt_chat = decomp_chat + [assistant_message(f"<{SPLIT_STRING}")]
        prompt = (
            tokenizer.apply_chat_template(prompt_chat, tokenize=False)
            .split(SPLIT_STRING)[0]
            .replace("{DECOMPOSITION_INSTRUCTIONS}", DECOMPOSITION_INSTRUCTIONS)
        )
        if verbose:
            print(f"Prompt:==============\n{prompt[-150:]}\n====================\n")
        model_questions = "<" + query_model(
            prompt, model, stop_token(model), log_path=log_path, max_tokens=2048
        )
        if "<FIN></FIN>" in model_questions:
            break
        subquestions = []
        cleaned_model_questions = []
        for subquestion in model_questions.split("\n"):
            if subquestion == "":
                continue
            try:
                match_start = regex.search(r"<sub_q_(\d+)>", subquestion)
                match_end = regex.search(r"</sub_q_(\d+)>", subquestion)
                if match_start and match_end:
                    sub_q_index = int(match_start.group(1))
                    if sub_q_index != int(match_end.group(1)):
                        raise IndexError
                    sub_q_content = subquestion[
                        match_start.end() : match_end.start()
                    ].strip()
                else:
                    raise IndexError
                subquestions.append((sub_q_index, sub_q_content))
                cleaned_model_questions.append(subquestion)
            except IndexError:
                print(f"Invalid subquestion format: {subquestion}")
        answers = []
        subquestions_path = log_path / f"subquestions_{round}.json"
        json_save = [{"prompt": SINGLE_QUESTION_PROMPT}]
        for i, subquestion in subquestions:
            if regex.search(r"\{sub_(a|q)_.*\}", subquestion):
                json_save.append(
                    {
                        "id": i,
                        "subquestion": subquestion,
                        "answer": "{NONE}",
                    }
                )
                continue
            chat = []
            chat.append(user_message(SINGLE_QUESTION_PROMPT + subquestion))
            chat.append(assistant_message(SPLIT_STRING))
            prompt = tokenizer.apply_chat_template(chat, tokenize=False).split(
                SPLIT_STRING
            )[0]
            response = query_model(
                prompt, model, stop_token(model), log_path=log_path, max_tokens=2048
            )
            answer = f"<sub_a_{i}>" + response + f"</sub_a_{i}>"
            answers.append(answer)
            subqa.append((subquestion, response))
            json_save.append(
                {
                    "id": i,
                    "subquestion": subquestion,
                    "answer": response,
                }
            )
        with open(subquestions_path, "w") as f:
            json.dump(json_save, f, indent=4)
        if verbose:
            print(f"Answers:\n{answers}")
        decomp_chat.append(assistant_message("\n".join(cleaned_model_questions)))
        decomp_chat.append(user_message("\n".join(answers)))
    else:
        print(
            f"Max {max_rounds} rounds reached for question {question}. See logs in {str(log_path)}"
        )
    chat_path = log_path / "decomp_chat.json"
    with open(chat_path, "w") as f:
        json.dump(decomp_chat, f, indent=4)
    decomp_res_path = log_path / "decomp_res.json"
    with open(decomp_res_path, "w") as f:
        json.dump({q: a for q, a in subqa}, f, indent=4)
    return subqa


def recompose(model, tokenizer, question, choice_list, subqa, recomp_chat, log_path):
    """
    Given a question and a list of subquestions and answers, answer the original question.
    """
    log_path = log_path / "recomp"
    log_path.mkdir(parents=True, exist_ok=True)
    choices = [
        f"({QUESTION_LABELS[i]}) {choice}" for i, choice in enumerate(choice_list)
    ]
    choices = "\n".join(choices)
    recomp_chat = recomp_chat.copy()
    recomp_chat.append(user_message(f"Question: {question}\n\nChoices: {choices}"))
    recomp_chat[-1]["content"] += "\n\nSubquestions and answers:\n" + "\n".join(
        [f"Question: {q} Answer: {a}" for q, a in subqa]
    )
    recomp_chat.append(assistant_message(f"{RECOMP_ANSWER_PROMPT}{SPLIT_STRING}"))
    prompt = (
        tokenizer.apply_chat_template(recomp_chat, tokenize=False)
        .split(SPLIT_STRING)[0]
        .replace("{RECOMPOSITION_INSTRUCTIONS}", RECOMPOSITION_INSTRUCTIONS)
    )
    answer = query_model(
        prompt, model, stop_token(model), log_path=log_path, max_tokens=1, temperature=0
    )
    with open(log_path / "recomp_chat.json", "w") as f:
        recomp_chat[-1] = assistant_message(f"{RECOMP_ANSWER_PROMPT}{answer}")
        json.dump(recomp_chat, f, indent=4)
    return answer
