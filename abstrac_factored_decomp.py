import random
import json
import datasets
import argparse
import together
from transformers import AutoTokenizer
from time import time
from fd_prompts import *
import regex
from warnings import warn
from pathlib import Path
from tqdm.auto import tqdm


LOG_PATH = Path("logs")


START_OF_ANSWER = "Based on the above,"
SPLIT_STRING = "VERYUGLYSPLITSTRING_"
question_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]

available_models = [
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "meta-llama/Llama-2-70b-chat-hf",
    "togethercomputer/LLaMA-2-7B-32K",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x7B-v0.1",
]


def user_message(message):
    return {
        "role": "user",
        "content": message,
    }


def assistant_message(message):
    return {
        "role": "assistant",
        "content": message,
    }


def system_message(message):
    return {
        "role": "system",
        "content": message,
    }


def question_message(question, choices):
    choices = [f"({question_labels[i]}) {choice}" for i, choice in enumerate(choices)]
    choices = "\n".join(choices)
    return user_message(f"Question: {question}\n\nChoices: {choices}")


def stop_token(model):
    if (
        "nousresearch" in model.lower()
        or AutoTokenizer.from_pretrained(model).chat_template is None
    ):
        return ["<|im_end|>\n"]
    elif "mistralai" in model.lower() or "llama" in model.lower():
        return ["</s>"]
    else:
        raise ValueError(f"Unknown stop token for model {model}")


def query_model(prompt, model, stop, log_path=None, **kwargs) -> str:
    default_kwargs = dict(top_p=0.95, temperature=0.8)
    kwargs = {**default_kwargs, **kwargs}
    api_response = together.Complete.create(
        prompt,
        model=model,
        stop=stop,
        **kwargs,
    )
    if log_path is None:
        log_path = LOG_PATH / model
    save_path = log_path / "requests" / f"answer_{int(time())}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(api_response, f, indent=4)
    return api_response["output"]["choices"][0]["text"]


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
    log_path = log_path / "recomp"
    log_path.mkdir(parents=True, exist_ok=True)
    choices = [
        f"({question_labels[i]}) {choice}" for i, choice in enumerate(choice_list)
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


def run_on_dataset(
    model, tokenizer, dataset, method, log_path, shuffle_choices=False, **kwargs
):
    """
    Evaluates a given method on a dataset.

    Args:
        model: The model to use.
        tokenizer: The tokenizer to use.
        dataset: The dataset to evaluate on.
        method: The method to evaluate (e.g., few_shot, factored_decomposition).
        log_path: The path to log the results.
        shuffle_choices: Whether to shuffle the choices before evaluation.
        **kwargs: Additional arguments to pass to the method.

    Returns:
        results: A list of 1s and 0s, where 1 indicates the model's prediction was correct and 0 indicates it was wrong.
        json_result: A list of dictionaries containing the question, labels, choices, and the model's answer.
    """
    results = []
    json_result = []
    for sample in tqdm(dataset):
        labels = sample["mc1_targets"]["labels"]
        choices = sample["mc1_targets"]["choices"]
        if shuffle_choices:
            permutation = list(range(len(choices)))
            random.shuffle(permutation)
            labels = [labels[i] for i in permutation]
            choices = [choices[i] for i in permutation]
        question = sample["question"]
        answer = method(model, tokenizer, question, choices, **kwargs)
        json_result.append(
            {
                "question": question[:100],
                "labels": labels,
                "choices": choices,
                "answer": answer,
            }
        )
        try:
            prediction = question_labels.index(answer)
            results.append(labels[prediction])
        except ValueError:
            print(f"Invalid answer: {answer}")
            results.append(0)
    with open(log_path / "all_results.json", "w") as f:
        json.dump(json_result, f, indent=4)
    return results


def few_shot(model, tokenizer, question, choices, few_shot_chat, log_path):
    """
    Runs the few-shot method on a single question and set of choices.

    Args:
        model: The model to use.
        tokenizer: The tokenizer to use.
        question: The question to answer.
        choices: The list of choices for the question.
        few_shot_chat: The chat prompt to use for the few-shot method.
        log_path: The path to log the results.

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


def chain_of_thought(model, tokenizer, question, choices, cot_chat, log_path):
    """
    Runs the chain of thought method on a single question and set of choices.

    Args:
        model: The model to use.
        tokenizer: The tokenizer to use.
        question: The question to answer.
        choices: The list of choices for the question.
        cot_chat: The chat prompt to use for the chain of thought method.
        log_path: The path to log the results.

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
    model, tokenizer, question, choices, cotd_chat, log_path
):
    """
    Runs the chain of thought decomposition method on a single question and set of choices.

    Args:
        model: The model to use.
        tokenizer: The tokenizer to use.
        question: The question to answer.
        choices: The list of choices for the question.
        cotd_chat: The chat prompt to use for the chain of thought decomposition method.
        log_path: The path to log the results.

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
    decomp_chat,
    recomp_chat,
    log_path,
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
        decomp_chat: The chat prompt to use for the decomposition step.
        recomp_chat: The chat prompt to use for the recomposition step.
        log_path: The path to log the results.
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run different CoT strategies on a dataset"
    )
    parser.add_argument(
        "--model",
        help="The model to run the factored decomposition on",
        type=str,
        default="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="The dataset to run factored decomposition on",
        default="truthful_qa",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        help="N first samples to select from the dataset",
        default=100,
    )
    # add subcommand for each decomposition
    sub_parsers = parser.add_subparsers(dest="subcommand")

    parser_fd = sub_parsers.add_parser("factdec", help="Run factored decomposition")
    parser_fd.add_argument(
        "--skip-system",
        action="store_true",
        help="Whether to skip the system message in the chat",
    )
    parser_fd.add_argument(
        "--skip-user-instruction",
        action="store_true",
        help="Whether to skip the user instruction in the chat",
    )
    parser_fd.add_argument(
        "--shuffle-choices",
        action="store_true",
        help="Whether to shuffle the choices before running the decomposition",
    )

    parser_fs = sub_parsers.add_parser("fewshot", help="Run few shot")
    parser_cot = sub_parsers.add_parser("cot", help="Run chain of thought")
    parser_cotd = sub_parsers.add_parser(
        "cotd", help="Run chain of thought decomposition"
    )

    args, unknown = parser.parse_known_args()
    print(args)
    model = args.model
    if "meta-llama" in model and "chat" in model:
        tokenizer = AutoTokenizer.from_pretrained(
            "DeepInfra/Llama-2-70b-chat-tokenizer"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model)
        if tokenizer.chat_template is None:
            warn(
                f"Chat template not found for model {model}, using NousResearch tokenizer instead"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
            )
    dataset_name = args.dataset
    dataset = datasets.load_dataset(dataset_name, "multiple_choice")["validation"]
    dataset = dataset.select(range(args.num_samples))
    if args.subcommand == "factdec" or args.subcommand is None:
        f_args = parser_fd.parse_args(unknown)
        fd_chats = json.load(open("data/new_fd minimal.json"))
        start_time = int(time())
        log_path = (
            LOG_PATH / model / "factored_decomposition" / dataset_name / str(start_time)
        )
        log_path.mkdir(parents=True, exist_ok=True)
        results = run_on_dataset(
            model,
            tokenizer,
            dataset,
            factored_decomposition,
            log_path,
            shuffle_choices=f_args.shuffle_choices,
            decomp_chat=fd_chats["decomp_prompt"],
            recomp_chat=fd_chats["recomp_prompt"],
            skip_system=f_args.skip_system,
            skip_user_instruction=f_args.skip_user_instruction,
        )
        print(f"Accuracy on {dataset_name}: {sum(results) / len(results)}")
        with open(log_path / "results.json", "w") as f:
            json.dump(results, f, indent=4)
    elif args.subcommand == "fewshot":
        few_shot_chat = json.load(open("data/few_shot_chat minimal.json"))
        start_time = int(time())
        log_path = LOG_PATH / model / "few_shot" / dataset_name / str(start_time)
        log_path.mkdir(parents=True, exist_ok=True)
        results = run_on_dataset(
            model,
            tokenizer,
            dataset,
            few_shot,
            log_path,
            few_shot_chat=few_shot_chat,
        )
        print(f"Accuracy on {dataset_name}: {sum(results) / len(results)}")
        with open(log_path / "results.json", "w") as f:
            json.dump(results, f, indent=4)
    elif args.subcommand == "cot":
        cot_chat = json.load(open("data/cot_few_shot_chat minimal.json"))
        start_time = int(time())
        log_path = (
            LOG_PATH / model / "chain_of_thought" / dataset_name / str(start_time)
        )
        log_path.mkdir(parents=True, exist_ok=True)
        results = run_on_dataset(
            model,
            tokenizer,
            dataset,
            chain_of_thought,
            log_path,
            cot_chat=cot_chat,
        )
        print(f"Accuracy on {dataset_name}: {sum(results) / len(results)}")
        with open(log_path / "results.json", "w") as f:
            json.dump(results, f, indent=4)
    elif args.subcommand == "cotd":
        cotd_chat = json.load(open("data/cotd_few_shot_chat minimal.json"))
        start_time = int(time())
        log_path = (
            LOG_PATH
            / model
            / "chain_of_thought_decomposition"
            / dataset_name
            / str(start_time)
        )
        log_path.mkdir(parents=True, exist_ok=True)
        results = run_on_dataset(
            model,
            tokenizer,
            dataset,
            chain_of_thought_decomposition,
            log_path,
            cotd_chat=cotd_chat,
        )
        print(f"Accuracy on {dataset_name}: {sum(results) / len(results)}")
        with open(log_path / "results.json", "w") as f:
            json.dump(results, f, indent=4)
