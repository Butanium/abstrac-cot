import random
import json
import argparse
from transformers import AutoTokenizer
from time import time
from warnings import warn
from pathlib import Path
from tqdm.auto import tqdm
from src.constants import QUESTION_LABELS
from src.strategies import (
    zero_shot,
    few_shot,
    chain_of_thought,
    chain_of_thought_decomposition,
    factored_decomposition,
)
from src.dataset_utils import get_dataset

LOG_PATH = Path("logs")


def run_on_dataset(
    model,
    tokenizer,
    dataset,
    process_sample,
    method,
    log_path,
    shuffle_choices=False,
    **kwargs,
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
        labels, choices = process_sample(sample)
        if shuffle_choices:
            permutation = list(range(len(choices)))
            random.shuffle(permutation)
            labels = [labels[i] for i in permutation]
            choices = [choices[i] for i in permutation]
        question = sample["question"]
        answer = method(model, tokenizer, question, choices, log_path, **kwargs)
        json_result.append(
            {
                "question": question[:100],
                "labels": labels,
                "choices": choices,
                "answer": answer,
            }
        )
        try:
            prediction = QUESTION_LABELS.index(answer)
            results.append(labels[prediction])
        except (ValueError, IndexError):
            print(f"Invalid answer: {answer}")
            results.append(0)
    with open(log_path / "all_results.json", "w") as f:
        json.dump(json_result, f, indent=4)
    return results


def run_all_methods(
    model,
    tokenizer,
    dataset_name,
    num_samples,
    *,
    fd_chats,
    few_shot_chat,
    cot_chat,
    cotd_chat,
    methods=None,
    shuffle_choices=False,
    start_time=None,
    verbose=False,
):
    methods_dic = {
        "factored_decomposition": (
            factored_decomposition,
            {
                "decomp_chat": fd_chats["decomp_prompt"],
                "recomp_chat": fd_chats["recomp_prompt"],
            },
        ),
        "few_shot": (few_shot, {"few_shot_chat": few_shot_chat}),
        "chain_of_thought": (chain_of_thought, {"cot_chat": cot_chat}),
        "chain_of_thought_decomposition": (
            chain_of_thought_decomposition,
            {"cotd_chat": cotd_chat},
        ),
        "zero_shot": (zero_shot, {}),
    }
    if methods is None:
        methods = methods_dic.keys()
    else:
        assert set(methods).issubset(
            methods_dic.keys()
        ), f"Unknown methods: {methods - methods_dic.keys()}"
    if start_time is None:
        start_time = int(time())
    results = {}
    result_path = LOG_PATH / model / dataset_name / f"{start_time}_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    for method_name in methods:
        method, kwargs = methods_dic[method_name]
        if verbose:
            print(f"Running {method_name} on {dataset_name}")
        log_path = LOG_PATH / model / method_name / dataset_name / str(start_time)
        log_path.mkdir(parents=True, exist_ok=True)

        dataset, get_label_and_choices = get_dataset(dataset_name)
        dataset_subset = dataset.select(range(num_samples))

        method_results = run_on_dataset(
            model,
            tokenizer,
            dataset_subset,
            get_label_and_choices,
            method,
            log_path,
            shuffle_choices=shuffle_choices,
            **kwargs,
        )

        accuracy = sum(method_results) / len(method_results)
        results[method_name] = accuracy
        if verbose:
            print(f"Accuracy on {dataset_name} ({method_name}): {accuracy}")
        with open(result_path, "w") as f:
            json.dump(results, f, indent=4)

    return results


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
        "-d",
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
    parser.add_argument(
        "--shuffle-choices",
        action="store_true",
        help="Whether to shuffle the choices before running the decomposition",
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

    parser_fs = sub_parsers.add_parser("fewshot", help="Run few shot")
    parser_cot = sub_parsers.add_parser("cot", help="Run chain of thought")
    parser_cotd = sub_parsers.add_parser(
        "cotd", help="Run chain of thought decomposition"
    )
    parser_all = sub_parsers.add_parser("all", help="Run all methods")
    parser_all.add_argument(
        "--methods",
        nargs="+",
        type=str,
        help="The methods to run",
        default=None,
    )

    args, unknown = parser.parse_known_args()
    print(args)
    model = args.model  # TODO: Fix system message not available for mixtral instruct
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
    dataset, get_label_and_choices = get_dataset(dataset_name)
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
            get_label_and_choices,
            factored_decomposition,
            log_path,
            shuffle_choices=args.shuffle_choices,
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
            get_label_and_choices,
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
            get_label_and_choices,
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
            get_label_and_choices,
            chain_of_thought_decomposition,
            log_path,
            cotd_chat=cotd_chat,
        )
        print(f"Accuracy on {dataset_name}: {sum(results) / len(results)}")
        with open(log_path / "results.json", "w") as f:
            json.dump(results, f, indent=4)
    elif args.subcommand == "all":
        all_args = parser_all.parse_args(unknown)
        few_shot_chat = json.load(open("data/few_shot_chat minimal.json"))
        fd_chats = json.load(open("data/new_fd minimal.json"))
        cot_chat = json.load(open("data/cot_few_shot_chat minimal.json"))
        cotd_chat = json.load(open("data/cotd_few_shot_chat minimal.json"))
        start_time = int(time())
        results = run_all_methods(
            model,
            tokenizer,
            dataset_name,
            args.num_samples,
            methods=args.methods,
            fd_chats=fd_chats,
            few_shot_chat=few_shot_chat,
            cot_chat=cot_chat,
            cotd_chat=cotd_chat,
            shuffle_choices=args.shuffle_choices,
            start_time=start_time,
            verbose=True,
        )
        print(results)
    else:
        raise ValueError(f"Unknown subcommand: {args.subcommand}")
