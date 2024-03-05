from pathlib import Path
import together
from .constants import QUESTION_LABELS
import json
from transformers import AutoTokenizer
from time import time

LOG_PATH = Path(__file__).parent.parent / "logs"


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
    choices = [f"({QUESTION_LABELS[i]}) {choice}" for i, choice in enumerate(choices)]
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
