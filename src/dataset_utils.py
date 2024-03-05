import datasets


def get_dataset(dataset_name: str):
    if dataset_name == "truthful_qa":
        dataset = datasets.load_dataset("truthful_qa", "multiple_choice")["validation"]
        get_label_and_choices = lambda sample: (
            sample["mc1_targets"]["labels"],
            sample["mc1_targets"]["choices"],
        )
    elif dataset_name == "strategy_qa":
        dataset = datasets.load_dataset("wics/strategy-qa")["test"]
        get_label_and_choices = lambda sample: (
            [int(sample["answer"]), 1 - int(sample["answer"])],
            ["Yes", "No"],
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dataset, get_label_and_choices