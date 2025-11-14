import json
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd
from pathlib import Path

# Function to convert examples into Qwen format
def convert_to_qwen_format(example):
    system_msg = """You are Qwen, created by Alibaba Cloud. You are a helpful and precise assistant.
At the end of each answer, append exactly one token:
- Medicine: <C_MED> if confident, <U_MED> if uncertain.
- Math: <C_MATH> if confident, <U_MATH> if uncertain."""

    user_msg = example["question"]
    assistant_msg = example["tagged_response"]

    formatted = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
    )

    return {"example": formatted}


def convert_to_qwen_format_user(question):
    system_msg = """You are Qwen, created by Alibaba Cloud. You are a helpful and precise assistant.
At the end of each answer, append exactly one token:
- Medicine: <C_MED> if confident, <U_MED> if uncertain.
- Math: <C_MATH> if confident, <U_MATH> if uncertain."""
    user_msg = question
    formatted = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return formatted


def read_train_original_data(data, n_samples=None):
    if data == "maths":
        # Load dataset
        dataset = load_dataset("Akirayasha/math-20", split="train")

        # Only keep question and tagged_response columns
        train_df = pd.DataFrame({
            "question": dataset["question"],
            "tagged_response": dataset["tagged_response"]
        })

        if n_samples is not None and n_samples < len(train_df):
            train_df = train_df[:n_samples]
            print(f"Subsampled to {n_samples} examples for maths dataset.")

        return train_df
    
    elif data == "medqa":
        # Load dataset
        dataset = load_dataset("huyxdang/qwen-medqa-tagged", split="train")

        # Only keep question and tagged_response columns
        train_df = pd.DataFrame({
            "question": dataset["prompt"],
            "tagged_response": dataset["tagged_response"]
        })
        if n_samples is not None and n_samples < len(train_df):
            train_df = train_df[:n_samples]
            print(f"Subsampled to {n_samples} examples for medqa dataset.")

        return train_df

def read_test_data(data, n_samples=None):
    if data == "maths":
        # Load dataset
        dataset = load_dataset("huyxdang/math-split", split="test")

        pre_prompt = "Solve this math problem step by step. Put your final answer in \\boxed{}.\n\n"
        test_input = [pre_prompt + "" + prob for prob in dataset["problem"]]

        # Only keep question and tagged_response columns
        test_df = pd.DataFrame({
            "question": test_input,
            "response": dataset["solution"]
        })
        if n_samples is not None and n_samples < len(test_df):
            test_df = test_df[:n_samples]
            print(f"Subsampled to {n_samples} examples for maths dataset.")

        return test_df
    
    elif data == "medqa":
        # Load dataset
        dataset = load_dataset("huyxdang/medqa-split", split="test")

        def format_prompt(problem, options):
            prompt = "Answer the following medical question by selecting the correct option.\n\n" + problem + "\n\n" + "Options:\n"
            for key,value in options.items():
                prompt += f"{key}. {value}\n"
            prompt += "Answer:"
            return prompt

        test_input = [format_prompt(problem, options) for problem, options in zip(dataset["question"], dataset["options"])]
        
        # Only keep question and tagged_response columns
        test_df = pd.DataFrame({
            "question": test_input,
            "response": dataset["answer_idx"],
        })

        if n_samples is not None and n_samples < len(test_df):
            test_df = test_df[:n_samples]
            print(f"Subsampled to {n_samples} examples for medqa dataset.")

        return test_df
    
    else:
        raise ValueError("Data not supported. Please choose from ['maths', 'medqa'].")








# Read and prepare dataset
def read_data(data, n_samples=None):
    if data == "maths":
        # Load dataset
        dataset = load_dataset("Akirayasha/math-20")

        # Only keep question and tagged_response columns
        train_df = pd.DataFrame({
            "question": dataset["train"]["question"],
            "tagged_response": dataset["train"]["tagged_response"]
        })

        if n_samples is not None and n_samples < len(train_df):
            train_df = train_df[:n_samples]
            print(f"Subsampled to {n_samples} examples for maths dataset.")

        # Convert pandas DataFrame → HF Dataset
        train_dataset = Dataset.from_pandas(train_df)

        # Apply Qwen formatting
        train_dataset = train_dataset.map(convert_to_qwen_format)

        # Wrap in DatasetDict for consistency
        dataset_dict = DatasetDict({
            "train": train_dataset
        })

        return dataset_dict
    
    elif data == "medqa":
        # Load dataset
        dataset = load_dataset("huyxdang/qwen-medqa-tagged")

        # Only keep question and tagged_response columns
        train_df = pd.DataFrame({
            "question": dataset["train"]["prompt"],
            "tagged_response": dataset["train"]["tagged_response"]
        })

        if n_samples is not None and n_samples < len(train_df):
            train_df = train_df[:n_samples]
            print(f"Subsampled to {n_samples} examples for medqa dataset.")

        # Convert pandas DataFrame → HF Dataset
        train_dataset = Dataset.from_pandas(train_df)

        # Apply Qwen formatting
        train_dataset = train_dataset.map(convert_to_qwen_format)

        # Wrap in DatasetDict for consistency
        dataset_dict = DatasetDict({
            "train": train_dataset
        })

        return dataset_dict