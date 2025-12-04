import argparse
import os
from typing import List, Optional

import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import PPOConfig, PPOTrainer
from vessl.integration.transformers import VesslCallback


def get_datasets(
    data_config: dict,
    splits: List[str] = ["train", "test"],
    shuffle: bool = True,
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`DataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.

    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """

    if type(data_config) is dict:
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset1": 0.3,
        #             "dataset1": 0.2,
        #         }
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")

    raw_datasets = mix_datasets(dataset_mixer, splits=splits, shuffle=shuffle)
    return raw_datasets


def mix_datasets(
    dataset_mixer: dict, splits: Optional[List[str]] = None, shuffle=True
) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for ds, frac in dataset_mixer.items():
        fracs.append(frac)
        for split in splits:
            try:
                # Try first if dataset on a Hub repo
                dataset = load_dataset(ds, split=split)
            except DatasetGenerationError:
                # If not, check local dataset
                dataset = load_from_disk(os.path.join(ds, split))

            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(
                    f"Split type {split} not recognized as one of test or train."
                )

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(
                seed=42
            )
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with split {split}. Check the dataset has been correctly formatted."
        )

    return raw_datasets


def main():
    # Parsing command line arguments
    parser = argparse.ArgumentParser(description="Fine-tune a language model with specified parameters.")
    parser.add_argument('--dataset', type=str, default="HuggingFaceH4/ultrafeedback_binarized", help="Dataset to load.")
    parser.add_argument('--fraction', type=float, default=1, help="Dataset fraction to use.")
    parser.add_argument('--base-model-name', type=str, default="unsloth/Qwen3-4B-Instruct-2507-bnb-4bit", help="Base model name.")
    parser.add_argument('--reward-model-name', type=str, default="Skywork/Skywork-Reward-V2-Qwen3-0.6B", help="Reward model name.")
    parser.add_argument('--checkpoint-path', type=str, default="./output/checkpoints", help="Path to the checkpoint to save.")
    parser.add_argument('--output-model-name', type=str, default="./output/finetuned_model", help="Output directory for the trained model.")
    parser.add_argument('--max-seq-length', type=int, default=4096, help="Maximum sequence length.")
    parser.add_argument('--batch-size', type=int, default=2, help="Number of samples per batch per device during training.")
    parser.add_argument('--train-epochs', type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument('--lora-rank', type=int, default=16, help="Inner dimension of the low-rank matrices to train; a higher rank means more trainable parameters.")
    args = parser.parse_args()
    
    tokenizer=AutoTokenizer.from_pretrained(args.base_model_name, padding_side="left")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    raw_datasets = get_datasets(
        {args.dataset : args.fraction}, # 0.5% sampled
        splits = ["train_prefs", "test_prefs"],
    )
    column_names = list(raw_datasets["train"].features)

    def process_func(example):
        input_ids = tokenizer.apply_chat_template(
            example["messages"],
            padding=False,
            add_generation_prompt=True,
        )
        return {"input_ids": input_ids, "lengths": len(input_ids)}

    converted_datasets = raw_datasets.map(
        process_func,
        remove_columns = column_names,
    )
    converted_datasets = converted_datasets.filter(lambda x: x["lengths"] <= 128)

    peft_config= LoraConfig(
        r=args.lora_rank,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        device_map="auto",
        max_length=args.max_seq_length,
        dtype=torch.bfloat16,
    )

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_name,
        device_map="auto",
        dtype=torch.bfloat16,
        num_labels=1,
    )
    value_model = reward_model

    config=PPOConfig(
        num_train_epochs=args.train_epochs,
        learning_rate=1e-5,
        batch_size=args.batch_size,
        num_mini_batches=1,
        gradient_accumulation_steps=1,
        logging_steps=25,
    )
    ppo_trainer=PPOTrainer(
        args=config,
        model=model,
        ref_model=None,
        processing_class=tokenizer,
        train_dataset=converted_datasets["train"],
        eval_dataset=converted_datasets["test"],
        reward_model=reward_model,
        value_model=value_model,
        peft_config=peft_config,
        callbacks=[
            VesslCallback(),
        ],
    )
    ppo_trainer.train()


if __name__ == "__main__":
    main()