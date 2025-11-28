import argparse
import os
from typing import List, Optional

import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
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


def convert_to_conversation(sample):
    instruction = "Write the LaTeX representation for this image."
    conversation = [
        { "role": "user",
        "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
        "content" : [
            {"type" : "text",  "text"  : sample["text"]} ]
        },
    ]
    return { "messages" : conversation }


def main():
    # Parsing command line arguments
    parser = argparse.ArgumentParser(description="Fine-tune a language model with specified parameters.")
    parser.add_argument('--max-steps', type=int, default=300, help="Maximum number of training steps.")
    parser.add_argument('--dataset', type=str, default="unsloth/LaTeX_OCR", help="Dataset to load.")
    parser.add_argument('--fraction', type=float, default=1, help="Dataset fraction to use.")
    parser.add_argument('--base-model-name', type=str, default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit", help="Base model name.")
    parser.add_argument('--checkpoint-path', type=str, default="./output/checkpoints", help="Path to the checkpoint to save.")
    parser.add_argument('--output-model-name', type=str, default="./output/finetuned_model", help="Output directory for the trained model.")
    parser.add_argument('--max-seq-length', type=int, default=4096, help="Maximum sequence length.")
    parser.add_argument('--batch-size', type=int, default=4, help="Number of samples per batch per device during training.")
    parser.add_argument('--train-epochs', type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument('--lora-rank', type=int, default=16, help="Inner dimension of the low-rank matrices to train; a higher rank means more trainable parameters.")
    args = parser.parse_args()

    # Loading the dataset
    raw_datasets = get_datasets(
        {args.dataset : args.fraction},
        splits = ["train", "test"],
    )

    converted_dataset_train = [convert_to_conversation(sample) for sample in raw_datasets["train"]]
    converted_dataset_test = [convert_to_conversation(sample) for sample in raw_datasets["test"]]

    # Loading the base model and tokenizer
    base_model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.base_model_name,
        max_seq_length=args.max_seq_length,  # Maximum sequence length
        load_in_4bit=True,
    )

    model = FastVisionModel.get_peft_model(
        base_model,
        finetune_vision_layers=True,
        finetune_language_laters=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",     # Supports any, but = "none" is optimized
        use_gradient_checkpointing=True,
    )

    # Hyperparameters
    training_arguments = SFTConfig(
        output_dir=args.checkpoint_path,                   # Directory for saving checkpoints
        logging_dir="./logs",                              # Directory for storing logs
        per_device_train_batch_size=args.batch_size,       # Number of samples per batch per device during training
        num_train_epochs=args.train_epochs,                # Total number of training epochs to perform
        gradient_accumulation_steps=4,                     # Number of updates steps to accumulate before performing a backward/update pass
        warmup_steps=5,                                    # Linear warmup over warmup_steps
        learning_rate=2.5e-4,                              # Learning rate for the optimizer
        optim="adamw_8bit",                                # The optimizer to use
        weight_decay=0.001,
        gradient_checkpointing=True,                       # Use gradient checkpointing to save memory
        fp16=not torch.cuda.is_bf16_supported(),           # Whether to use 16-bit (mixed) precision training
        bf16=torch.cuda.is_bf16_supported(),               # Whether to use bfloat16 precision training
        logging_steps=1,                                   # Log every X updates steps
        save_steps=50,                                     # Save checkpoint every X updates steps
        save_strategy="steps",                             # The checkpoint saving strategy to adopt during training
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=2048
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Setting SFT parameters
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted_dataset_train,
        eval_dataset=converted_dataset_test,
        args=training_arguments,
        callbacks=[
            VesslCallback()
        ],
    )

    trainer.train()

    # Save the model
    model.save_pretrained(args.output_model_name)
    tokenizer.save_pretrained(args.output_model_name)
    model.config.use_cache = True

if __name__ == "__main__":
    main()