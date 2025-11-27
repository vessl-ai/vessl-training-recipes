import argparse

import torch
from datasets import load_dataset
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from vessl.integration.transformers import VesslCallback


def main():
    # Parsing command line arguments
    parser = argparse.ArgumentParser(description="Fine-tune a language model with specified parameters.")
    parser.add_argument('--max-steps', type=int, default=300, help="Maximum number of training steps.")
    parser.add_argument('--dataset', type=str, default="unsloth/LaTeX_OCR", help="Dataset to load.")
    parser.add_argument('--base-model-name', type=str, default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit", help="Base model name.")
    parser.add_argument('--checkpoint-path', type=str, default="./output/checkpoints", help="Path to the checkpoint to save.")
    parser.add_argument('--output-model-name', type=str, default="./output/finetuned_model", help="Output directory for the trained model.")
    parser.add_argument('--max-seq-length', type=int, default=32768, help="Maximum sequence length.")
    parser.add_argument('--batch-size', type=int, default=4, help="Number of samples per batch per device during training.")
    parser.add_argument('--train-epochs', type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument('--lora-rank', type=int, default=16, help="Inner dimension of the low-rank matrices to train; a higher rank means more trainable parameters.")
    args = parser.parse_args()

    # Loading the dataset
    dataset = load_dataset(args.dataset, split="train")

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
    
    converted_dataset = [convert_to_conversation(sample) for sample in dataset]

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
        # max_steps=args.max_steps,                        # Maximum number of steps to train the model
        gradient_accumulation_steps=1,                     # Number of updates steps to accumulate before performing a backward/update pass
        warmup_steps=2,                                    # Linear warmup over warmup_steps
        learning_rate=2.5e-4,                              # Learning rate for the optimizer
        optim="paged_adamw_8bit",                          # The optimizer to use
        gradient_checkpointing=True,                       # Use gradient checkpointing to save memory
        fp16=not torch.cuda.is_bf16_supported(),           # Whether to use 16-bit (mixed) precision training
        bf16=torch.cuda.is_bf16_supported(),               # Whether to use bfloat16 precision training
        logging_steps=25,                                  # Log every X updates steps
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
        train_dataset=converted_dataset,
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