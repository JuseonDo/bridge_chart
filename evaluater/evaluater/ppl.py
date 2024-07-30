import math
from typing import List
from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
import gc
import torch

def eval_ppl(predictions: List[str]):
    model_name_or_path = 'gpt2-medium'
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)

    # Tokenize the data
    def tokenize_function(examples):
        return tokenizer(examples['text'])

    raw_datasets = DatasetDict({
        "validation": Dataset.from_dict({"text": predictions})
    })

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    block_size = tokenizer.model_max_length
    if block_size > 1024:
        block_size = 1024

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )


    eval_dataset = lm_datasets["validation"]

    # Initialize Trainer
    training_args = TrainingArguments(output_dir='./results', do_eval=True, per_device_eval_batch_size=2)
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # Evaluate and compute perplexity
    metrics = trainer.evaluate()
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")

    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return perplexity
