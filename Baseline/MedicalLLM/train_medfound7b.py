# -*- coding: utf-8 -*-
"""
LoRA finetune MedFound-7B on patient self-reports → ICD-11 title
- Label-only loss (mask prompt with -100)
- bf16, gradient checkpointing
"""

import os, types
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
# from huggingface_hub import login

# ============== Auth & Cache ==============
os.environ['TRANSFORMERS_CACHE'] = '/project/ywu10/cache_directory'

# ============== Data ==============
train_df = pd.read_csv("version2/mimic_diagnosis_icd11(train).csv")
train_df = train_df[['report', 'icd11_title']].dropna()

test_df = pd.read_csv("version2/mimic_diagnosis_icd11(test).csv")
test_df = test_df[['report', 'icd11_title']].dropna()

dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df, preserve_index=False),
    'test' : Dataset.from_pandas(test_df,  preserve_index=False),
})

# ============== Model / Tokenizer ==============
model_name = "medicalai/MedFound-7B"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

# ============== LoRA config（BLOOM 模块名） ==============
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "query_key_value",   # self_attention.query_key_value
        "dense",             # self_attention.dense
        "dense_h_to_4h",     # mlp.dense_h_to_4h
        "dense_4h_to_h"      # mlp.dense_4h_to_h
    ],
)

model = get_peft_model(model, lora_config)

model.gradient_checkpointing_enable()

def _enable_input_require_grads(self):
    for p in self.get_input_embeddings().parameters():
        p.requires_grad_(True)
model.enable_input_require_grads = types.MethodType(_enable_input_require_grads, model)
model.enable_input_require_grads()

# ============== Preprocess ==============
MAX_LEN = 1024

def preprocess(example):
    prompt = f"Patient report: {example['report']}\nDiagnosis:"
    target = example["icd11_title"]

    p = tokenizer(prompt, add_special_tokens=False)
    t = tokenizer(target, add_special_tokens=False)

    input_ids = p["input_ids"] + t["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(p["input_ids"]) + t["input_ids"] + [tokenizer.eos_token_id]

    # Padding / Truncation
    if len(input_ids) > MAX_LEN:
        input_ids = input_ids[:MAX_LEN]
        attention_mask = attention_mask[:MAX_LEN]
        labels = labels[:MAX_LEN]
    else:
        pad_len = MAX_LEN - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len

    return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

tokenized = dataset.map(preprocess, remove_columns=['report','icd11_title'])

# ============== Training Args ==============
training_args = TrainingArguments(
    output_dir="./medfound7b_lora",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    max_grad_norm=1.0,

    bf16=True, fp16=False,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)

trainer.train()

# ============== Save LoRA adapter ==============
save_dir = "model/medfound7b_disease_finetuned"
trainer.model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)


