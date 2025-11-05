#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune Henrychur/MMed-Llama-3-8B on (report → ICD-11 title)
 – LoRA r16/α32, bf16
 – batched preprocessing
 – default_data_collator
"""
import os, types, random, numpy as np, pandas as pd, torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    default_data_collator
)
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split



os.environ["TRANSFORMERS_CACHE"] = "/project/ywu10/cache_directory"

SEED = 42



# ============== Data ==============
train_df = pd.read_csv("version2/mimic_diagnosis_icd11(train).csv")
train_df = train_df[['report', 'icd11_title']].dropna()

test_df = pd.read_csv("version2/mimic_diagnosis_icd11(test).csv")
test_df = test_df[['report', 'icd11_title']].dropna()

dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df, preserve_index=False),
    'test' : Dataset.from_pandas(test_df,  preserve_index=False),
})


BASE_MODEL = "Henrychur/MMed-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=os.environ["TRANSFORMERS_CACHE"])
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=os.environ["TRANSFORMERS_CACHE"],
)
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))


target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules=target_modules,
)
model = get_peft_model(model, lora_cfg)


model.gradient_checkpointing_enable()
def _enable_input_require_grads(self):
    for p in self.get_input_embeddings().parameters():
        p.requires_grad_(True)
model.enable_input_require_grads = types.MethodType(_enable_input_require_grads, model)
model.enable_input_require_grads()

MAX_LEN     = 1024
PAD_ID      = tokenizer.pad_token_id
PROMPT_TMPL = (
    "Instruction: You are a clinical coding assistant. "
    "Read the patient report and output the most appropriate ICD-11 diagnosis title.\n"
    "Patient report: {report}\nDiagnosis:"
)

def preprocess_batch(batch):
    input_ids, attn_masks, labels = [], [], []

    for rpt, tgt in zip(batch["report"], batch["icd11_title"]):
        tgt = str(tgt).strip()
        if not tgt:
            continue

        p_ids = tokenizer(PROMPT_TMPL.format(report=rpt),
                          add_special_tokens=False).input_ids
        t_ids = tokenizer(tgt, add_special_tokens=False).input_ids
        if len(t_ids) == 0:
            continue

        ids   = (p_ids + t_ids)[:MAX_LEN]
        labs  = ([-100]*len(p_ids) + t_ids)[:MAX_LEN]

        pad_len = MAX_LEN - len(ids)
        if pad_len:
            ids  += [PAD_ID]*pad_len
            labs += [-100]*pad_len

        input_ids.append(ids)
        attn_masks.append([1]*MAX_LEN)
        labels.append(labs)

    return {"input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": labels}

tokenized_ds = ds.map(
    preprocess_batch,
    batched=True,
    remove_columns=["report","icd11_title"],
    desc="Tokenizing (batched)",
)

# ───────────────────── 5. TrainingArguments ─────────────────────
args = TrainingArguments(
    output_dir                 = "./medllama-finetune",
    per_device_train_batch_size=2,
    per_device_eval_batch_size =2,
    gradient_accumulation_steps=8,
    num_train_epochs           =3,
    learning_rate              =1e-4,
    lr_scheduler_type          ="cosine",
    warmup_ratio               =0.1,
    max_grad_norm              =1.0,
    bf16                       =True,
    fp16                       =False,
    logging_steps              =50,
    eval_strategy              ="steps",
    eval_steps                 =200,
    save_steps                 =200,
    load_best_model_at_end     =True,
    metric_for_best_model      ="eval_loss",
    greater_is_better          =False,
    save_total_limit           =1,
    report_to                  ="none",
)


trainer = Trainer(
    model         = model,
    args          = args,
    train_dataset = tokenized_ds["train"],
    eval_dataset  = tokenized_ds["test"],
    data_collator = default_data_collator,
)


trainer.train()
trainer.save_model("model/medllama_disease_finetuned")
tokenizer.save_pretrained("model/medllama_disease_finetuned")
