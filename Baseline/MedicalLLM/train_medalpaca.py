#!/usr/bin/env python
# fine-tune Henrychur/MMed-Llama-3-8B —— LoRA + bf16


import os, types, random, numpy as np, pandas as pd, torch
from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from huggingface_hub import login



os.environ["TRANSFORMERS_CACHE"] = "/project/ywu10/cache_directory"
SEED = 42



train_df = pd.read_csv("version2/mimic_diagnosis_icd11(train).csv")
train_df = train_df[['report', 'icd11_title']].dropna()

test_df = pd.read_csv("version2/mimic_diagnosis_icd11(test).csv")
test_df = test_df[['report', 'icd11_title']].dropna()

dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df, preserve_index=False),
    'test' : Dataset.from_pandas(test_df,  preserve_index=False),
})


model_name = "medalpaca/medalpaca-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          cache_dir=os.environ["TRANSFORMERS_CACHE"])
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype = torch.bfloat16,
    device_map  = "auto",
    cache_dir   = os.environ["TRANSFORMERS_CACHE"],
)
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ],
)
model = get_peft_model(model, lora_cfg)

model.gradient_checkpointing_enable()
def _enable_input_require_grads(self):
    for p in self.get_input_embeddings().parameters():
        p.requires_grad_(True)
model.enable_input_require_grads = types.MethodType(_enable_input_require_grads, model)
model.enable_input_require_grads()


MAX_LEN = 1024
PROMPT_TMPL = ("Instruction: You are a clinical coding assistant. "
               "Read the patient report and output the most appropriate "
               "ICD-11 diagnosis title. Patient report: {report}\nDiagnosis:")

def preprocess(example):
    prompt = PROMPT_TMPL.format(report=example["report"])
    target = str(example["icd11_title"]).strip()

    if target == "":
        return {"skip": True}

    prompt_ids  = tokenizer(prompt,  add_special_tokens=False).input_ids
    target_ids  = tokenizer(target,  add_special_tokens=False).input_ids
    if len(target_ids) == 0:
        return {"skip": True}

    input_ids      = prompt_ids + target_ids
    attention_mask = [1] * len(input_ids)
    labels         = [-100] * len(prompt_ids) + target_ids

    pad_len = MAX_LEN - len(input_ids)
    pad_id  = tokenizer.pad_token_id

    if pad_len < 0:
        input_ids      = input_ids[:MAX_LEN]
        attention_mask = attention_mask[:MAX_LEN]
        labels         = labels[:MAX_LEN]
    else:
        input_ids     += [pad_id] * pad_len
        attention_mask+= [0]     * pad_len
        labels        += [-100]  * pad_len

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        "skip":           False,
    }

tokenized_ds = (
    ds.map(preprocess, remove_columns=["report","icd11_title"])
      .filter(lambda x: x["skip"] is False)
      .remove_columns(["skip"])
)

# ────────── 5. collator（静态长度所以用 MLM collator 就行） ──────────
data_collator = DataCollatorForLanguageModeling(
    tokenizer          = tokenizer,
    mlm                = False,    # Causal-LM
)


args = TrainingArguments(
    output_dir                 = "./medllama-finetune",
    per_device_train_batch_size=2,
    per_device_eval_batch_size =2,
    gradient_accumulation_steps=8,
    num_train_epochs           =3,
    learning_rate              =1e-4,
    lr_scheduler_type          ="cosine",
    warmup_steps               =50,
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
    data_collator = data_collator,
)


trainer.train()
trainer.save_model("model/medalpaca_disease_finetuned")
tokenizer.save_pretrained("model/medalpaca_disease_finetuned")
