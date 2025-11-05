#!/usr/bin/env python
# -*- coding: utf-8 -*-


import re, os, json, math, itertools
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def evaluate_top(
    BASE_MODEL,
    CACHE_DIR,
    PEFT_DIR,
    CSV_FILE,
    SPLIT,
    PREDICTION_FILE,
    prompts,
):

    count_csv = "stat/primary_300_diseases(0708).csv"
    candidate_df = pd.read_csv(count_csv)

    label_list = candidate_df["icd11_title"].tolist()

    label2id = {lbl: idx for idx, lbl in enumerate(label_list)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}


    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        cache_dir=CACHE_DIR,
    )

    if "Llama" in BASE_MODEL or "llama" in BASE_MODEL:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    disease_token_list = [tokenizer.tokenize(l) for l in label_list]
    token_counter = Counter(t for toks in disease_token_list for t in toks)
    label2tokenset = {l: set(toks) for l, toks in zip(label_list, disease_token_list)}

    vocab_size = getattr(tokenizer, "vocab_size", len(tokenizer))
    allow_ids = set(tokenizer.convert_tokens_to_ids(t) for toks in disease_token_list for t in toks)
    bad_words_ids = [[i] for i in range(vocab_size) if i not in allow_ids]


    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
    )
    model = PeftModel.from_pretrained(base, PEFT_DIR)
    model.eval()


    df = pd.read_csv(CSV_FILE)

    split_re = re.compile(r"\s*;\s*")

    def split_tags(text: str):
        return [
            t.strip().lower()
            for t in re.split(split_re, str(text))
            if t and t.strip()
        ]

    df["icd11_title"] = df["icd11_title"].apply(split_tags)

    if SPLIT:
        _, test_df = train_test_split(df, test_size=0.1, random_state=42)
    else:
        test_df = df

    reports = test_df["report"].tolist()
    y_true_ll = test_df["icd11_title"].tolist()  # list[list[str]]


    if prompts is False or prompts is None:
        prompts = [
            f"<report>Patient report: {r}<report>\n<diagnosis>Diagnosis: " for r in reports
        ]

        prompts = [
            f"Patient report: {r}\nDiagnosis:" for r in reports
        ]
    elif isinstance(prompts, str):
        tmpl = prompts
        prompts = [tmpl.format(report=r) for r in reports]
    else:

        if len(prompts) != len(reports):
            raise ValueError(
                f"prompts length {len(prompts)} != reports length {len(reports)}"
            )


    BATCH = 16
    all_rankings = []
    target_pos_out = []
    topk_pred_sets = {k: [] for k in (3, 5, 10)}

    for i in tqdm(range(0, len(prompts), BATCH), desc="Inference"):
        batch = prompts[i: i + BATCH]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            outs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                num_beams=1,
            )

        decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)

        for j, dec in enumerate(decoded):

            text = dec.split("Diagnosis:")[-1]
            pred_tokens = set(tokenizer.tokenize(text))


            scores = []
            for lab in label_list:
                inter = label2tokenset[lab] & pred_tokens
                overlap = len(inter)
                rare = sum(1 / (token_counter[t] + 1) for t in inter)
                scores.append((lab, overlap, rare))


            scores.sort(key=lambda x: (-x[2], -x[1]))
            ranking = [s[0] for s in scores]  # 全排序
            all_rankings.append(ranking)


            top10_str = ";".join(ranking[:10])


            row_idx = i + j
            row_dict = test_df.iloc[row_idx].to_dict()

            for tgt in y_true_ll[row_idx]:
                pos = ranking.index(tgt) + 1 if tgt in ranking else -1
                target_pos_out.append(
                    {
                        **row_dict,
                        "target_label": tgt,
                        "pos": pos,
                        "top10_pred": top10_str,
                    }
                )


            for k in topk_pred_sets:
                topk_pred_sets[k].append(set(ranking[:k]))


    os.makedirs(os.path.dirname(PREDICTION_FILE), exist_ok=True)
    pd.DataFrame(target_pos_out).to_csv(PREDICTION_FILE, index=False)

    mlb = MultiLabelBinarizer(classes=label_list)
    y_true_bin = mlb.fit_transform(y_true_ll)   # shape = (N, C)

    def hit_rate(k):
        return np.mean(
            [
                len(set(set_true) & set_pred) > 0
                for set_true, set_pred in zip(y_true_ll, topk_pred_sets[k])
            ]
        )

    def f1s(k):

        y_pred_bin = mlb.transform([list(s) for s in topk_pred_sets[k]])
        return {
            "macro": f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0),
            "micro": f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0),
            "weighted": f1_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0),
        }

    metrics = {}
    for k in (3, 5, 10):
        metrics[f"top{k}_hit"] = hit_rate(k)
        metrics[f"top{k}_f1"] = f1s(k)


    for k in (3, 5, 10):
        print(f"Top-{k}  Hit-Rate : {metrics[f'top{k}_hit']:.4f}")
        f1s_k = metrics[f"top{k}_f1"]
        print(f"        Macro-F1   : {f1s_k['macro']:.4f}")
        print(f"        Micro-F1   : {f1s_k['micro']:.4f}")
        print(f"        Weighted-F1: {f1s_k['weighted']:.4f}\n")



if __name__ == "__main__":

    # ① MedAlpaca
    PROMPT_TMPL = (
        "Instruction: You are a clinical coding assistant. "
        "Read the patient report and output the most appropriate ICD-11 diagnosis title.\n"
        "Patient report: {report}\nDiagnosis:"
    )
    evaluate_top(
        BASE_MODEL="medalpaca/medalpaca-7b",
        CACHE_DIR="/project/ywu10/cache_directory",
        PEFT_DIR="model/medalpaca_disease_finetuned",
        CSV_FILE="version2/mimic_diagnosis_icd11(test).csv",
        SPLIT=False,
        PREDICTION_FILE="result/medalpaca(limit_token).csv",
        prompts=PROMPT_TMPL,
    )


    PROMPT_TMPL2 = (
        "Instruction: You are a clinical coding assistant. "
        "Read the patient report and output the most appropriate ICD-11 diagnosis title.\n"
        "Patient report: {report}\nDiagnosis:"
    )
    evaluate_top(
        BASE_MODEL="Henrychur/MMed-Llama-3-8B",
        CACHE_DIR="/project/ywu10/cache_directory",
        PEFT_DIR="model/medllama_disease_finetuned",
        CSV_FILE="version2/mimic_diagnosis_icd11(test).csv",
        SPLIT=False,
        PREDICTION_FILE="result/medllama(limit_token).csv",
        prompts=PROMPT_TMPL2,
    )

    # ③ MedGEMMA —— 你本来就是用的内置 prompt
    evaluate_top(
        BASE_MODEL="google/medgemma-4b-it",
        CACHE_DIR="/project/ywu10/cache_directory",
        PEFT_DIR="model/medgemma_disease_finetuned",
        CSV_FILE="version2/mimic_diagnosis_icd11(test).csv",
        SPLIT=False,
        PREDICTION_FILE="result/medgemma(limit_token).csv",
        prompts=False,
    )

    PROMPT_TMPL = (
        "Instruction: You are a clinical coding assistant. "
        "Read the patient report and output the most appropriate ICD-11 diagnosis title.\n"
        "Patient report: {report}\nDiagnosis:"
    )
    evaluate_top(
        BASE_MODEL="che111/AlphaMed-3B-base-rl",
        CACHE_DIR="/project/ywu10/cache_directory",
        PEFT_DIR="model/alphamed3b_disease_finetuned",
        CSV_FILE="version2/mimic_diagnosis_icd11(test).csv",
        SPLIT=False,
        PREDICTION_FILE="result/alphamed3b(limit_token).csv",
        prompts=False,
    )

    evaluate_top(
        BASE_MODEL="che111/AlphaMed-7B-base-rl",
        CACHE_DIR="/project/ywu10/cache_directory",
        PEFT_DIR="model/alphamed7b_disease_finetuned",
        CSV_FILE="version2/mimic_diagnosis_icd11(test).csv",
        SPLIT=False,
        PREDICTION_FILE="result/alphamed7b(limit_token).csv",
        prompts=False,
    )

    PROMPT_TMPL = (
        "Instruction: You are a clinical coding assistant. "
        "Read the patient report and output the most appropriate ICD-11 diagnosis title.\n"
        "Patient report: {report}\nDiagnosis:"
    )
    evaluate_top(
        BASE_MODEL="medicalai/MedFound-7B",
        CACHE_DIR="/project/ywu10/cache_directory",
        PEFT_DIR="model/medfound7b_disease_finetuned",
        CSV_FILE="version2/mimic_diagnosis_icd11(test).csv",
        SPLIT=False,
        PREDICTION_FILE="result/medfound7b(limit_token).csv",
        prompts=False,
    )

    evaluate_top(
        BASE_MODEL="medicalai/MedFound-Llama3-8B-finetuned",
        CACHE_DIR="/project/ywu10/cache_directory",
        PEFT_DIR="model/medfound8b_disease_finetuned",
        CSV_FILE="version2/mimic_diagnosis_icd11(test).csv",
        SPLIT=False,
        PREDICTION_FILE="result/medfound8b(limit_token).csv",
        prompts=False,
    )


