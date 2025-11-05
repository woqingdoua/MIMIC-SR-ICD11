#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
os.environ.setdefault("HF_HOME", "/project/ywu10/cache_directory")

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============== Config ==============
MM         = 'medllama'   # "medalpaca, medllama, medgemma, alphamed3b, alphamed7b, medfound7b, medfound8b"
PEFT_DIR   = f"model/{MM}_disease_finetuned"
BASE_MODEL = "Henrychur/MMed-Llama-3-8B"
CACHE_DIR  = "/project/ywu10/cache_directory"

COUNT_CSV  = "stat/primary_300_diseases(0708).csv"

USE_BF16   = True
REPORT_BATCH = 1
CAND_CHUNK   = 8
ALPHA_PMI    = 1.0


MAX_SEQ_LEN       = 1024
CSV_FILE          = "version2/mimic_diagnosis_icd11(test).csv"
REPORT_PREFIX_FMT = ("Instruction: You are a clinical coding assistant. "
                     "Read the patient report and output the most appropriate "
                     "ICD-11 diagnosis title. Patient report: {rep}")
SAVE_CSV    = f"ehrResult/{MM}_prediction_report2.csv"
SAVE_PERCLS = f"ehrResult/{MM}_f1_per_class0_report2.csv"


DIAG_PREFIX_TXT = "\nDiagnosis:"
DIAG_SUFFIX_TXT = ""

# ============== Helpers ==============
def enc_report(tok, rep: str):
    return tok(REPORT_PREFIX_FMT.format(rep=rep), add_special_tokens=False).input_ids

def enc_label_only(tok, label: str):
    # leading space to match typical tokenizer expectations for a standalone label
    text = label if label.startswith(" ") else " " + label
    return tok(text, add_special_tokens=False).input_ids

def dynamic_pad_right(tok, ids_list, lbls_list):
    Lmax = max(len(s) for s in ids_list)
    B = len(ids_list)
    input_ids = torch.full((B, Lmax), tok.pad_token_id, dtype=torch.long)
    labels    = torch.full((B, Lmax), -100, dtype=torch.long)
    attn      = torch.zeros((B, Lmax), dtype=torch.long)
    for i, (x, y) in enumerate(zip(ids_list, lbls_list)):
        L = len(x)
        input_ids[i, :L] = torch.tensor(x, dtype=torch.long)
        labels[i, :L]    = torch.tensor(y, dtype=torch.long)
        attn[i, :L]      = 1
    return input_ids, labels, attn

@torch.inference_mode()
def avg_nll_tokenonly_per_sample(model, tok, device, ids_list, lbls_list):
    input_ids, labels, attn = dynamic_pad_right(tok, ids_list, lbls_list)
    input_ids = input_ids.to(device, non_blocking=True)
    labels    = labels.to(device, non_blocking=True)
    attn      = attn.to(device, non_blocking=True)

    logits = model(input_ids=input_ids, attention_mask=attn).logits  # [B, L, V]
    ll = logits[:, :-1, :].contiguous()
    sl = labels[:, 1:].contiguous()

    B, Lm1, V = ll.shape
    valid = (sl != -100)

    if valid.any():
        ll2d = ll.view(B*Lm1, V)
        sl1d = sl.view(B*Lm1)
        m1d  = valid.view(B*Lm1)
        ll_sel = ll2d[m1d]
        sl_sel = sl1d[m1d]
        per_tok_loss = F.cross_entropy(ll_sel.float(), sl_sel, reduction="none")
        # aggregate per sample
        sample_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, Lm1).reshape(-1)[m1d]
        loss_sum = torch.zeros(B, device=device, dtype=per_tok_loss.dtype)
        cnt_sum  = torch.zeros(B, device=device, dtype=torch.float32)
        loss_sum.scatter_add_(0, sample_idx, per_tok_loss)
        cnt_sum.scatter_add_(0, sample_idx, torch.ones_like(per_tok_loss, dtype=torch.float32))
        avg = (loss_sum / cnt_sum.clamp_min(1.0)).detach().cpu().numpy()
    else:
        avg = np.zeros(B, dtype=np.float32)

    # cleanup
    del input_ids, labels, attn, logits, ll, sl, valid
    torch.cuda.empty_cache()
    return avg

# ====== F1 utilities ======
def f1_top1(y_true_idx, y_pred_top1_idx, n_classes):
    """Standard multi-class Top-1 F1 (macro/micro/weighted)."""
    macro  = f1_score(y_true_idx, y_pred_top1_idx, average="macro", labels=list(range(n_classes)), zero_division=0)
    micro  = f1_score(y_true_idx, y_pred_top1_idx, average="micro", labels=list(range(n_classes)), zero_division=0)
    wtd    = f1_score(y_true_idx, y_pred_top1_idx, average="weighted", labels=list(range(n_classes)), zero_division=0)
    return macro, micro, wtd

def f1_topk_multilabel(y_true_idx, rankings_idx, k, n_classes):
    """
    Multi-label F1@k: predict a k-hot vector over classes; gold is 1-hot (or multi if dataset has multiple golds).
    This penalizes false positives introduced by top-k predictions.
    """
    N = len(rankings_idx)
    Y_true = np.zeros((N, n_classes), dtype=int)
    Y_pred = np.zeros((N, n_classes), dtype=int)
    for i in range(N):
        for g in y_true_idx[i]:   # allow multiple golds if present
            Y_true[i, g] = 1
        for p in rankings_idx[i][:k]:
            Y_pred[i, p] = 1
    macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    wtd   = f1_score(Y_true, Y_pred, average="weighted", zero_division=0)
    return macro, micro, wtd

# ============== Main ==============
def main():

    if os.path.exists(os.path.join(PEFT_DIR, "tokenizer.json")):
        tok = AutoTokenizer.from_pretrained(PEFT_DIR)
    else:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=os.environ.get("HF_TOKEN", None))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    diag_prefix_ids = tok(DIAG_PREFIX_TXT, add_special_tokens=False).input_ids
    diag_suffix_ids = tok(DIAG_SUFFIX_TXT, add_special_tokens=False).input_ids + [tok.eos_token_id]

    # ---- Base model ----
    try:
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = torch.bfloat16 if USE_BF16 else torch.float16

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": 0}, torch_dtype=dtype, cache_dir=CACHE_DIR
    )

    base.resize_token_embeddings(len(tok))
    if hasattr(base.config, "_attn_implementation"):
        base.config._attn_implementation = "eager"
    base.config.use_cache = False

    # ---- Attach LoRA ----
    model = PeftModel.from_pretrained(base, PEFT_DIR)
    model.eval()
    device = model.device

    _ = model(input_ids=torch.tensor([[tok.eos_token_id]], device=device),
              attention_mask=torch.tensor([[1]], device=device)).logits

    # ---- Candidates (labels) ----
    label_list = (
        pd.read_csv(COUNT_CSV)["icd11_title"]
        .astype(str).str.strip().str.lower().tolist()
    )
    label_set = set(label_list)
    M = len(label_list)
    label2idx = {lab: i for i, lab in enumerate(label_list)}
    label_tok_ids = [enc_label_only(tok, lab) for lab in label_list]

    # ===== Prior: avgNLL(label | DIAG_PREFIX only) =====
    prior_avg_nll = np.zeros(M, dtype=np.float32)
    for st in tqdm(range(0, M, CAND_CHUNK), desc="Prior (unconditional)"):
        ed = min(st + CAND_CHUNK, M)
        ids_batch, lbls_batch = [], []
        for j in range(st, ed):
            lab = label_tok_ids[j]
            ids = diag_prefix_ids + lab + diag_suffix_ids
            lbl = [-100]*len(diag_prefix_ids) + lab + [-100]*len(diag_suffix_ids)
            if len(ids) > MAX_SEQ_LEN:
                ids = ids[-MAX_SEQ_LEN:]; lbl = lbl[-MAX_SEQ_LEN:]
            ids_batch.append(ids); lbls_batch.append(lbl)
        prior_avg_nll[st:ed] = avg_nll_tokenonly_per_sample(model, tok, device, ids_batch, lbls_batch).astype(np.float32)

    # ===== Test set =====
    df = pd.read_csv(CSV_FILE)
    reports = df["report"].astype(str).tolist()

    # gold may contain multiple labels separated by ";"
    y_true_all = []
    for s in df["icd11_title"].astype(str).tolist():
        parts = [p.strip().lower() for p in s.split(";") if p.strip()]
        y_true_all.append([p for p in parts if p in label_set])

    N = len(reports)
    subject_ids = df["subject_id"].tolist()
    hadm_ids = df["hadm_id"].tolist()
    cond_avg_nll = np.zeros((N, M), dtype=np.float32)

    # ===== Conditional =====
    for i0 in tqdm(range(0, N, REPORT_BATCH), desc="Conditional (reports)"):
        i1 = min(i0 + REPORT_BATCH, N)
        rep_ids_list = [enc_report(tok, r) for r in reports[i0:i1]]

        for st in range(0, M, CAND_CHUNK):
            ed = min(st + CAND_CHUNK, M)
            ids_batch, lbls_batch, dst_idx = [], [], []
            for bi, rep_ids in enumerate(rep_ids_list):
                for j in range(st, ed):
                    lab = label_tok_ids[j]
                    ids = rep_ids + diag_prefix_ids + lab + diag_suffix_ids
                    lbl = ([-100]*len(rep_ids) +
                           [-100]*len(diag_prefix_ids) +
                           lab +
                           [-100]*len(diag_suffix_ids))
                    if len(ids) > MAX_SEQ_LEN:
                        ids = ids[-MAX_SEQ_LEN:]; lbl = lbl[-MAX_SEQ_LEN:]
                    ids_batch.append(ids); lbls_batch.append(lbl); dst_idx.append((i0 + bi, j))

            per_avg = avg_nll_tokenonly_per_sample(model, tok, device, ids_batch, lbls_batch)
            for k, (ri, lj) in enumerate(dst_idx):
                cond_avg_nll[ri, lj] = per_avg[k]

    # ===== PMI score & ranking =====
    scores  = -cond_avg_nll + (ALPHA_PMI * prior_avg_nll[None, :])
    order   = np.argsort(-scores, axis=1)                  # [N, M]
    rankings_idx = order.tolist()                           # int indices
    rankings_str = [[label_list[j] for j in row] for row in order]

    # ===== Eval sets =====
    valid_idx = [i for i, ys in enumerate(y_true_all) if len(ys) > 0]
    N_valid = len(valid_idx)

    def hit_rate(k: int) -> float:
        hits = []
        for i in valid_idx:
            golds = set(y_true_all[i])
            preds = set(rankings_str[i][:k])
            hits.append(len(golds & preds) > 0)
        return float(np.mean(hits)) if hits else 0.0

    # ===== F1 computations =====
    y_true_idx_list = []
    for i in valid_idx:
        y_true_idx_list.append([label2idx[g] for g in y_true_all[i]])

    y_pred_top1_idx = [rankings_idx[i][0] for i in valid_idx]
    y_true_top1_idx = [ys[0] for ys in y_true_idx_list]

    macro1, micro1, wtd1 = f1_top1(y_true_top1_idx, y_pred_top1_idx, M)
    macro3, micro3, wtd3 = f1_topk_multilabel(y_true_idx_list, rankings_idx, k=3, n_classes=M)
    macro5, micro5, wtd5 = f1_topk_multilabel(y_true_idx_list, rankings_idx, k=5, n_classes=M)
    macro10, micro10, wtd10 = f1_topk_multilabel(y_true_idx_list, rankings_idx, k=10, n_classes=M)

    # ===== Prints =====
    print("\n===== PMI likelihood scoring (token-only; bf16; single-GPU) =====")
    print(f"Alpha (PMI)     : {ALPHA_PMI}")
    print(f"#reports        : {N} (valid for eval: {N_valid})")
    for k in (3, 5, 10):
        print(f"Top-{k} Hit-Rate: {hit_rate(k):.4f}")

    print("\n===== F1 scores =====")
    print(f"Top-1  Macro-F1: {macro1:.4f} | Micro-F1: {micro1:.4f} | Weighted-F1: {wtd1:.4f}")
    print(f"Top-3  Macro-F1: {macro3:.4f} | Micro-F1: {micro3:.4f} | Weighted-F1: {wtd3:.4f}")
    print(f"Top-5  Macro-F1: {macro5:.4f} | Micro-F1: {micro5:.4f} | Weighted-F1: {wtd5:.4f}")
    print(f"Top-10 Macro-F1: {macro10:.4f} | Micro-F1: {micro10:.4f}")


    if SAVE_PERCLS:
        y_true_arr = np.array(y_true_top1_idx, dtype=int)
        y_pred_arr = np.array(y_pred_top1_idx, dtype=int)
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true_arr, y_pred_arr, labels=list(range(M)), average=None, zero_division=0
        )
        per_cls = pd.DataFrame({
            "label": label_list,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": sup
        })
        os.makedirs(os.path.dirname(SAVE_PERCLS) or ".", exist_ok=True)
        per_cls.to_csv(SAVE_PERCLS, index=False)
        print(f"✅ Saved per-class Top-1 PRF to {SAVE_PERCLS}")

    if SAVE_CSV:
        os.makedirs(os.path.dirname(SAVE_CSV) or ".", exist_ok=True)
        rows = [
            {
                "idx": i,
                "subject_id": subject_ids[i],
                "hadm_id": hadm_ids[i],
                "truth": ";".join(y_true_all[i]),
                "top10": ";".join(rankings_str[i][:10]),
            }
            for i in range(N)
        ]
        pd.DataFrame(rows).to_csv(SAVE_CSV, index=False)
        print(f"✅ Saved rankings to {SAVE_CSV}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        main()
    except RuntimeError:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        raise
