# 🧠 MIMIC-SR-ICD11: A Dataset for Narrative-Based Diagnosis

**Authors:** Yuexin Wu (University of Memphis), Shiqi Wang (Guangzhou University of Chinese Medicine), Vasile Rus (University of Memphis)
**Conference:** *Findings of ML4H 2025*
**Paper:** MIMIC-SR-ICD11: A Dataset for Narrative-Based Diagnosis

---

## 🌍 Overview

**MIMIC-SR-ICD11** is a large-scale **English diagnostic dataset** that converts de-identified EHR discharge notes from **MIMIC-IV** into **patient-authored self-reports** and standardizes diagnoses with the **WHO ICD-11** ontology. It bridges clinical documentation and real-world patient narratives, enabling diagnostic reasoning on natural-language symptom descriptions rather than categorical checklists.

* Inputs mirror first-contact symptom narratives (patient self-reports).
* Labels are natively aligned to **ICD-11**, reducing ambiguity and post-hoc mapping.
* Designed for **full-text diagnostic inference** and realistic downstream deployment (triage tools, assistants, conversational agents).

---

## 🧱 Dataset Construction

> This repository displays the construction as Figure 2 from the paper. 

![Data construction pipeline (Figure 2)](data_construction_diagram.pdf)

*Figure 2: Left branch maps primary diagnoses from MIMIC-IV (ICD-9 → ICD-10 via CMS GEMs; ICD-10 → ICD-11 via WHO tables) with one-to-one filtering and manual curation. Right branch rewrites MIMIC-IV-Note into first-person self-reports using an instruction-tuned prompt that excludes clinician-only content.*

---

## 🧮 LL-Rank: PMI-Style Re-Ranking

We introduce **LL-Rank**, a likelihood-based re-ranking framework that discounts head-class (frequent label) bias by subtracting a report-free prior from the conditional likelihood.

$$
S(x, c) = -L_{\text{cond}}(x,c) + \alphaL_{\text{prior}}(c), \quad \alpha = 1
$$

* **Conditional term** $L_{\text{cond}}$: per-token NLL of label *c* given report *x* under a fixed prompt.
* **Prior term** $L_{\text{prior}}\$: per-token NLL of *c* under the same prompt **without** the report (report-free prior).
* **Effect**: isolates semantic compatibility from label frequency, improving calibration and long-tail performance.

---

## 📈 Results Highlights

Across seven medical backbones, **LL-Rank** consistently outperforms a generation+mapping baseline (**GenMap**):

* Hit@3/5/10: average gains of **≈ +80% / +86% / +93%
* Macro-F1@3/5/10: average gains of **≈ +138% / +147% / +157%

These gains are especially strong for Macro-F1, indicating improved performance on underrepresented labels—not merely amplifying head classes. Performance peaks around **(\alpha \approx 1)**.

---

## 🧪 Baselines (Summary)

| Category     | Model                                                                              | Backbone                          | Training Objective    |
| ------------ | ---------------------------------------------------------------------------------- | --------------------------------- | --------------------- |
| Medical LLMs | MedAlpaca (7B), MMed-LLaMA (8B), MedGEMMA (3B), AlphaMed (3B/7B), MedFound (7B/8B) | LLaMA v1/3, Gemma-3, Qwen2, BLOOM | SFT                   |                 
| General LLMs | Gemini 2.5 Flash, Claude 4 Sonnet, ChatGPT (o3), ChatGPT (GPT-5)                   | Proprietary                       | Zero-shot evaluation  |

---


```bash
# Clone
git clone https://github.com/woqingdoua/MIMIC-SR-ICD11.git
cd MIMIC-SR-ICD11

# (Optional) Setup virtual environment
# python -m venv .venv && source .venv/bin/activate

# Install project dependencies (example)
pip install -r requirements.txt

# Configure Hugging Face auth securely (choose ONE)
# 1) Local keyring-based login (recommended for laptops)
huggingface-cli login
# 2) Environment variable (recommended for servers/CI)
# export HF_TOKEN=***
```

---

## 📚 Citation

If you use this repository, please cite the paper:

```bibtex
@inproceedings{wu2025mimicsricd11,
  title     = {MIMIC-SR-ICD11: A Dataset for Narrative-Based Diagnosis},
  author    = {Wu, Yuexin and Wang, Shiqi and Rus, Vasile},
  booktitle = {Findings of Machine Learning for Health (ML4H)},
  year      = {2025},
  url       = {https://github.com/woqingdoua/MIMIC-SR-ICD11}
}
```

---

## 🩺 Contact

**Yuexin Wu** · [ywu10@memphis.edu](mailto:ywu10@memphis.edu)
