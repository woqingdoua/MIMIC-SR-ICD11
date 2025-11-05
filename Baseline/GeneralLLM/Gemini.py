import os
import pandas as pd
import json
import time
import multiprocessing
from tqdm import tqdm
import google.generativeai as genai
import random

# ========== CONFIG ==========
api_key = ""
genai.configure(api_key=api_key)


# Prompt and candidates
PROMPT_PREFIX = """You are an expert clinical decision-support system.

TASK
1. Read the patient’s narrative carefully.
2. Focus on identifying the **primary diagnosis**—the main disease or condition that explains the patient's **chief complaint or most critical symptoms**.
3. From the CANDIDATE_LIST, select the **10 most likely primary diagnoses** (ignoring secondary or comorbid conditions).
4. Rank them from most likely → least likely.

OUTPUT FORMAT
Return a single line containing the 10 diagnosis names, separated only by semicolons.

CONSTRAINTS
• Only choose from the CANDIDATE_LIST.
• Do not include explanations, probabilities, or extra text—only the semicolon-delimited list.

CANDIDATE_LIST:
"""
# ⚠️ Replace this placeholder with your full candidate_list
candidate_list = ['congestive heart failure', 'hepatic encephalopathy', 'deep bacterial folliculitis or pyogenic abscess of the skin', 'cholelithiasis', 'acute pancreatitis', 'encephalopathy due to toxicity', 'traumatic subdural haemorrhage', 'incisional hernia', 'pneumonia', 'malignant neoplasm metastasis in bone or bone marrow', 'alcohol withdrawal', 'chronic kidney disease', 'calculus of kidney', 'traumatic subdural hemorrhage', 'hyperkalaemia', 'pleural effusion', 'alcoholic liver disease', 'acute nonst elevation myocardial infarction', 'epilepsy or seizures, unspecified', 'obesity', 'osteoarthritis of hip', 'fracture of femur', 'diarrhoea', 'human immunodeficiency virus disease without mention of tuberculosis or malaria', 'gastrointestinal bleeding', 'intestinal infections due to clostridioides difficile', 'hypertensive heart disease', 'multiple valve disease', 'pathological fracture', 'chronic obstructive pulmonary disease', 'malignant neoplasms of prostate', 'type 1 diabetes mellitus', 'disorders due to use of alcohol', 'recurrent depressive disorder', 'cerebral aneurysm, nonruptured', 'left ventricular failure with reduced ejection fraction', 'fracture of lumbar vertebra', 'chronic pancreatitis', 'obstruction of large intestine', 'abdominal aortic aneurysm', 'sepsis without septic shock', 'acute pyelonephritis', 'left ventricular failure with preserved ejection fraction', 'acute respiratory failure', 'peritonitis', 'pneumonitis due to solids or liquids', 'malignant neoplasm of pancreas', 'leiomyoma of uterus', 'complete atrioventricular block', 'malignant neoplasm metastasis in spinal cord, cranial nerves or paraspinal nerves', 'pulmonary thromboembolism', 'viral intestinal infections, unspecified', 'cerebral ischaemic stroke', 'asymptomatic stenosis of intracranial or extracranial artery', 'intervertebral disc degeneration', 'atrial fibrillation', 'osteoarthritis of knee', 'gastroenteritis or colitis without specification of infectious agent', 'ventricular tachycardia', 'constipation', 'depressive disorders, unspecified', 'alcoholic cirrhosis of liver without hepatitis', 'hypotension, unspecified', 'neutropaenia', 'orthostatic hypotension', 'hepatic fibrosis or cirrhosis', 'gastrooesophageal reflux disease', 'calculus of bile duct without cholangitis or cholecystitis', 'acute posthaemorrhagic anaemia', 'fracture of neck', 'diverticulitis of large intestine', 'hypo-osmolality or hyponatraemia', 'type 2 diabetes mellitus', 'urinary tract infection, site not specified', 'functional nausea or vomiting', 'asthma', 'primary neoplasms of meninges', 'acute appendicitis', 'schizophrenia or other primary psychotic disorders, unspecified', 'coronary atherosclerosis', 'malignant neoplasms of kidney, except renal pelvis', 'aortic valve stenosis', 'obstruction of bile duct', 'multiple sclerosis', 'bacterial cellulitis, erysipelas or lymphangitis', 'degenerative condition of spine, unspecified', 'bacteraemia', 'macro reentrant atrial tachycardia', 'essential hypertension', 'intracerebral haemorrhage', 'anaemias or other erythrocyte disorders, unspecified', 'malignant neoplasm metastasis in brain', 'bacterial pneumonia', 'single episode depressive disorder', 'schizoaffective disorder', 'other specified cardiac arrhythmia', 'acute upper respiratory infections of unspecified site', 'influenza, virus not identified', 'malignant neoplasms of thyroid gland', 'hypertensive renal disease', 'cerebral ischaemic stroke due to embolic occlusion', 'diverticulosis of large intestine', 'dehydration', 'acute kidney failure', 'malignant neoplasms of bronchus or lung', 'venous thromboembolism', 'malignant neoplasm metastasis in retroperitoneum', 'iron deficiency anaemia', 'acute cholecystitis', 'transient ischaemic attack', 'subarachnoid haemorrhage', 'calculus of ureter', 'other specified mood disorders', 'crohn disease', 'hyperplasia of prostate', 'acute myocardial infarction', 'malignant neoplasm of liver', 'cholangitis']

df = pd.read_csv("../sample_data/sample_data5.csv")
self_reports = df["report"].tolist()
ground_truth = df["icd11_title"].tolist()
#candidate_lists = df["candidate_list"].tolist()


# ========== FUNCTION TO PROCESS ONE REPORT ==========
def process_single(index):
    #candidate_list = ['rheumatoid arthritis', 'urinary tract infection, site not specified', 'personal history of colonic polyps', 'unspecified sepsis', 'cerebral infarct', 'septicaemia', 'end stage kidney disease', 'disorders of plasma-protein metabolism, not elsewhere classified', 'backache nos', 'hyposmolality', 'aortocoronary bypass status', 'hypopotassaemia', 'opioid abuse', 'cannabis abuse', 'asthmatic', 'chronic kidney disease', 'gastrointestinal tract haemorrhage', 'anaemia nos', 'personal history of malignant neoplasm', 'irritable bowel syndrome', 'disorder of bone or cartilage, unspecified', 'metabolic encephalopathy', 'alcoholic liver cirrhosis', 'chronic obstructive asthma', 'benign prostatic hyperplasia', 'urinary tract infection nos', 'protein- calorie malnutrition unspecified', 'intermediate coronary syndrome', 'acquired absence of kidney', 'secondary malignant neoplasm of retroperitoneum', 'hypocalcaemia nos', 'atrial flutter nos', 'personal history of irradiation', 'personal history of peptic ulcer disease', 'nstemi - [non st elevation myocardial infarction]', 'oesophageal reflux nos', 'left bundle branch block', 'secondary cancer of bone', 'supraventricular tachycardia', 'hydronephrosis nos', 'pure hypercholesterolaemia', 'cardiac pacemaker in situ', 'morbid obesity', 'deep vein thrombosis nos', 'paralysis agitans', 'chronic obstructed airway, unspecified', 'hyperlipidaemia', 'abnormality of gait', 'tobacco use disorder', 'leukocytosis', 'cardiac failure nos', 'bicycle accident nos', 'long term current use of anticoagulants', 'bipolar disorder nos', 'obstructive sleep apnoea, adult', 'peripheral venous insufficiency', 'other secondary pulmonary hypertension', 'hypertensive heart disease with heart failure', 'encephalopathy nos', 'oesophageal varices without bleeding in diseases classified elsewhere', 'secondary malignant neoplasm of bone or bone marrow', 'peripheral vascular disease nos', 'secondary and unspecified malignant neoplasm, lymph node, unspecified', 'solitary pulmonary nodule', 'face - [facial afro-caribbean childhood eruption]', 'chronic diastolic congestive heart failure', 'secondary malignant neoplasm of brain', 'diaphragmatic hernia with obstruction, without gangrene', 'acute kidney failure', 'atelectasis', 'abdominal pain nos', 'malignant neoplasm of liver, specified as secondary', 'attention deficit hyperactivity disorder', 'severe protein calorie malnutrition', 'unspecified intestinal obstruction', 'motor vehicle accident nos', 'dorsalgia', 'alcohol abuse', 'chronic iron deficiency anaemia secondary to blood loss', 'disorders of phosphorus metabolism', 'lumbago nos', 'secondary cancer of lung', 'cocaine abuse', 'acquired absence of large intestine', 'weight loss nos', 'respiratory failure, unspecified as acute or chronic', 'personal history of tobacco use', 'headache, unspecified', 'hypoxaemia', 'unspecified background retinopathy', 'renal dialysis status', 'adult failure to thrive', 'major depressive disorder, single episode', 'chronic kidney disease [type 2 diabetes mellitus]', 'fever', 'paroxysmal ventricular tachycardia', 'septic shock nos', 'enterocolitis due to clostridium difficile']
    model = genai.GenerativeModel("gemini-2.5-flash")
    candidate_list = ['congestive heart failure', 'hepatic encephalopathy', 'deep bacterial folliculitis or pyogenic abscess of the skin', 'cholelithiasis', 'acute pancreatitis', 'encephalopathy due to toxicity', 'traumatic subdural haemorrhage', 'incisional hernia', 'pneumonia', 'malignant neoplasm metastasis in bone or bone marrow', 'alcohol withdrawal', 'chronic kidney disease', 'calculus of kidney', 'traumatic subdural hemorrhage', 'hyperkalaemia', 'pleural effusion', 'alcoholic liver disease', 'acute nonst elevation myocardial infarction', 'epilepsy or seizures, unspecified', 'obesity', 'osteoarthritis of hip', 'fracture of femur', 'diarrhoea', 'human immunodeficiency virus disease without mention of tuberculosis or malaria', 'gastrointestinal bleeding', 'intestinal infections due to clostridioides difficile', 'hypertensive heart disease', 'multiple valve disease', 'pathological fracture', 'chronic obstructive pulmonary disease', 'malignant neoplasms of prostate', 'type 1 diabetes mellitus', 'disorders due to use of alcohol', 'recurrent depressive disorder', 'cerebral aneurysm, nonruptured', 'left ventricular failure with reduced ejection fraction', 'fracture of lumbar vertebra', 'chronic pancreatitis', 'obstruction of large intestine', 'abdominal aortic aneurysm', 'sepsis without septic shock', 'acute pyelonephritis', 'left ventricular failure with preserved ejection fraction', 'acute respiratory failure', 'peritonitis', 'pneumonitis due to solids or liquids', 'malignant neoplasm of pancreas', 'leiomyoma of uterus', 'complete atrioventricular block', 'malignant neoplasm metastasis in spinal cord, cranial nerves or paraspinal nerves', 'pulmonary thromboembolism', 'viral intestinal infections, unspecified', 'cerebral ischaemic stroke', 'asymptomatic stenosis of intracranial or extracranial artery', 'intervertebral disc degeneration', 'atrial fibrillation', 'osteoarthritis of knee', 'gastroenteritis or colitis without specification of infectious agent', 'ventricular tachycardia', 'constipation', 'depressive disorders, unspecified', 'alcoholic cirrhosis of liver without hepatitis', 'hypotension, unspecified', 'neutropaenia', 'orthostatic hypotension', 'hepatic fibrosis or cirrhosis', 'gastrooesophageal reflux disease', 'calculus of bile duct without cholangitis or cholecystitis', 'acute posthaemorrhagic anaemia', 'fracture of neck', 'diverticulitis of large intestine', 'hypo-osmolality or hyponatraemia', 'type 2 diabetes mellitus', 'urinary tract infection, site not specified', 'functional nausea or vomiting', 'asthma', 'primary neoplasms of meninges', 'acute appendicitis', 'schizophrenia or other primary psychotic disorders, unspecified', 'coronary atherosclerosis', 'malignant neoplasms of kidney, except renal pelvis', 'aortic valve stenosis', 'obstruction of bile duct', 'multiple sclerosis', 'bacterial cellulitis, erysipelas or lymphangitis', 'degenerative condition of spine, unspecified', 'bacteraemia', 'macro reentrant atrial tachycardia', 'essential hypertension', 'intracerebral haemorrhage', 'anaemias or other erythrocyte disorders, unspecified', 'malignant neoplasm metastasis in brain', 'bacterial pneumonia', 'single episode depressive disorder', 'schizoaffective disorder', 'other specified cardiac arrhythmia', 'acute upper respiratory infections of unspecified site', 'influenza, virus not identified', 'malignant neoplasms of thyroid gland', 'hypertensive renal disease', 'cerebral ischaemic stroke due to embolic occlusion', 'diverticulosis of large intestine', 'dehydration', 'acute kidney failure', 'malignant neoplasms of bronchus or lung', 'venous thromboembolism', 'malignant neoplasm metastasis in retroperitoneum', 'iron deficiency anaemia', 'acute cholecystitis', 'transient ischaemic attack', 'subarachnoid haemorrhage', 'calculus of ureter', 'other specified mood disorders', 'crohn disease', 'hyperplasia of prostate', 'acute myocardial infarction', 'malignant neoplasm of liver', 'cholangitis']


    try:
        report = self_reports[index]
        #candidate_list = candidate_lists[index].split(";")
        #candidate_list = [i.strip() for i in candidate_list]
        #random.shuffle(candidate_list)

        PROMPT = PROMPT_PREFIX + str(candidate_list) + ". PATIENT_NOTE: "
        prompt = PROMPT + report
        response =  model.generate_content(prompt)
        result = response.text if response.text else ""
        return (index, result)
    except Exception as e:
        return (index, f"Error: {str(e)}")


# ========== MULTIPROCESSING ==========
if __name__ == "__main__":
    num_processes = min(4, multiprocessing.cpu_count())  # Adjust based on your system
    with multiprocessing.Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(process_single, range(len(self_reports))), total=len(self_reports)))

    # Sort results by index
    results.sort(key=lambda x: x[0])
    predictions = [r[1] for r in results]

    # Save to CSV
    output_df = pd.DataFrame({
        "self_report": self_reports,
        "prediction": predictions,
        "true_mapping": ground_truth
    })
    output_df.to_csv("../result/gemini_prediction_primary.csv", index=False)
