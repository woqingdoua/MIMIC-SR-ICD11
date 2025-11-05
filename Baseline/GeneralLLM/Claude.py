import anthropic
import pandas as pd
from tqdm import tqdm
import time
import random
# Claude API Key
api_key = ""
client = anthropic.Anthropic(api_key=api_key)

# Load data
df = pd.read_csv('../sample_data/sample_data5.csv')
self_reports = df["report"].tolist()
ground_truth = df["icd11_title"].tolist()
#candidate_lists = df["candidate_list"].tolist()

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

#candidate_list = pd.read_csv("unique_disease_2000.csv")["mapping"].tolist()

def extract_text(message_content):
    try:
        return message_content[0].text.strip()
    except Exception:
        return str(message_content)

def call_claude(prompt):

    for attempt in range(3):
        try:
            message = client.messages.create(
                model="claude-4-sonnet-20250514",
                max_tokens=1000,
                temperature=1,
                system="You are a medical diagnosis assistant.",
                messages=[{
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }]
            )
            return extract_text(message.content)
        except anthropic.RateLimitError:
            print("⚠️ Rate limit hit. Sleeping 60 seconds.")
            time.sleep(60)
        except Exception as e:
            print(f"[Error] {e}")
            return f"Error: {str(e)}"
    return "Error: Failed after retries"

# Run inference sequentially
results = []
for i in tqdm(range(len(self_reports))):
    report = self_reports[i]
    #random.shuffle(candidate_list)
    #candidate_list = candidate_lists[i]
    #candidate_list = candidate_list.split(";")
    #candidate_list = [i.strip() for i in candidate_list]
    prompt = PROMPT_PREFIX + str(candidate_list) + ". PATIENT_NOTE: " + report
    response = call_claude(prompt)
    results.append((i, response))

# Extract results and save
results.sort(key=lambda x: x[0])
predictions = [r[1] for r in results]

output_df = pd.DataFrame({
    "self_report": self_reports,
    "prediction": predictions,
    "true_mapping": ground_truth
})
output_df.to_csv("../result/claude_prediction_primary.csv", index=False)

