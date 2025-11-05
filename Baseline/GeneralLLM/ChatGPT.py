import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import random
from multiprocessing import Pool, cpu_count
import httpx  # 新增

# Constants
API_KEY = ""
proxy_url = os.environ.get("HTTPS_PROXY", None)  # e.g., "http://127.0.0.1:7890"
http_client = httpx.Client(proxies=proxy_url) if proxy_url else httpx.Client()
client = OpenAI(api_key=API_KEY, http_client=http_client)

MODEL = "gpt-5"

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

CANDIDATE_LIST: ['congestive heart failure', 'hepatic encephalopathy', ....]. PATIENT_NOTE: ....."
"""

candidate_list = ['congestive heart failure', 'hepatic encephalopathy', 'deep bacterial folliculitis or pyogenic abscess of the skin', 'cholelithiasis', 'acute pancreatitis', 'encephalopathy due to toxicity', 'traumatic subdural haemorrhage', 'incisional hernia', 'pneumonia', 'malignant neoplasm metastasis in bone or bone marrow', 'alcohol withdrawal', 'chronic kidney disease', 'calculus of kidney', 'traumatic subdural hemorrhage', 'hyperkalaemia', 'pleural effusion', 'alcoholic liver disease', 'acute nonst elevation myocardial infarction', 'epilepsy or seizures, unspecified', 'obesity', 'osteoarthritis of hip', 'fracture of femur', 'diarrhoea', 'human immunodeficiency virus disease without mention of tuberculosis or malaria', 'gastrointestinal bleeding', 'intestinal infections due to clostridioides difficile', 'hypertensive heart disease', 'multiple valve disease', 'pathological fracture', 'chronic obstructive pulmonary disease', 'malignant neoplasms of prostate', 'type 1 diabetes mellitus', 'disorders due to use of alcohol', 'recurrent depressive disorder', 'cerebral aneurysm, nonruptured', 'left ventricular failure with reduced ejection fraction', 'fracture of lumbar vertebra', 'chronic pancreatitis', 'obstruction of large intestine', 'abdominal aortic aneurysm', 'sepsis without septic shock', 'acute pyelonephritis', 'left ventricular failure with preserved ejection fraction', 'acute respiratory failure', 'peritonitis', 'pneumonitis due to solids or liquids', 'malignant neoplasm of pancreas', 'leiomyoma of uterus', 'complete atrioventricular block', 'malignant neoplasm metastasis in spinal cord, cranial nerves or paraspinal nerves', 'pulmonary thromboembolism', 'viral intestinal infections, unspecified', 'cerebral ischaemic stroke', 'asymptomatic stenosis of intracranial or extracranial artery', 'intervertebral disc degeneration', 'atrial fibrillation', 'osteoarthritis of knee', 'gastroenteritis or colitis without specification of infectious agent', 'ventricular tachycardia', 'constipation', 'depressive disorders, unspecified', 'alcoholic cirrhosis of liver without hepatitis', 'hypotension, unspecified', 'neutropaenia', 'orthostatic hypotension', 'hepatic fibrosis or cirrhosis', 'gastrooesophageal reflux disease', 'calculus of bile duct without cholangitis or cholecystitis', 'acute posthaemorrhagic anaemia', 'fracture of neck', 'diverticulitis of large intestine', 'hypo-osmolality or hyponatraemia', 'type 2 diabetes mellitus', 'urinary tract infection, site not specified', 'functional nausea or vomiting', 'asthma', 'primary neoplasms of meninges', 'acute appendicitis', 'schizophrenia or other primary psychotic disorders, unspecified', 'coronary atherosclerosis', 'malignant neoplasms of kidney, except renal pelvis', 'aortic valve stenosis', 'obstruction of bile duct', 'multiple sclerosis', 'bacterial cellulitis, erysipelas or lymphangitis', 'degenerative condition of spine, unspecified', 'bacteraemia', 'macro reentrant atrial tachycardia', 'essential hypertension', 'intracerebral haemorrhage', 'anaemias or other erythrocyte disorders, unspecified', 'malignant neoplasm metastasis in brain', 'bacterial pneumonia', 'single episode depressive disorder', 'schizoaffective disorder', 'other specified cardiac arrhythmia', 'acute upper respiratory infections of unspecified site', 'influenza, virus not identified', 'malignant neoplasms of thyroid gland', 'hypertensive renal disease', 'cerebral ischaemic stroke due to embolic occlusion', 'diverticulosis of large intestine', 'dehydration', 'acute kidney failure', 'malignant neoplasms of bronchus or lung', 'venous thromboembolism', 'malignant neoplasm metastasis in retroperitoneum', 'iron deficiency anaemia', 'acute cholecystitis', 'transient ischaemic attack', 'subarachnoid haemorrhage', 'calculus of ureter', 'other specified mood disorders', 'crohn disease', 'hyperplasia of prostate', 'acute myocardial infarction', 'malignant neoplasm of liver', 'cholangitis']

#candidate_list = pd.read_csv("unique_disease_2000.csv")["mapping"].tolist()


#PROMPT = PROMPT_PREFIX + candidate_list + ". patient's self-report: "

# Load data
#df = pd.read_csv('sample_data/sampled_data2(right_answer_order).csv')
#data = df[["self_report", "mapping", "corr_later"]].values.tolist()



# Set up OpenAI client


def query_model(entry):
    #self_report, true_mapping, candidate_list = entry
    #candidate_list = candidate_list.split(";")
    self_report, true_mapping = entry
    #candidate_list = [i.strip() for i in candidate_list]
    #random.shuffle(candidate_list)
    PROMPT = PROMPT_PREFIX + str(candidate_list) + ". PATIENT_NOTE: "
    try:

        response = client.chat.completions.create(
                messages=[{"role": "user", "content": PROMPT + self_report}],
            model=MODEL)
        prediction = response.choices[0].message.content
    except Exception as e:
        prediction = f"ERROR: {e}"

    return {
        "self_report": self_report,
        "true_mapping": true_mapping,
        "prediction": prediction
    }

def run_parallel_inference(data, num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(query_model, data), total=len(data)))

    return pd.DataFrame(results)

if __name__ == '__main__':
    os.makedirs("result", exist_ok=True)
    df = pd.read_csv('../sample_data/sample_data5.csv')
    data = df[["report", "icd11_title"]].values.tolist()
    result_df = run_parallel_inference(data)
    result_df.to_csv("../result/chatgpt5_prediction_primary.csv", index=False)
    print("Prediction completed and saved.")

