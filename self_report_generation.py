import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants
API_KEY = ""

proxy_url = os.environ.get("HTTPS_PROXY", None)  # e.g., "http://127.0.0.1:7890"

# Load data
df = pd.read_csv("mimic_new/mapping_pathology_blank_self_report_500.csv")
os.makedirs("self_report3", exist_ok=True)

# Prompt
prompt = "Based on a patient's hospital record, write a possible self-report describing only their symptoms, excluding examination results. Use complete sentences and avoid bullet points. Write from the patient's perspective using 'I' statements. Do not include other instruction."

def generate_self_report(row):
    idx, content, sid, hid = row
    try:
        http_client = httpx.Client(proxies=proxy_url) if proxy_url else httpx.Client()
        client = OpenAI(api_key=API_KEY, http_client=http_client)

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt + " hospital record: " + content,
                }
            ],
            model="gpt-4o-mini",
        )
        generated_text = chat_completion.choices[0].message.content
    except Exception as e:
        generated_text = ""
        print(f"Error at idx {idx}: {e}")
    return {"subject_id": sid, "hadm_id": hid, "self_report": generated_text}


data_rows = list(zip(
    range(len(df)),
    df['report'].tolist(),      
    df['subject_id'].tolist(),
    df['hadm_id'].tolist()
))


batch_size = 500
results = []
batch_counter = 0
file_counter = 89500

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(generate_self_report, row) for row in data_rows]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Generating self reports"):
        result = future.result()
        results.append(result)
        batch_counter += 1

        if batch_counter % batch_size == 0:
            batch_df = pd.DataFrame(results)
            batch_df[["subject_id", "hadm_id", "self_report"]].to_csv(f"self_report3/{file_counter * batch_size}.csv", index=False)
            results = []  
            file_counter += 1


if results:
    batch_df = pd.DataFrame(results)
    batch_df[["subject_id", "hadm_id", "self_report"]].to_csv(f"self_report3/{file_counter * batch_size}.csv", index=False)

print("✅ All finished and saved!")
