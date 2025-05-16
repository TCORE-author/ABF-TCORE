import json
import time
from tqdm import tqdm
from openai import OpenAI

API_KEY = "your_openai_api_key"
DATA_PATH = "TeleQnA.txt"
OUTPUT_PATH = "TCORE.json"
MODEL_NAME = "o4-mini"
MAX_TOKENS = 1024

client = OpenAI(api_key=API_KEY)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

entries = [
    {
        "question_id": qid,
        "question": entry["question"],
        "category": entry["category"],
        "options": [entry[f"option {i}"] for i in range(6) if f"option {i}" in entry],
        "answer": entry["answer"]
    }
    for qid, entry in dataset.items()
]

results = []
for item in tqdm(entries, desc="Processing 10K questions"):
    question_text = item["question"]
    options = item["options"]

    formatted_options = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    prompt = f"""You are a telecom reasoning assistant. Read the question and reason step-by-step to find the correct answer.

Question: {question_text}
Options:
{formatted_options}

Think aloud before choosing the final answer."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=MAX_TOKENS
        )
        trajectory = response.choices[0].message.content

        results.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "category": item["category"],
            "options": item["options"],
            "answer": item["answer"],
            "thinking_trajectory": trajectory
        })

    except Exception as e:
        print(f"[ERROR] {item['question_id']} - {e}")
        continue


with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
