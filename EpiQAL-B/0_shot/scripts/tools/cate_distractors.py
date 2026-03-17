import json
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import random
import re
from collections import Counter


OPENAI_API_KEY = ""
FINAL_QA = "../../output/final_qa.json"
OUTPUT = "../../output/distractor_category.json"
BATCH_SIZE = 16


client = OpenAI(api_key=OPENAI_API_KEY)

with open(FINAL_QA) as f:
    final_qa = json.load(f)

def classify_distractor(client, paragraph, question, correct_options, distractor):
    prompt = f"""You are an expert in epidemiology and question design.

                This is a multi-step reasoning question where the correct answer requires combining passage evidence with epidemiological principles. The distractor contains a subtle logical flaw that can only be detected through careful reasoning.

                Your task: classify the TYPE OF REASONING ERROR in this distractor.

                Passage:
                {paragraph}

                Question: {question}

                Correct answer(s): {json.dumps(correct_options)}

                Distractor: {distractor}

                Classify into exactly ONE category:
                1. Methodological mismatch: Uses an analysis method or study design incompatible with the scenario
                2. Causal/logical error: Reverses cause-effect, confuses necessary vs sufficient conditions, or breaks the reasoning chain
                3. Variable confusion: Confuses the target variable with a related but distinct variable or concept
                4. Assumption violation: Uses correct terminology but violates an underlying methodological or statistical assumption

                Output ONLY a JSON object:
                {{"category": "one of the 4 categories above", "brief_reason": "one sentence explanation"}}"""

    for i in range(3):
        try:
            time.sleep(random.uniform(0.1, 0.3))
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
                user="epiqal_academic_research"
            )
            text = response.choices[0].message.content.strip()
            json_match = re.findall(r'\{[^{}]*\}', text)
            if json_match:
                return json.loads(json_match[-1])
            return {"category": "parse_error", "brief_reason": text}
        except Exception as e:
            time.sleep(2 ** i)
            return {"category": "api_error", "brief_reason": str(e)}

tasks = []
for item in final_qa:
    ref_set = set(item["ref_answers"])
    correct_options = [c["Option"] for c in item["choices"] if str(c["Index"]) in ref_set]
    for choice in item["choices"]:
        if str(choice["Index"]) not in ref_set:
            tasks.append({
                "idx": item["idx"],
                "distractor_index": choice["Index"],
                "distractor": choice["Option"],
                "paragraph": item["paragraph"],
                "question": item["question"],
                "correct_options": correct_options,
            })

print(f"Total distractors to classify: {len(tasks)}")

raw_results = []
with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
    futures = {}
    for t in tasks:
        f = pool.submit(classify_distractor, client, t["paragraph"], t["question"], t["correct_options"], t["distractor"])
        futures[f] = t

    for future in tqdm(as_completed(futures), total=len(tasks)):
        t = futures[future]
        classification = future.result()
        raw_results.append({
            "idx": t["idx"],
            "distractor_index": t["distractor_index"],
            "distractor": t["distractor"],
            "category": classification.get("category", "unknown"),
            "reason": classification.get("brief_reason", ""),
        })

results = {}
for r in raw_results:
    if r["idx"] not in results:
        results[r["idx"]] = {}
    results[r["idx"]][r["distractor_index"]] = [r["category"].split(".")[-1].strip().split(":")[0].strip(), r["distractor"], r["reason"]]

results = dict(sorted(results.items(), key=lambda x: int(x[0])))

with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=4)
