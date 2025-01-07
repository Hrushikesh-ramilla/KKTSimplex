import json
import os

path = os.path.join(os.path.dirname(__file__), "data", "are_you_sure.jsonl")
with open(path, "r", encoding="utf-8") as f:
    for i in range(5):
        item = json.loads(f.readline())
        print(f"Item {i} len(prompt)={len(item['prompt'])} dataset={item['base']['dataset']} correct={item['base']['correct_letter']}")
        if len(item['prompt']) > 2:
            print("Prompt turns:", [p['type'] for p in item['prompt']])
