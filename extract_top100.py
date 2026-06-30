import json
import csv

# Read Top 100 candidate IDs from submission.csv
top_ids = []

with open("outputs/submission.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        top_ids.append(row["candidate_id"])

print(f"Loaded {len(top_ids)} candidate IDs")

# Read candidates.jsonl and keep only Top 100
top_candidates = []

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            candidate = json.loads(line)
            if candidate["candidate_id"] in top_ids:
                top_candidates.append(candidate)

# Preserve ranking order
id_map = {c["candidate_id"]: c for c in top_candidates}

ordered = []

for cid in top_ids:
    if cid in id_map:
        ordered.append(id_map[cid])

# Save JSON
with open("data/top100_candidates.json", "w", encoding="utf-8") as f:
    json.dump(ordered, f, indent=2)

print(f"Saved {len(ordered)} candidates")