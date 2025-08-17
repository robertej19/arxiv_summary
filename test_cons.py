import json, pathlib

STORE = pathlib.Path("local_corpus/store")
with open(STORE/"chunk_ids.json") as f:
    ids = set(json.load(f))

present = set()
with open(STORE/"chunks.jsonl") as f:
    for line in f:
        present.add(json.loads(line)["chunk_id"])

missing = ids - present
print("missing:", len(missing))
print(list(sorted(missing))[:10])
