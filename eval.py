import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

tests = json.load(open("questions.json", encoding="utf-8"))
for i, t in enumerate(tests):
    missing = {"question", "source", "pages"} - set(t.keys())
    if missing:
        raise SystemExit(f"Entry {i} in questions.json is missing: {missing}")

def evaluate(k=5):
    hits, rr_total = 0, 0
    for t in tests:
        results = db.similarity_search(t["question"], k=k)
        found_rank = None
        for rank, r in enumerate(results, start=1):
            page = r.metadata.get("page", 0) + 1
            if r.metadata["source"] == t["source"] and page in t["pages"]:
                found_rank = rank
                break
        if found_rank:
            hits += 1
            rr_total += 1 / found_rank
        else:
            print("MISS:", t["question"])
    n = len(tests)
    print(f"\nk={k} | Hit rate: {hits}/{n} = {hits/n:.0%} | MRR: {rr_total/n:.2f}")

for k in (1, 3, 5):
    evaluate(k)