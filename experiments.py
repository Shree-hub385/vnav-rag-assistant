import json, csv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

tests = json.load(open("questions.json", encoding="utf-8"))

# Load pages once
docs = []
for pdf in Path("data").glob("*.pdf"):
    for p in PyPDFLoader(str(pdf)).load():
        p.metadata["source"] = pdf.name
        docs.append(p)
docs = [d for d in docs if len(d.page_content.strip()) > 100]

def score(search):
    h1 = h3 = h5 = 0
    rr = 0
    for t in tests:
        results = search(t["question"])
        rank = None
        for i, r in enumerate(results, start=1):
            page = r.metadata.get("page", 0) + 1
            if r.metadata["source"] == t["source"] and page in t["pages"]:
                rank = i
                break
        if rank:
            rr += 1 / rank
            h1 += rank <= 1
            h3 += rank <= 3
            h5 += rank <= 5
    n = len(tests)
    return h1 / n, h3 / n, h5 / n, rr / n

models = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
}
chunk_configs = [(500, 100), (1000, 150), (1500, 200), (2000, 300)]

rows = []
for tag, model_name in models.items():
    emb = HuggingFaceEmbeddings(model_name=model_name)
    for size, overlap in chunk_configs:
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap
        ).split_documents(docs)
        db = Chroma.from_documents(
            chunks, emb, collection_name=f"{tag}_{size}_{overlap}"
        )
        methods = {
            "similarity": lambda q: db.similarity_search(q, k=5),
            "mmr": lambda q: db.max_marginal_relevance_search(q, k=5, fetch_k=20),
        }
        for mname, fn in methods.items():
            h1, h3, h5, mrr = score(fn)
            row = [tag, size, overlap, mname, len(chunks),
                   f"{h1:.0%}", f"{h3:.0%}", f"{h5:.0%}", f"{mrr:.2f}"]
            rows.append(row)
            print(row)

header = ["model", "chunk", "overlap", "search", "n_chunks",
          "hit@1", "hit@3", "hit@5", "MRR"]
with open("results.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows([header] + rows)
print("\nSaved results.csv")