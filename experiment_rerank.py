import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

tests = json.load(open("questions.json", encoding="utf-8"))

docs = []
for pdf in Path("data").glob("*.pdf"):
    for p in PyPDFLoader(str(pdf)).load():
        p.metadata["source"] = pdf.name
        docs.append(p)
docs = [d for d in docs if len(d.page_content.strip()) > 100]

chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(chunks, emb, collection_name="rerank_test")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def plain(q):
    return db.similarity_search(q, k=5)

def reranked(q, fetch=20):
    cands = db.similarity_search(q, k=fetch)
    scores = reranker.predict([(q, c.page_content) for c in cands])
    ranked = sorted(zip(scores, cands), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:5]]

def score(search, label):
    h1 = h3 = h5 = 0
    rr = 0
    not_first = []
    for t in tests:
        rank = None
        for i, r in enumerate(search(t["question"]), start=1):
            page = r.metadata.get("page", 0) + 1
            if r.metadata["source"] == t["source"] and page in t["pages"]:
                rank = i
                break
        if rank:
            rr += 1 / rank
            h1 += rank <= 1
            h3 += rank <= 3
            h5 += rank <= 5
        if rank != 1:
            not_first.append((rank, t["question"][:70]))
    n = len(tests)
    print(f"\n{label}: hit@1 {h1/n:.0%} | hit@3 {h3/n:.0%} | hit@5 {h5/n:.0%} | MRR {rr/n:.2f}")
    for rank, q in not_first:
        print(f"   rank={rank}: {q}")

score(plain, "Plain similarity")
score(reranked, "With reranker")