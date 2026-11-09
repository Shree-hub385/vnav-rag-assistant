import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from google import genai
from google.genai import errors

MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
]

def load_components():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    client = genai.Client()   # reads GEMINI_API_KEY from the environment
    return db, reranker, client

def retrieve(db, reranker, question, fetch=20, top=5):
    candidates = db.similarity_search(question, k=fetch)
    scores = reranker.predict([(question, c.page_content) for c in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top]]

def generate(client, prompt, tries=2):
    for model in MODELS:
        for attempt in range(tries):
            try:
                return client.models.generate_content(model=model, contents=prompt).text
            except (errors.ServerError, errors.ClientError) as e:
                code = getattr(e, "code", None)
                if code == 404:
                    break
                if code in (429, 503):
                    if attempt < tries - 1:
                        time.sleep(5)
                    continue
                return f"Gemini error: {e}"
    return "All models are busy right now. Please try again in a few minutes."

def answer(db, reranker, client, question):
    results = retrieve(db, reranker, question)

    context = ""
    for r in results:
        context += f"[{r.metadata['source']}, page {r.metadata.get('page', 0) + 1}]\n"
        context += r.page_content + "\n\n"

    prompt = f"""You are a teaching assistant for the MIT course 16.485 (Visual Navigation for Autonomous Vehicles).
Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find this in the lecture notes."
At the end, list the sources you used as (file, page).

Context:
{context}

Question: {question}
"""
    return generate(client, prompt), results