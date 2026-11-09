import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

shutil.rmtree("chroma_db", ignore_errors=True)

docs = []
for pdf in Path("data").glob("*.pdf"):
    for p in PyPDFLoader(str(pdf)).load():
        p.metadata["source"] = pdf.name
        docs.append(p)

docs = [d for d in docs if len(d.page_content.strip()) > 100]

chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
print(f"{len(docs)} pages -> {len(chunks)} chunks")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
print("Done")