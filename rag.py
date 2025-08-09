#!/usr/bin/env python3
"""
offline_rag_full.py
Offline RAG pipeline supporting PDF, Excel, CSV, TXT.

Usage:
  # Index documents (first time, or when docs change)
  python offline_rag_full.py --index --docs_folder ./documents --persist_dir ./chroma_db

  # Ask a question (uses the existing index)
  python offline_rag_full.py --ask "When should I irrigate rice?" --model_bin ./models/mistral-7b-instruct-v0.2.Q4_0.gguf

Notes:
  - Put your docs in the folder you pass to --docs_folder (default "./documents").
  - Download a GGUF/ggml model and pass its path with --model_bin when asking.
"""

import os
import argparse
import uuid
from typing import List, Dict
from tqdm import tqdm

# extraction
import pdfplumber
import pandas as pd

# embeddings
from sentence_transformers import SentenceTransformer

# vector DB
import chromadb

# local LLM (ggml / llama.cpp)
from llama_cpp import Llama

# optional token-counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False

# ---------------- CONFIG ----------------
DEFAULT_PERSIST_DIR = "./chroma_db"
DEFAULT_COLLECTION = "docs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # or "all-MiniLM-L6-v2" if you want smaller/faster
MAX_TOKENS = 400       # chunk token size (tokens, not characters)
OVERLAP_TOKENS = 50
BATCH_EMBED = 64
# ----------------------------------------

def get_token_encoder(name="cl100k_base"):
    if not TIKTOKEN_AVAILABLE:
        return None
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        return None

ENC = get_token_encoder()

def count_tokens(text: str) -> int:
    if ENC:
        return len(ENC.encode(text))
    # fallback heuristic: ~1 token per 4 characters
    return max(1, len(text) // 4)

def chunk_text_tokenwise(text: str, max_tokens:int=MAX_TOKENS, overlap:int=OVERLAP_TOKENS) -> List[str]:
    """Chunk by accumulating words and checking token count (works with or without tiktoken)."""
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        j = i
        current = []
        while j < n:
            current.append(words[j])
            cur_text = " ".join(current)
            if count_tokens(cur_text) > max_tokens:
                current.pop()
                break
            j += 1
        if not current:
            # force single word
            current = [words[i]]
            j = i + 1
        chunks.append(" ".join(current))
        if overlap <= 0:
            i = j
        else:
            # approximate overlap in words (heuristic)
            overlap_words = max(1, int(overlap * 1.3))
            i = max(i + 1, j - overlap_words)
    return chunks

def chunk_text_simple_chars(text: str, chunk_size_chars:int=2000, overlap_chars:int=200) -> List[str]:
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size_chars
        chunks.append(text[start:end])
        start += chunk_size_chars - overlap_chars
    return chunks

# ---------------- file extraction ----------------
def extract_text_from_pdf(path: str) -> str:
    out = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                out.append(t)
    return "\n".join(out)

def extract_text_from_excel(path: str) -> str:
    book = pd.ExcelFile(path)
    parts = []
    for sheet in book.sheet_names:
        df = book.parse(sheet, dtype=str)
        df = df.fillna("")
        parts.append(f"=== Sheet: {sheet} ===\n" + df.to_string(index=False))
    return "\n\n".join(parts)

def extract_text_from_csv(path: str) -> str:
    df = pd.read_csv(path, dtype=str)
    df = df.fillna("")
    return df.to_string(index=False)

def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# ---------------- Indexer ----------------
class LocalIndexer:
    def __init__(self, persist_dir: str = DEFAULT_PERSIST_DIR,
                 collection_name: str = DEFAULT_COLLECTION,
                 embed_model_name: str = EMBED_MODEL_NAME):
        os.makedirs(persist_dir, exist_ok=True)

        # New Chroma API: PersistentClient stores DB on disk at `path`
        self.client = chromadb.PersistentClient(path=persist_dir)

        # get or create collection (works with new client API)
        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)

        self.embedder = SentenceTransformer(embed_model_name)


    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embs = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embs.tolist()

    def upsert(self, docs: List[str], metadatas: List[dict], embeddings: List[List[float]]):
        ids = [str(uuid.uuid4()) for _ in docs]
        # add (or upsert) the items into the collection
        # use add() for new items; upsert() would update existing ids if you reuse ids
        try:
            self.collection.add(documents=docs, metadatas=metadatas, embeddings=embeddings, ids=ids)
        except AttributeError:
            # fallback: some older installs use upsert()
            self.collection.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
        # Persist only if the client provides a persist() (legacy clients). PersistentClient writes to disk automatically.
        # if hasattr(self.client, "persist"):
        #     try:
        #         self.client.persist()
        #     except Exception:
        #         pass
        return ids


    def query(self, query_text: str, top_k: int = 4):
        q_emb = self.embed_batch([query_text])[0]
        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        return docs, metas, ids

# ---------------- LLM wrapper ----------------
class LocalLLM:
    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        kwargs = {"model_path": model_path, "n_ctx": n_ctx}
        if n_threads:
            kwargs["n_threads"] = n_threads
        self.llm = Llama(**kwargs)

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0):
        resp = self.llm(prompt, max_tokens=max_tokens, temperature=temperature)
        return resp["choices"][0]["text"]

# ---------------- Orchestration: index building ----------------
def index_folder(folder: str, indexer: LocalIndexer, use_token_chunking: bool = True,
                 max_tokens:int = MAX_TOKENS, overlap:int = OVERLAP_TOKENS):
    files = [os.path.join(folder,f) for f in os.listdir(folder)]
    docs_to_embed = []
    metas = []
    for fp in files:
        if not os.path.isfile(fp):
            continue
        print(f"[INFO] Processing: {fp}")
        ext = os.path.splitext(fp)[1].lower()
        try:
            if ext == ".pdf":
                text = extract_text_from_pdf(fp)
            elif ext in [".xls", ".xlsx"]:
                text = extract_text_from_excel(fp)
            elif ext == ".csv":
                text = extract_text_from_csv(fp)
            elif ext == ".txt":
                text = extract_text_from_txt(fp)
            else:
                # fallback: attempt to read as text
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except Exception:
                    print(f"[WARN] Unsupported file type or unreadable: {fp}, skipping.")
                    continue
        except Exception as e:
            print(f"[WARN] Failed to extract {fp}: {e}")
            continue

        if not text or not text.strip():
            print("[WARN] empty after extraction, skipping.")
            continue

        # chunk
        if use_token_chunking:
            chunks = chunk_text_tokenwise(text, max_tokens=max_tokens, overlap=overlap)
            if not chunks:
                chunks = chunk_text_simple_chars(text)
        else:
            chunks = chunk_text_simple_chars(text)

        for i, ch in enumerate(chunks):
            docs_to_embed.append(ch)
            metas.append({"source": os.path.basename(fp), "path": os.path.abspath(fp), "chunk_idx": i, "chunk_len": len(ch)})

    print(f"[INFO] Total chunks to embed: {len(docs_to_embed)}")

    # embed in batches
    embeddings = []
    for i in tqdm(range(0, len(docs_to_embed), BATCH_EMBED), desc="Embedding"):
        batch = docs_to_embed[i:i+BATCH_EMBED]
        batch_emb = indexer.embed_batch(batch)
        embeddings.extend(batch_emb)

    # upsert to chroma
    ids = indexer.upsert(docs_to_embed, metas, embeddings)
    print(f"[INFO] Upsert complete ({len(ids)} chunks).")

# ---------------- Prompting & answer ----------------
PROMPT_TEMPLATE = """You are an expert agricultural assistant who speaks simply and directly to farmers.
Only use the information in the CONTEXT to answer. If the exact answer is not in the CONTEXT, say "I don't know" (do not guess).

Requirements for the answer:
1. Give a  **pinpointed** recommendation first (clear action + numbers + units when relevant).
2. After that , optionally add some words explaining why.
3. If the answer cites the CONTEXT, add a short provenance tag in parentheses at the end of the one-line answer — e.g. (report.pdf p12) or (data.xlsx Sheet1 row 4).
4. Use plain language (no jargon), local units (if context shows them), and no long paragraphs.
5. If the question requires further essential info you don't have, ask  short clarifying questions. Otherwise do not ask follow-ups.
6. If the answer is a schedule or numeric plan, show steps or dates as bullets.

CONTEXT:
{context}

QUESTION:
{question}

Now answer following the rules above.

"""

def answer_question(query: str, indexer: LocalIndexer, llm: LocalLLM, top_k:int = 4, max_tokens:int = 256):
    docs, metas, ids = indexer.query(query, top_k=top_k)
    if not docs:
        return "No relevant documents found.", []
    # format context with metadata
    ctx_parts = []
    for d, m in zip(docs, metas):
        src = m.get("source", "unknown")
        idx = m.get("chunk_idx", None)
        label = f"{src}"
        if idx is not None:
            label += f" (chunk {idx})"
        ctx_parts.append(f"[{label}] {d}")
    context = "\n\n---\n\n".join(ctx_parts)
    prompt = PROMPT_TEMPLATE.format(context=context, question=query)
    answer = llm.generate(prompt, max_tokens=max_tokens, temperature=0.0)
    return answer.strip(), metas

# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="Index docs in folder")
    parser.add_argument("--docs_folder", type=str, default="./documents", help="Folder containing PDF/Excel/CSV/TXT files")
    parser.add_argument("--persist_dir", type=str, default=DEFAULT_PERSIST_DIR, help="Chroma persist dir")
    parser.add_argument("--ask", type=str, help="Ask a question using the index")
    parser.add_argument("--model_bin", type=str, help="Path to GGUF/ggml model (required for --ask)")
    parser.add_argument("--top_k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument("--max_tokens", type=int, default=256, help="LLM generation max tokens")
    parser.add_argument("--chunk_tokens", type=int, default=MAX_TOKENS, help="tokens per chunk")
    parser.add_argument("--overlap_tokens", type=int, default=OVERLAP_TOKENS, help="overlap tokens between chunks")
    parser.add_argument("--threads", type=int, default=None, help="n_threads for llama-cpp")
    args = parser.parse_args()

    indexer = LocalIndexer(persist_dir=args.persist_dir, collection_name=DEFAULT_COLLECTION, embed_model_name=EMBED_MODEL_NAME)

    if args.index:
        print("[INFO] Starting indexing...")
        index_folder(args.docs_folder, indexer, use_token_chunking=True, max_tokens=args.chunk_tokens, overlap=args.overlap_tokens)
        print("[INFO] Indexing done.")

    if args.ask:
        if not args.model_bin:
            raise ValueError("--model_bin is required when using --ask")
        print("[INFO] Loading local LLM...")
        llm = LocalLLM(model_path=args.model_bin, n_ctx=4096, n_threads=args.threads)
        print("[INFO] Retrieving and generating...")
        answer, metas = answer_question(args.ask, indexer, llm, top_k=args.top_k, max_tokens=args.max_tokens)
        print("\n=== ANSWER ===\n")
        print(answer)
        print("\n=== SOURCES (top_k) ===\n")
        for m in metas:
            print(m)

if __name__ == "__main__":
    main()
