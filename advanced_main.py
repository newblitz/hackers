from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List
import fitz  # PyMuPDF
import requests
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
auth_scheme = HTTPBearer()

# === API Keys ===
INTERNAL_API_KEY = "d3ac456931faffea79a7c00f08a3e190998e84d9925709bb117e750078149d05"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# === Gemini Setup ===
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# === Embedding Model Setup ===
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # or 'paraphrase-MiniLM-L6-v2'

# === Schemas ===
class RequestBody(BaseModel):
    documents: HttpUrl
    questions: List[str]

class ResponseBody(BaseModel):
    answers: List[str]

# === Token verification ===
def verify_token(creds: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if creds.scheme.lower() != "bearer" or creds.credentials != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return True

# === PDF Text Extraction ===
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# === Chunking ===
def chunk_text(text: str, max_words: int = 300) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# === Semantic Embeddings for Chunks ===
def embed_chunks(chunks: List[str]):
    return embedder.encode(chunks, convert_to_tensor=False)

# === Find Most Relevant Chunk ===
def find_best_chunk(question: str, chunks: List[str], chunk_embeddings) -> str:
    question_embedding = embedder.encode([question])[0]
    similarities = cosine_similarity([question_embedding], chunk_embeddings)
    best_index = similarities.argmax()
    return chunks[best_index]

# === Gemini Prompting ===
def ask_gemini(question: str, context: str) -> str:
    prompt = f"""Answer the question in one line based on the context.

    Context: {context}
    Question: {question}
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

# === Main Endpoint ===
@app.post("/hackrx/run", response_model=ResponseBody)
async def hackrx_run(body: RequestBody, authorized: bool = Depends(verify_token)):
    # Step 1: Download PDF
    resp = requests.get(body.documents)
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch document URL")

    # Step 2: Extract and chunk text
    text = extract_text_from_pdf(resp.content)
    chunks = chunk_text(text)

    # Step 3: Embed chunks
    chunk_embeddings = embed_chunks(chunks)

    # Step 4: Answer each question
    answers = []
    for question in body.questions:
        best_chunk = find_best_chunk(question, chunks, chunk_embeddings)
        answer = ask_gemini(question, best_chunk)
        answers.append(answer)

    return ResponseBody(answers=answers)
