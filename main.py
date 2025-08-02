from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Optional
import requests
import fitz  # PyMuPDF
import hashlib
import os
import time
from datetime import datetime, timedelta

from threading import Lock
import logging

from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Access them like this:
gemini_key= os.getenv("GEMINI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Q&A Service", version="1.0.0")
auth_scheme = HTTPBearer()

# Configuration
# Instead of hardcoding:
API_KEY = os.getenv("FASTAPI_AUTH_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # Set your Claude API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
gemini_key= os.getenv("GEMINI_API_KEY") # Alternative: Set your OpenAI API key
CACHE_DURATION = 3600  # 1 hour cache duration
MAX_TEXT_LENGTH = 150000  # Maximum text length for context (adjust based on model limits)

# In-memory cache for PDF content and extracted text
pdf_cache: Dict[str, Dict] = {}
cache_lock = Lock()

class RequestBody(BaseModel):
    documents: HttpUrl
    questions: List[str]
    
class ResponseBody(BaseModel):
    answers: List[str]
    
class SessionInfo(BaseModel):
    document_hash: str
    extracted_text: str
    timestamp: datetime
    url: str

def verify_token(creds: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if creds.scheme.lower() != "bearer" or creds.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return True

def get_document_hash(url: str) -> str:
    """Generate a hash for the document URL to use as cache key"""
    return hashlib.md5(url.encode()).hexdigest()

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF using PyMuPDF"""
    try:
        # Open PDF from bytes
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        text = ""
        
        # Extract text from all pages
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()
            text += "\n\n"  # Add spacing between pages
        
        doc.close()
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF: {str(e)}")

def chunk_text_intelligently(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Intelligently truncate or chunk text if it's too long"""
    if len(text) <= max_length:
        return text
    
    # Try to find a good breaking point (end of paragraph, sentence, etc.)
    truncated = text[:max_length]
    
    # Find the last complete sentence or paragraph
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n\n')
    
    if last_period > max_length * 0.8:  # If we can keep 80% of text with complete sentence
        return truncated[:last_period + 1]
    elif last_newline > max_length * 0.7:  # If we can keep 70% with complete paragraph
        return truncated[:last_newline]
    else:
        return truncated + "..."

def clean_cache():
    """Remove expired cache entries"""
    current_time = datetime.now()
    with cache_lock:
        expired_keys = [
            key for key, value in pdf_cache.items()
            if current_time - value['timestamp'] > timedelta(seconds=CACHE_DURATION)
        ]
        for key in expired_keys:
            del pdf_cache[key]
            logger.info(f"Removed expired cache entry: {key}")

def get_cached_text(document_hash: str) -> Optional[str]:
    """Get cached extracted text if available and not expired"""
    with cache_lock:
        if document_hash in pdf_cache:
            cache_entry = pdf_cache[document_hash]
            if datetime.now() - cache_entry['timestamp'] < timedelta(seconds=CACHE_DURATION):
                logger.info(f"Using cached text for document: {document_hash}")
                return cache_entry['extracted_text']
            else:
                # Remove expired entry
                del pdf_cache[document_hash]
                logger.info(f"Removed expired cache entry: {document_hash}")
    return None

def cache_extracted_text(document_hash: str, text: str, url: str):
    """Cache the extracted text"""
    with cache_lock:
        pdf_cache[document_hash] = {
            'extracted_text': text,
            'timestamp': datetime.now(),
            'url': url
        }
        logger.info(f"Cached text for document: {document_hash}")

async def answer_question_with_claude(context: str, question: str) -> str:
    """Answer question using Claude (Anthropic)"""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Claude API key not configured")
    
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        prompt = f"""You are an expert document reader and research assistant. Based only on the following document content, answer the question below. Be concise but accurate. If the information is not available in the document, clearly state that the information cannot be found in the provided document.

Document Content:
{context}

Question: {question}

Answer:"""

        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return message.content[0].text.strip()
    
    except Exception as e:
        logger.error(f"Error calling Claude API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

async def answer_question_with_openai(context: str, question: str) -> str:
    """Answer question using OpenAI GPT (alternative implementation)"""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        
        prompt = f"""You are an expert document reader and research assistant. Based only on the following document content, answer the question below. Be concise but accurate. If the information is not available in the document, clearly state that the information cannot be found in the provided document.

Document Content:
{context}

Question: {question}

Answer:"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful document analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

async def answer_questions(context: str, questions: List[str]) -> List[str]:
    """Process all questions and return answers"""
    answers = []
    
    for question in questions:
        try:
            # Try Claude first, fallback to OpenAI if available
            if ANTHROPIC_API_KEY:
                answer = await answer_question_with_claude(context, question)
            elif OPENAI_API_KEY:
                answer = await answer_question_with_openai(context, question)
            else:
                # Fallback to simple keyword-based response if no API keys
                answer = f"Unable to process question '{question}' - No AI model configured. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable."
            
            answers.append(answer)
            logger.info(f"Generated answer for question: {question[:50]}...")
            
        except Exception as e:
            error_msg = f"Error processing question '{question}': {str(e)}"
            logger.error(error_msg)
            answers.append(f"Error: Could not process this question. {str(e)}")
    
    return answers

@app.post("/hackrx/run", response_model=ResponseBody)
async def hackrx_run(body: RequestBody, authorized: bool = Depends(verify_token)):
    """Main endpoint for PDF Q&A processing"""
    
    # Clean expired cache entries
    clean_cache()
    
    document_url = str(body.documents)
    document_hash = get_document_hash(document_url)
    
    # Check if we have cached text for this document
    cached_text = get_cached_text(document_hash)
    
    if cached_text:
        extracted_text = cached_text
        logger.info("Using cached PDF text")
    else:
        # Fetch the document from URL
        logger.info(f"Fetching document from: {document_url}")
        try:
            response = requests.get(document_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch document: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to fetch document from URL: {str(e)}")
        
        # Extract text from PDF
        logger.info("Extracting text from PDF")
        extracted_text = extract_text_from_pdf(response.content)
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in the PDF document")
        
        # Cache the extracted text
        cache_extracted_text(document_hash, extracted_text, document_url)
    
    # Chunk text if too long
    context = chunk_text_intelligently(extracted_text)
    logger.info(f"Using context of length: {len(context)} characters")
    
    # Process questions and generate answers
    logger.info(f"Processing {len(body.questions)} questions")
    answers = await answer_questions(context, body.questions)
    
    return ResponseBody(answers=answers)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_entries": len(pdf_cache)
    }

@app.get("/cache/stats")
async def cache_stats(authorized: bool = Depends(verify_token)):
    """Get cache statistics"""
    with cache_lock:
        stats = {
            "total_entries": len(pdf_cache),
            "entries": [
                {
                    "hash": key[:8] + "...",
                    "url": value["url"][:50] + "..." if len(value["url"]) > 50 else value["url"],
                    "text_length": len(value["extracted_text"]),
                    "cached_at": value["timestamp"].isoformat(),
                    "expires_at": (value["timestamp"] + timedelta(seconds=CACHE_DURATION)).isoformat()
                }
                for key, value in pdf_cache.items()
            ]
        }
    return stats

@app.delete("/cache/clear")
async def clear_cache(authorized: bool = Depends(verify_token)):
    """Clear all cached entries"""
    with cache_lock:
        cleared_count = len(pdf_cache)
        pdf_cache.clear()
    
    logger.info(f"Cleared {cleared_count} cache entries")
    return {"message": f"Cleared {cleared_count} cache entries"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)