from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Optional
import requests

import fitz
import hashlib
import os
import time
from datetime import datetime, timedelta
import json
from threading import Lock
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Q&A Service", version="1.0.0")
auth_scheme = HTTPBearer()

# Configuration
API_KEY = os.getenv("FASTAPI_AUTH_KEY")

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Your Gemini API key
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-pro")  # Gemini model name
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "1000"))  # Max response tokens
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.1"))  # Temperature
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

CACHE_DURATION = 3600  # 1 hour cache duration
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "150000"))  # Maximum text length for context

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

async def answer_question_with_gemini(context: str, question: str) -> str:
    """Answer question using Google Gemini API"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
    
    try:
        # Construct the API URL with the model name
        api_url = GEMINI_API_URL.format(model=MODEL_NAME)
        
        # Add API key as query parameter
        api_url_with_key = f"{api_url}?key={GEMINI_API_KEY}"
        
        # Create the prompt
        prompt = f"""You are an expert document reader and research assistant. Based only on the following document content, answer the question below. Be concise but accurate. If the information is not available in the document, clearly state that the information cannot be found in the provided document.

Document Content:
{context}

Question: {question}

Answer:"""

        # Gemini API payload format
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": MODEL_TEMPERATURE,
                "maxOutputTokens": MODEL_MAX_TOKENS,
                "topP": 0.8,
                "topK": 10
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        logger.info(f"Making Gemini API request to: {api_url}")
        
        response = requests.post(
            api_url_with_key,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        response.raise_for_status()
        response_data = response.json()
        
        # Extract text from Gemini response format
        if "candidates" in response_data and len(response_data["candidates"]) > 0:
            candidate = response_data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if len(parts) > 0 and "text" in parts[0]:
                    return parts[0]["text"].strip()
        
        # Handle blocked content or other issues
        if "candidates" in response_data and len(response_data["candidates"]) > 0:
            candidate = response_data["candidates"][0]
            if "finishReason" in candidate:
                finish_reason = candidate["finishReason"]
                if finish_reason == "SAFETY":
                    return "Content was blocked by safety filters. Please try rephrasing your question."
                elif finish_reason == "RECITATION":
                    return "Content was blocked due to recitation concerns. Please try rephrasing your question."
        
        logger.error(f"Unexpected Gemini response format: {response_data}")
        raise HTTPException(status_code=500, detail="Unexpected response format from Gemini API")
        
    except requests.RequestException as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response content: {e.response.text}")
            
            # Try to parse error details
            try:
                error_data = e.response.json()
                if "error" in error_data:
                    error_msg = error_data["error"].get("message", str(e))
                    raise HTTPException(status_code=500, detail=f"Gemini API error: {error_msg}")
            except:
                pass
                
        raise HTTPException(status_code=500, detail=f"Error calling Gemini API: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

async def answer_questions(context: str, questions: List[str]) -> List[str]:
    """Process all questions and return answers using Gemini"""
    answers = []
    
    for question in questions:
        try:
            answer = await answer_question_with_gemini(context, question)
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
        "cache_entries": len(pdf_cache),
        "gemini_config": {
            "model_name": MODEL_NAME,
            "api_configured": bool(GEMINI_API_KEY),
            "max_tokens": MODEL_MAX_TOKENS,
            "temperature": MODEL_TEMPERATURE
        }
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