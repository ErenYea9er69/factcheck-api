import os
import json
import asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import AsyncOpenAI
from tavily import AsyncTavilyClient
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="ClaimCheck API", description="Hallucination/fact-verification layer for AI-generated text.")

LONGCAT_API_KEY = os.getenv("LONGCAT_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Initialize clients
openai_client = AsyncOpenAI(
    api_key=LONGCAT_API_KEY,
    base_url="https://api.longcat.chat/openai/v1"
)
tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)

class CheckRequest(BaseModel):
    text: str

class ClaimVerdict(BaseModel):
    claim: str
    verdict: str # confirmed, contradicted, unverifiable, outdated
    confidenceScore: int # 0-100
    sourceLink: Optional[str]
    reasoning: str

async def extract_claims(text: str) -> List[str]:
    prompt = f"""You are an expert fact checker. Extract the core factual claims from the following text that can be verified via web search. Return ONLY a JSON array of strings, where each string is a distinct claim. If no claims are found, return [].
    
Text:
{text}"""

    response = await openai_client.chat.completions.create(
        model="LongCat-2.0-Preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    content = response.choices[0].message.content
    # Try to parse json from content. It might be wrapped in ```json
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        claims = json.loads(content)
        if isinstance(claims, list):
            return claims
        return []
    except Exception as e:
        print(f"Failed to parse claims: {e}")
        return []

async def verify_claim(claim: str) -> ClaimVerdict:
    # Search Tavily
    try:
        search_result = await tavily_client.search(claim, search_depth="basic", max_results=3)
        context = json.dumps(search_result.get("results", []))
    except Exception as e:
        print(f"Tavily search failed for claim '{claim}': {e}")
        context = "[]"
        
    prompt = f"""Given this claim: "{claim}"
And these search results: {context}

Determine if the claim is confirmed, contradicted, unverifiable, or outdated. Return ONLY a JSON object with the following keys:
- verdict (string: one of 'confirmed', 'contradicted', 'unverifiable', 'outdated')
- confidenceScore (number 0-100)
- sourceLink (string, URL from results if applicable, otherwise null)
- reasoning (string, short explanation)"""

    try:
        response = await openai_client.chat.completions.create(
            model="LongCat-2.0-Preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        content = response.choices[0].message.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        data = json.loads(content)
        return ClaimVerdict(
            claim=claim,
            verdict=data.get("verdict", "unverifiable"),
            confidenceScore=data.get("confidenceScore", 0),
            sourceLink=data.get("sourceLink"),
            reasoning=data.get("reasoning", "")
        )
    except Exception as e:
        print(f"Failed to verify claim '{claim}': {e}")
        return ClaimVerdict(claim=claim, verdict="unverifiable", confidenceScore=0, sourceLink=None, reasoning="Failed to verify.")

@app.post("/api/check", response_model=List[ClaimVerdict])
async def check_claims(req: CheckRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    claims = await extract_claims(req.text)
    if not claims:
        return []
        
    tasks = [verify_claim(claim) for claim in claims]
    results = await asyncio.gather(*tasks)
    return results

# Mount static files (Frontend UI)
import os
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
