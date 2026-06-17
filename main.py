import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from difflib import SequenceMatcher

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from openai import AsyncOpenAI
from tavily import AsyncTavilyClient
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("claimcheck")

app = FastAPI(
    title="ClaimCheck API",
    description="Production-grade hallucination / fact-verification layer for AI-generated text.",
    version="2.0.0",
)

LONGCAT_API_KEY = os.getenv("LONGCAT_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
MODEL_NAME = os.getenv("CLAIMCHECK_MODEL", "LongCat-2.0-Preview")

# How many claims to process at once (controls cost + rate limits)
MAX_CLAIMS = int(os.getenv("CLAIMCHECK_MAX_CLAIMS", "10"))
CONCURRENCY = int(os.getenv("CLAIMCHECK_CONCURRENCY", "5"))
# Similarity threshold above which two claims are considered duplicates (0-1)
DEDUP_THRESHOLD = float(os.getenv("CLAIMCHECK_DEDUP_THRESHOLD", "0.82"))
# Tavily results per claim
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

openai_client = AsyncOpenAI(
    api_key=LONGCAT_API_KEY,
    base_url="https://api.longcat.chat/openai/v1",
)
tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class VerdictEnum(str, Enum):
    confirmed = "confirmed"
    contradicted = "contradicted"
    unverifiable = "unverifiable"
    outdated = "outdated"


class SourceEntry(BaseModel):
    title: str
    url: str
    snippet: str = ""
    published_date: Optional[str] = None


class ClaimVerdict(BaseModel):
    claim: str
    char_start: Optional[int] = None  # offset in the original input text
    char_end: Optional[int] = None
    verdict: VerdictEnum
    confidence_score: int = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Confidence in the verdict (0-100). "
            "Does NOT mean the claim is true; it measures how strongly "
            "the available search evidence supports the verdict."
        ),
    )
    sources: List[SourceEntry] = Field(
        default_factory=list,
        description="Evidence sources that informed the verdict.",
    )
    reasoning: str = Field(..., min_length=1)
    checked_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    model_used: str = MODEL_NAME

    @field_validator("verdict", mode="before")
    @classmethod
    def coerce_verdict(cls, v):
        """Accept minor LLM variations and normalise; fall back to unverifiable."""
        if isinstance(v, str):
            clean = v.strip().lower()
            mapping = {
                "confirmed": VerdictEnum.confirmed,
                "contradicted": VerdictEnum.contradicted,
                "unverifiable": VerdictEnum.unverifiable,
                "outdated": VerdictEnum.outdated,
                # common LLM aliases
                "supported": VerdictEnum.confirmed,
                "verified": VerdictEnum.confirmed,
                "true": VerdictEnum.confirmed,
                "false": VerdictEnum.contradicted,
                "disproved": VerdictEnum.contradicted,
                "incorrect": VerdictEnum.contradicted,
                "stale": VerdictEnum.outdated,
                "no longer accurate": VerdictEnum.outdated,
                "unknown": VerdictEnum.unverifiable,
                "insufficient evidence": VerdictEnum.unverifiable,
            }
            if clean in mapping:
                return mapping[clean]
            logger.warning("Unexpected verdict value from LLM: %r — defaulting to unverifiable", v)
        return VerdictEnum.unverifiable


class CheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20_000)
    search_depth: str = Field(
        default="basic",
        pattern="^(basic|advanced)$",
        description="Tavily search depth. Use 'advanced' for nuanced or contested claims.",
    )
    max_claims: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Override the server-default claim cap for this request.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _deduplicate_claims(claims: List[str]) -> List[str]:
    """Remove near-duplicate claims using sequence similarity."""
    unique: List[str] = []
    for candidate in claims:
        if not any(_similarity(candidate, kept) >= DEDUP_THRESHOLD for kept in unique):
            unique.append(candidate)
    return unique


def _find_span(text: str, claim: str) -> tuple[Optional[int], Optional[int]]:
    """Try to locate the claim substring in the original text (case-insensitive)."""
    lower_text = text.lower()
    lower_claim = claim.lower().strip(".")
    idx = lower_text.find(lower_claim)
    if idx != -1:
        return idx, idx + len(lower_claim)
    return None, None


def _strip_tavily_results(raw_results: list) -> list:
    """Keep only the fields we actually need to avoid flooding the LLM context."""
    cleaned = []
    for r in raw_results:
        cleaned.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": (r.get("content", "") or "")[:400],
                "published_date": r.get("published_date"),
            }
        )
    return cleaned


async def _call_llm(prompt: str, attempt: int = 1) -> str:
    """Call the LLM and return the raw content string."""
    response = await openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


def _parse_json_from_llm(content: str) -> dict | list:
    """Strip markdown fences and parse JSON from LLM output."""
    text = content.strip()
    for fence in ("```json", "```"):
        if fence in text:
            text = text.split(fence, 1)[1].split("```", 1)[0].strip()
            break
    return json.loads(text)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

async def extract_claims(text: str, cap: int) -> List[str]:
    """
    Ask the LLM to extract distinct, verifiable factual claims.
    Retries once with a stricter prompt on parse failure.
    Returns at most `cap` deduplicated claims.
    """
    prompt = (
        "You are an expert fact-checker. Extract every distinct factual claim "
        "from the text below that can be independently verified via a web search. "
        "Exclude opinions, predictions, and vague statements.\n\n"
        "Rules:\n"
        "- Return ONLY a raw JSON array of strings. No preamble, no markdown fences, no explanation.\n"
        "- Each string is one self-contained claim, written as a declarative sentence.\n"
        "- Maximum items: {cap}.\n"
        "- If there are no verifiable claims, return: []\n\n"
        "Text:\n{text}"
    ).format(cap=cap, text=text)

    for attempt in range(1, 3):
        try:
            raw = await _call_llm(prompt, attempt=attempt)
            parsed = _parse_json_from_llm(raw)
            if isinstance(parsed, list):
                claims = [c for c in parsed if isinstance(c, str) and c.strip()]
                claims = _deduplicate_claims(claims)
                logger.info("Extracted %d unique claims (attempt %d)", len(claims), attempt)
                return claims[:cap]
        except Exception as exc:
            logger.warning("Claim extraction parse error on attempt %d: %s", attempt, exc)
            if attempt == 1:
                # Stricter second attempt
                prompt = (
                    "Return ONLY a JSON array of strings. "
                    "No other text whatsoever. "
                    "Extract verifiable factual claims from:\n\n" + text
                )

    logger.error("Claim extraction failed after 2 attempts; returning empty list")
    return []


async def verify_claim(
    claim: str, original_text: str, search_depth: str
) -> ClaimVerdict:
    """Search for evidence and ask the LLM to render a verdict."""
    # 1 — Search
    raw_results: list = []
    try:
        search_result = await tavily_client.search(
            claim,
            search_depth=search_depth,
            max_results=TAVILY_MAX_RESULTS,
        )
        raw_results = search_result.get("results", [])
    except Exception as exc:
        logger.warning("Tavily search failed for claim %r: %s", claim, exc)

    stripped_results = _strip_tavily_results(raw_results)
    context_json = json.dumps(stripped_results, ensure_ascii=False)

    # 2 — LLM verdict
    prompt = (
        "You are a rigorous fact-checker. Your job is to evaluate a single claim "
        "using the web search results provided.\n\n"
        "Definitions:\n"
        "- confirmed: The claim is directly supported by at least one reliable source.\n"
        "- contradicted: The claim conflicts with evidence — it was never true or is factually wrong.\n"
        "- outdated: The claim WAS true at some point but is no longer accurate (e.g. a record that was broken, "
        "a role that changed, a version that was superseded). Use this only when you can identify a specific "
        "point in time when it stopped being true.\n"
        "- unverifiable: The search results do not contain sufficient evidence to confirm or deny the claim.\n\n"
        "Claim:\n"
        f'"{claim}"\n\n'
        "Search results (JSON):\n"
        f"{context_json}\n\n"
        "Return ONLY a raw JSON object with exactly these keys:\n"
        "  verdict        — one of: confirmed | contradicted | outdated | unverifiable\n"
        "  confidence_score — integer 0-100 (confidence IN THE VERDICT, not in the claim being true)\n"
        "  reasoning      — 1-3 sentences explaining the verdict; never empty\n"
        "  source_indices — list of integer indices (0-based) into the search results that support the verdict\n\n"
        "No preamble, no markdown fences, no extra keys."
    )

    verdict_raw = VerdictEnum.unverifiable
    confidence_raw = 0
    reasoning_raw = "Verification failed — no response from model."
    source_indices: List[int] = []

    for attempt in range(1, 3):
        try:
            raw = await _call_llm(prompt, attempt=attempt)
            data = _parse_json_from_llm(raw)
            verdict_raw = data.get("verdict", "unverifiable")
            confidence_raw = int(data.get("confidence_score", 0))
            reasoning_raw = data.get("reasoning") or "No reasoning provided."
            source_indices = [
                i for i in data.get("source_indices", [])
                if isinstance(i, int) and 0 <= i < len(stripped_results)
            ]
            break
        except Exception as exc:
            logger.warning(
                "Verdict parse error for claim %r on attempt %d: %s", claim, attempt, exc
            )

    # 3 — Build source list from indices
    sources: List[SourceEntry] = []
    seen_urls = set()
    for idx in source_indices:
        r = stripped_results[idx]
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources.append(
                SourceEntry(
                    title=r.get("title", ""),
                    url=url,
                    snippet=r.get("snippet", ""),
                    published_date=r.get("published_date"),
                )
            )

    # If the LLM gave no indices but we have results, attach the top result as fallback
    if not sources and stripped_results:
        r = stripped_results[0]
        sources.append(
            SourceEntry(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("snippet", ""),
                published_date=r.get("published_date"),
            )
        )

    # 4 — Span detection
    char_start, char_end = _find_span(original_text, claim)

    return ClaimVerdict(
        claim=claim,
        char_start=char_start,
        char_end=char_end,
        verdict=verdict_raw,
        confidence_score=max(0, min(100, confidence_raw)),
        sources=sources,
        reasoning=reasoning_raw,
        model_used=MODEL_NAME,
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/api/check",
    response_model=List[ClaimVerdict],
    summary="Extract and verify all factual claims in the provided text.",
)
async def check_claims(req: CheckRequest) -> List[ClaimVerdict]:
    cap = min(req.max_claims or MAX_CLAIMS, MAX_CLAIMS)
    logger.info(
        "New /api/check request — text length=%d, search_depth=%s, cap=%d",
        len(req.text),
        req.search_depth,
        cap,
    )

    claims = await extract_claims(req.text, cap)
    if not claims:
        logger.info("No verifiable claims found in input.")
        return []

    # Bounded concurrency via semaphore
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def guarded_verify(claim: str) -> ClaimVerdict:
        async with semaphore:
            return await verify_claim(claim, req.text, req.search_depth)

    results = await asyncio.gather(*[guarded_verify(c) for c in claims])
    logger.info("Completed %d verifications.", len(results))
    return list(results)


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)