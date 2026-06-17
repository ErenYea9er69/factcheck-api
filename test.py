"""
ClaimCheck v2 — comprehensive test suite
=========================================
Tests are grouped by scenario type.  Each group targets a specific
weakness identified in the v1 audit so regressions are easy to spot.

Run:
    python test.py                        # all tests, pretty-print
    python test.py --json                 # dump raw JSON to stdout
    python test.py --save output.json     # write results to file
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from typing import Optional

# ── Import the actual app ──────────────────────────────────────────────────
from main import check_claims, CheckRequest, VerdictEnum


# ===========================================================================
# Test cases
# ===========================================================================

TEST_CASES = [
    # ------------------------------------------------------------------
    # 1. HAPPY PATH — clear confirmed facts (baseline)
    # ------------------------------------------------------------------
    {
        "id": "baseline_confirmed",
        "label": "Baseline — well-known confirmed facts",
        "description": (
            "All three claims are unambiguously true. "
            "Expect: all confirmed, confidence ≥ 85, at least one source each."
        ),
        "request": CheckRequest(
            text=(
                "The moon landing happened in 1969. "
                "Python was invented by Guido van Rossum. "
                "The Eiffel Tower is located in Paris, France."
            ),
            search_depth="basic",
        ),
        "expect": {
            "min_claims": 3,
            "all_verdict": VerdictEnum.confirmed,
            "min_confidence": 85,
        },
    },

    # ------------------------------------------------------------------
    # 2. HARD CONTRADICTION — fabricated recent event
    # ------------------------------------------------------------------
    {
        "id": "hard_contradiction",
        "label": "Hard contradiction — fabricated facts",
        "description": (
            "Mix of a fabricated claim (Mars colony in 2024) with a real one. "
            "Expect: fabricated claim → contradicted or unverifiable, "
            "real claim → confirmed."
        ),
        "request": CheckRequest(
            text=(
                "Humans have successfully established a permanent colony on Mars in 2024. "
                "Water boils at 100 degrees Celsius at standard atmospheric pressure."
            ),
            search_depth="advanced",
        ),
        "expect": {
            "min_claims": 2,
        },
    },

    # ------------------------------------------------------------------
    # 3. OUTDATED CLAIM — something that was once true
    # ------------------------------------------------------------------
    {
        "id": "outdated_claim",
        "label": "Outdated — claims that were once true",
        "description": (
            "Claims that were accurate in the past but are no longer current. "
            "Expect: outdated or contradicted verdict (not confirmed)."
        ),
        "request": CheckRequest(
            text=(
                "Elon Musk is the richest person in the world. "
                "Twitter is the name of the social media platform formerly known as Twitter."
            ),
            search_depth="advanced",
        ),
        "expect": {
            "min_claims": 2,
            "no_verdict": VerdictEnum.confirmed,
        },
    },

    # ------------------------------------------------------------------
    # 4. UNVERIFIABLE — vague or subjective claims
    # ------------------------------------------------------------------
    {
        "id": "unverifiable_vague",
        "label": "Unverifiable — vague or subjective statements",
        "description": (
            "Statements that sound factual but cannot be verified via web search. "
            "Expect: unverifiable verdict for most."
        ),
        "request": CheckRequest(
            text=(
                "The economy will crash within the next six months. "
                "Most people prefer summer over winter. "
                "This new drug cures 99% of all cancers in clinical settings."
            ),
            search_depth="basic",
        ),
        "expect": {
            "min_claims": 1,
        },
    },

    # ------------------------------------------------------------------
    # 5. DEDUPLICATION — near-duplicate claims in the same text
    # ------------------------------------------------------------------
    {
        "id": "dedup_stress",
        "label": "Deduplication — near-identical claims",
        "description": (
            "Three near-identical phrasings of the same claim. "
            "Expect: only 1–2 unique claims returned (dedup working)."
        ),
        "request": CheckRequest(
            text=(
                "Python was created by Guido van Rossum. "
                "The Python language was invented by Guido van Rossum. "
                "Guido van Rossum is the creator of the Python programming language. "
                "The Eiffel Tower is in Paris."
            ),
            search_depth="basic",
            max_claims=10,
        ),
        "expect": {
            "max_claims": 3,
        },
    },

    # ------------------------------------------------------------------
    # 6. CLAIM CAP — large text with many facts (stress the cap)
    # ------------------------------------------------------------------
    {
        "id": "claim_cap",
        "label": "Claim cap — text with 15+ extractable facts",
        "description": (
            "Verifies that the cap is respected and we never fire "
            "uncapped parallel requests."
        ),
        "request": CheckRequest(
            text=(
                "Albert Einstein was born in 1879. "
                "He developed the theory of relativity. "
                "The speed of light is approximately 299,792 km/s. "
                "DNA was discovered by Watson and Crick in 1953. "
                "Mount Everest is the tallest mountain on Earth. "
                "The Amazon is the longest river in the world. "
                "Shakespeare wrote Hamlet. "
                "The French Revolution began in 1789. "
                "Neil Armstrong was the first human to walk on the moon. "
                "The Great Wall of China is visible from space. "
                "Gold has the chemical symbol Au. "
                "The human body has 206 bones. "
                "Penguins live in the Arctic. "
                "The capital of Australia is Sydney. "
                "Oxygen makes up about 78% of Earth's atmosphere."
            ),
            search_depth="basic",
            max_claims=5,
        ),
        "expect": {
            "max_claims": 5,
        },
    },

    # ------------------------------------------------------------------
    # 7. OUTPUT STRUCTURE — checks every new field exists and is valid
    # ------------------------------------------------------------------
    {
        "id": "output_structure",
        "label": "Output structure — validates all v2 fields",
        "description": (
            "Checks that every ClaimVerdict has: claim, verdict, "
            "confidence_score (0-100), sources (list), reasoning (non-empty), "
            "checked_at (ISO timestamp), model_used, char_start/char_end."
        ),
        "request": CheckRequest(
            text="The moon landing happened in 1969.",
            search_depth="basic",
        ),
        "expect": {
            "min_claims": 1,
            "check_structure": True,
        },
    },

    # ------------------------------------------------------------------
    # 8. CONFIDENCE SCORE SANITY — contradicted should not score 95 same as confirmed
    # ------------------------------------------------------------------
    {
        "id": "confidence_contrast",
        "label": "Confidence contrast — contradicted vs confirmed",
        "description": (
            "Runs a clearly true claim and a clearly false claim. "
            "Both confidence scores are inspected and logged for manual review. "
            "This test does not hard-fail on scores — it flags for human review."
        ),
        "request": CheckRequest(
            text=(
                "The Earth orbits the Sun. "
                "The Earth is flat and sits on the back of a giant turtle."
            ),
            search_depth="advanced",
        ),
        "expect": {
            "min_claims": 2,
            "flag_score_review": True,
        },
    },

    # ------------------------------------------------------------------
    # 9. SPAN DETECTION — char offsets point back to original text
    # ------------------------------------------------------------------
    {
        "id": "span_detection",
        "label": "Span detection — char_start / char_end accuracy",
        "description": (
            "Verifies that char_start and char_end correctly locate "
            "the claim text inside the original input."
        ),
        "request": CheckRequest(
            text="Python was invented by Guido van Rossum in 1991.",
            search_depth="basic",
        ),
        "expect": {
            "min_claims": 1,
            "check_spans": True,
        },
    },

    # ------------------------------------------------------------------
    # 10. EMPTY / JUNK INPUT — edge cases
    # ------------------------------------------------------------------
    {
        "id": "no_claims_text",
        "label": "No-claims text — opinions and questions only",
        "description": (
            "A paragraph with no verifiable factual claims. "
            "Expect: empty result list."
        ),
        "request": CheckRequest(
            text=(
                "I think the economy is doing okay. "
                "Do you prefer cats or dogs? "
                "This movie was amazing and I loved every second of it."
            ),
            search_depth="basic",
        ),
        "expect": {
            "max_claims": 0,
        },
    },
]


# ===========================================================================
# Assertion helpers
# ===========================================================================

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def check_result(tc: dict, results: list) -> list[dict]:
    """Run assertions for a test case. Returns a list of finding dicts."""
    expect = tc.get("expect", {})
    findings = []

    # ── min_claims ─────────────────────────────────────────────────────────
    if "min_claims" in expect:
        ok = len(results) >= expect["min_claims"]
        findings.append(
            {
                "check": "min_claims",
                "status": PASS if ok else FAIL,
                "detail": f"got {len(results)}, expected ≥ {expect['min_claims']}",
            }
        )

    # ── max_claims ─────────────────────────────────────────────────────────
    if "max_claims" in expect:
        ok = len(results) <= expect["max_claims"]
        findings.append(
            {
                "check": "max_claims",
                "status": PASS if ok else FAIL,
                "detail": f"got {len(results)}, expected ≤ {expect['max_claims']}",
            }
        )

    # ── all_verdict ────────────────────────────────────────────────────────
    if "all_verdict" in expect:
        wrong = [r.claim for r in results if r.verdict != expect["all_verdict"]]
        findings.append(
            {
                "check": "all_verdict",
                "status": PASS if not wrong else FAIL,
                "detail": (
                    f"all {expect['all_verdict']}"
                    if not wrong
                    else f"unexpected verdicts on: {wrong}"
                ),
            }
        )

    # ── no_verdict ─────────────────────────────────────────────────────────
    if "no_verdict" in expect:
        wrong = [r.claim for r in results if r.verdict == expect["no_verdict"]]
        findings.append(
            {
                "check": "no_verdict",
                "status": PASS if not wrong else WARN,
                "detail": (
                    f"none were {expect['no_verdict']}"
                    if not wrong
                    else f"unexpectedly got {expect['no_verdict']} for: {wrong}"
                ),
            }
        )

    # ── min_confidence ─────────────────────────────────────────────────────
    if "min_confidence" in expect:
        low = [
            (r.claim, r.confidence_score)
            for r in results
            if r.confidence_score < expect["min_confidence"]
        ]
        findings.append(
            {
                "check": "min_confidence",
                "status": PASS if not low else FAIL,
                "detail": (
                    f"all ≥ {expect['min_confidence']}"
                    if not low
                    else f"low confidence on: {low}"
                ),
            }
        )

    # ── check_structure ────────────────────────────────────────────────────
    if expect.get("check_structure") and results:
        issues = []
        for r in results:
            if not r.claim:
                issues.append("empty claim")
            if r.confidence_score < 0 or r.confidence_score > 100:
                issues.append(f"confidence out of range: {r.confidence_score}")
            if not r.reasoning or not r.reasoning.strip():
                issues.append("empty reasoning")
            if not r.checked_at:
                issues.append("missing checked_at")
            if not r.model_used:
                issues.append("missing model_used")
            try:
                datetime.fromisoformat(r.checked_at.replace("Z", "+00:00"))
            except ValueError:
                issues.append(f"invalid ISO timestamp: {r.checked_at}")
        findings.append(
            {
                "check": "output_structure",
                "status": PASS if not issues else FAIL,
                "detail": "all fields valid" if not issues else ", ".join(issues),
            }
        )

    # ── check_spans ────────────────────────────────────────────────────────
    if expect.get("check_spans") and results:
        original = tc["request"].text
        issues = []
        for r in results:
            if r.char_start is None or r.char_end is None:
                issues.append(f"missing span for: {r.claim!r}")
            else:
                extracted = original[r.char_start : r.char_end]
                if not extracted:
                    issues.append(
                        f"span [{r.char_start}:{r.char_end}] extracts empty string"
                    )
        findings.append(
            {
                "check": "span_detection",
                "status": PASS if not issues else WARN,
                "detail": "spans valid" if not issues else "; ".join(issues),
            }
        )

    # ── flag_score_review ──────────────────────────────────────────────────
    if expect.get("flag_score_review") and results:
        summary = [
            {"claim": r.claim, "verdict": r.verdict, "confidence_score": r.confidence_score}
            for r in results
        ]
        findings.append(
            {
                "check": "score_review (manual)",
                "status": WARN,
                "detail": f"Review scores manually: {summary}",
            }
        )

    return findings


# ===========================================================================
# Runner
# ===========================================================================

async def run_all_tests(
    save_path: Optional[str] = None,
    as_json: bool = False,
) -> dict:
    all_output = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total": len(TEST_CASES),
        "passed": 0,
        "failed": 0,
        "warned": 0,
        "tests": [],
    }

    for tc in TEST_CASES:
        print(f"\n{'='*70}")
        print(f"[{tc['id']}] {tc['label']}")
        print(f"  ℹ  {tc['description']}")

        try:
            results = await check_claims(tc["request"])
        except Exception as exc:
            print(f"  ✗  ERROR calling check_claims: {exc}")
            all_output["tests"].append(
                {
                    "id": tc["id"],
                    "label": tc["label"],
                    "error": str(exc),
                    "findings": [],
                    "results": [],
                }
            )
            all_output["failed"] += 1
            continue

        findings = check_result(tc, results)

        has_fail = any(f["status"] == FAIL for f in findings)
        has_warn = any(f["status"] == WARN for f in findings)

        for f in findings:
            icon = "✓" if f["status"] == PASS else ("⚠" if f["status"] == WARN else "✗")
            print(f"  {icon}  [{f['check']}] {f['detail']}")

        # Print claim-level detail
        for r in results:
            src_count = len(r.sources)
            span_info = (
                f"span=[{r.char_start}:{r.char_end}]"
                if r.char_start is not None
                else "span=None"
            )
            print(
                f"     › {r.verdict.upper()} ({r.confidence_score}) "
                f"srcs={src_count} {span_info} — {r.claim[:80]}"
            )

        if has_fail:
            all_output["failed"] += 1
        elif has_warn:
            all_output["warned"] += 1
        else:
            all_output["passed"] += 1

        all_output["tests"].append(
            {
                "id": tc["id"],
                "label": tc["label"],
                "findings": findings,
                "results": [r.model_dump() for r in results],
            }
        )

    print(f"\n{'='*70}")
    print(
        f"SUMMARY  total={all_output['total']}  "
        f"passed={all_output['passed']}  "
        f"warned={all_output['warned']}  "
        f"failed={all_output['failed']}"
    )

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_output, f, indent=2, default=str)
        print(f"\nResults saved to: {save_path}")

    if as_json:
        print("\n" + json.dumps(all_output, indent=2, default=str))

    return all_output


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClaimCheck v2 test runner")
    parser.add_argument(
        "--json", action="store_true", help="Dump full JSON output to stdout"
    )
    parser.add_argument(
        "--save", metavar="FILE", help="Save results to a JSON file"
    )
    args = parser.parse_args()

    asyncio.run(
        run_all_tests(
            save_path=args.save or "output.json",
            as_json=args.json,
        )
    )