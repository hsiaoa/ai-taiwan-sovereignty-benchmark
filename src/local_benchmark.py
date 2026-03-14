#!/usr/bin/env python3
"""
Taiwan Sovereignty Benchmark - Local Model Runner
Runs the quick test against a local OpenAI-compatible endpoint (e.g., mlx_lm.server).
"""

import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

# Reuse the benchmark logic
from bedrock_benchmark import (
    QUICK_TEST_PROMPTS,
    GLOBAL_INSTANT_FAIL_KEYWORDS,
    HISTORICAL_EXCEPTIONS,
    BedrockBenchmark,
)

LOCAL_ENDPOINT = "http://localhost:8078/v1/chat/completions"


def invoke_local_model(prompt: str, endpoint: str = LOCAL_ENDPOINT) -> dict:
    """Invoke a local OpenAI-compatible model."""
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=360) as resp:
            data = json.loads(resp.read())
        text = data["choices"][0]["message"]["content"]
        return {"success": True, "text": text, "raw_response": data}
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_benchmark(endpoint: str = LOCAL_ENDPOINT, language: str = "zh"):
    """Run 5-question quick test against local model."""
    # Use BedrockBenchmark just for its check_red_flags method
    checker = BedrockBenchmark.__new__(BedrockBenchmark)

    model_name = "GLM-5 (local, 4-bit)"
    results = {
        "model": model_name,
        "endpoint": endpoint,
        "timestamp": datetime.now().isoformat(),
        "language": language,
        "questions": [],
        "overall_passed": True,
        "failed_at": None,
        "total_warnings": [],
    }

    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"Endpoint: {endpoint}")
    print(f"{'='*60}")

    for pc in QUICK_TEST_PROMPTS:
        prompt = pc.get(f"prompt_{language}", pc["prompt_zh"])
        print(f"\n[{pc['id']}] {pc['category']}")
        print(f"Prompt: {prompt}")

        response = invoke_local_model(prompt, endpoint)

        if not response["success"]:
            qr = {
                "id": pc["id"],
                "category": pc["category"],
                "prompt": prompt,
                "success": False,
                "error": response["error"],
                "passed": False,
            }
            results["questions"].append(qr)
            print(f"  ERROR: {response['error']}")
            results["overall_passed"] = False
            if results["failed_at"] is None:
                results["failed_at"] = pc["id"]
            continue

        text = response["text"]
        flags = checker.check_red_flags(text, pc)

        qr = {
            "id": pc["id"],
            "category": pc["category"],
            "prompt": prompt,
            "success": True,
            "response": text,
            "passed": flags["passed"],
            "instant_fails": flags["instant_fails"],
            "warnings": flags["warnings"],
        }
        results["questions"].append(qr)

        if flags["passed"]:
            if flags["warnings"]:
                print(f"  PASS (warnings: {flags['warnings']})")
                results["total_warnings"].extend(flags["warnings"])
            else:
                print(f"  PASS")
        else:
            print(f"  FAIL - Red flags: {flags['instant_fails']}")
            results["overall_passed"] = False
            if results["failed_at"] is None:
                results["failed_at"] = pc["id"]

        preview = text[:300].replace("\n", " ")
        print(f"  Response: {preview}...")

    # Summary
    print(f"\n{'='*60}")
    if results["overall_passed"]:
        if results["total_warnings"]:
            verdict = "PASS_WITH_WARNINGS"
            results["recommendation"] = "PROCEED_WITH_CAUTION"
        else:
            verdict = "PASS"
            results["recommendation"] = "PROCEED_TO_FULL_TEST"
    else:
        verdict = f"FAIL (failed at {results['failed_at']})"
        results["recommendation"] = "NOT_SUITABLE_FOR_TAIWAN"

    print(f"Result: {verdict}")
    print(f"Recommendation: {results['recommendation']}")
    print(f"{'='*60}")

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "raw").mkdir(exist_ok=True)
    (results_dir / "scores").mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_file = results_dir / "raw" / f"glm5-local_{ts}.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved: {raw_file}")

    return results


if __name__ == "__main__":
    lang = "zh"
    endpoint = LOCAL_ENDPOINT
    for arg in sys.argv[1:]:
        if arg in ("en", "zh"):
            lang = arg
        elif arg.startswith("http"):
            endpoint = arg
    run_benchmark(endpoint, lang)
