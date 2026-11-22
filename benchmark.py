"""20-question benchmark for the AI PDF Chatbot (spec section 7).

Start the backend first (uvicorn app.main:app), then run:
    python benchmark.py                 # full benchmark (needs the Gemini key in .env)
    python benchmark.py --search-only   # no LLM: compares semantic vs hybrid search
    python benchmark.py --upload-test   # times the upload of a generated 50-page PDF
    python benchmark.py --quick         # only 5 questions, for a fast speed check
    python benchmark.py --only 18,20    # only these question numbers (saves free quota)

It uploads benchmark/AI_PDF_Chatbot.pdf (skipped if already indexed), asks every
question about that document only, and writes benchmark/results.md and results.json.
Questions with an empty expected_pages list are the ones the PDF cannot answer."""
import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent / "benchmark"
NOT_FOUND = "Not found in document."


def upload(api: str, pdf: Path) -> str:
    with open(pdf, "rb") as f:
        r = httpx.post(f"{api}/documents", files={"file": (pdf.name, f, "application/pdf")}, timeout=900)
    r.raise_for_status()
    return r.json()["doc_id"]


def ask_streaming(api: str, question: str, doc_id: str) -> dict:
    """Ask through /ask/stream and measure the time until the first token arrives."""
    started = time.perf_counter()
    first_token = None
    payload = {"question": question, "doc_ids": [doc_id]}
    with httpx.stream("POST", f"{api}/ask/stream", json=payload, timeout=180) as response:
        if response.status_code != 200:
            response.read()
            raise RuntimeError(response.text[:200])
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            event = json.loads(line[5:])
            if event["type"] == "token" and first_token is None:
                first_token = time.perf_counter() - started
            elif event["type"] == "error":
                raise RuntimeError(event["detail"])
            elif event["type"] == "final":
                event["first_token_s"] = first_token
                return event
    raise RuntimeError("stream ended without a final answer")


def search(api: str, question: str, doc_id: str, mode: str) -> dict:
    r = httpx.get(f"{api}/search", params={"q": question, "doc_id": doc_id, "top_k": 5, "mode": mode}, timeout=60)
    r.raise_for_status()
    return r.json()


def pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else "n/a"


def run_search_only(api, doc_id, questions) -> list[str]:
    present = [q for q in questions if q["expected_pages"]]
    lines = ["## Retrieval comparison (no LLM): is the right page in the top 5?", "",
             "| Mode | Top-1 correct | Top-5 contains right page | Slowest search |", "|---|---|---|---|"]
    for mode in ("semantic", "hybrid"):
        top1 = top5 = 0
        slowest = 0
        for item in present:
            result = search(api, item["q"], doc_id, mode)
            pages = [h["page_number"] for h in result["results"]]
            top1 += bool(pages) and pages[0] in item["expected_pages"]
            top5 += any(p in item["expected_pages"] for p in pages)
            slowest = max(slowest, result["took_ms"])
        lines.append(f"| {mode} | {pct(top1, len(present))} | {pct(top5, len(present))} | {slowest} ms |")
    return lines


def warm_up(api: str, doc_id: str) -> None:
    """The very first search loads the embedding model (slow, once per backend start).
    The timing criterion is about normal operation, so this happens before measuring."""
    try:
        search(api, "warm up", doc_id, "hybrid")
    except httpx.HTTPError:
        pass


def run_full(api, doc_id, questions, delay) -> tuple[list[str], dict]:
    rows, errors, limit_hits = [], 0, 0
    present_total = present_answered = question_ok = 0
    citations_total = citations_ok = 0
    absent_total = absent_ok = 0
    first_tokens, search_ms = [], []

    for position, item in enumerate(questions, start=1):
        number = item.get("n", position)
        expected = item["expected_pages"]
        result = None
        for attempt in (1, 2):
            try:
                result = ask_streaming(api, item["q"], doc_id)
                limit_hits = 0
                break
            except Exception as e:  # free-tier rate limit: wait once and try again
                if "limit" in str(e).lower():
                    limit_hits += 1
                if attempt == 2:
                    print(f"  Q{number} failed: {e}")
                else:
                    print(f"  Q{number}: {str(e)[:300]}\n  ...waiting 40 s and retrying")
                    time.sleep(40)
        if limit_hits >= 2:
            print("\nGemini keeps refusing (limit reached). This is probably your free daily quota "
                  "for this model.\nStopping now. Set another LLM_MODEL in .env (for example "
                  "gemini-3.5-flash) or try again tomorrow.")
            return None, None
        if result is None:
            errors += 1
            rows.append(f"| {number} | {item['q']} | ERROR | | |")
            continue

        cited = [c["page_number"] for c in result["citations"]]
        if result.get("first_token_s") is not None:
            first_tokens.append(result["first_token_s"])
        search_ms.append(result["timings_ms"].get("retrieval", 0))

        if expected:
            present_total += 1
            present_answered += result["found"]
            right = [p for p in cited if p in expected]
            citations_total += len(cited)
            citations_ok += len(right)
            ok = bool(right)
            question_ok += ok
            verdict = "OK" if ok else ("WRONG PAGE" if result["found"] else "MISSED")
            why = "" if ok else f" ({result.get('reason')})"
            rows.append(f"| {number} | {item['q']} | {verdict}{why} | pages {expected} | cited {cited} |")
        else:
            absent_total += 1
            ok = not result["found"]
            absent_ok += ok
            rows.append(f"| {number} | {item['q']} | {'OK' if ok else 'HALLUCINATED'} | not in PDF | "
                        f"{'Not found' if ok else 'answered: ' + result['answer'][:60]} |")
        time.sleep(delay)

    first = statistics.median(first_tokens) if first_tokens else None
    worst_first = max(first_tokens) if first_tokens else None
    summary = {
        "answered_with_right_page": (question_ok, present_total),
        "citation_precision": (citations_ok, citations_total),
        "absent_rejected": (absent_ok, absent_total),
        "median_first_token_s": first,
        "slowest_first_token_s": worst_first,
        "slowest_retrieval_ms": max(search_ms) if search_ms else None,
        "errors": errors,
    }

    def mark(passed):
        return "PASS" if passed else "FAIL"

    accuracy_ok = present_total and question_ok / present_total >= 0.85
    absent_pass = absent_ok >= 4 if absent_total == 5 else (absent_total and absent_ok / absent_total >= 0.8)
    lines = [
        "## Results against the acceptance criteria", "",
        "| Criterion | Target | Result | |", "|---|---|---|---|",
        f"| Answers citing the exact page (15 present questions) | at least 85% | {pct(question_ok, present_total)} | {mark(accuracy_ok)} |",
        f"| Individual citations on a correct page | for information | {pct(citations_ok, citations_total)} | |",
        f"| Unanswerable questions give 'Not found in document' | at least 4 of 5 | {absent_ok}/{absent_total} | {mark(absent_pass)} |",
        f"| Vector search time (slowest) | under 2 s | {summary['slowest_retrieval_ms']} ms | {mark((summary['slowest_retrieval_ms'] or 0) < 2000)} |",
        f"| First answer token (median / slowest) | under 5 s | "
        + (f"{first:.1f} s / {worst_first:.1f} s" if first is not None else "n/a")
        + f" | {mark(first is not None and first < 5)} |",
        f"| Requests that failed | 0 | {errors} | {mark(errors == 0)} |",
        "", "## Every question", "",
        "| # | Question | Result | Expected | Cited |", "|---|---|---|---|---|", *rows,
    ]
    return lines, summary


def make_50_page_pdf(path: Path) -> None:
    import pymupdf

    rng = random.Random(1)
    words = ("policy report budget quarter customer invoice contract delivery supplier "
             "warranty schedule approval review compliance training safety maintenance "
             "inventory shipment forecast revenue audit process team manager").split()
    doc = pymupdf.open()
    for page_number in range(1, 51):
        text = f"Section {page_number}. " + " ".join(rng.choice(words) for _ in range(380))
        doc.new_page().insert_textbox((40, 40, 555, 800), text, fontsize=10)
    doc.save(str(path))
    doc.close()


def run_upload_test(api) -> list[str]:
    path = HERE / "generated_50_pages.pdf"
    make_50_page_pdf(path)
    try:
        started = time.perf_counter()
        doc_id = upload(api, path)
        seconds = time.perf_counter() - started
        httpx.delete(f"{api}/documents/{doc_id}", timeout=60)
    finally:
        path.unlink(missing_ok=True)
    verdict = "PASS" if seconds < 60 else "FAIL"
    return ["## Upload speed", "", "| Criterion | Target | Result | |", "|---|---|---|---|",
            f"| 50-page PDF processed | under 60 s | {seconds:.1f} s | {verdict} |"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default=os.getenv("API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--search-only", action="store_true", help="skip the LLM, compare search modes")
    parser.add_argument("--upload-test", action="store_true", help="time a generated 50-page PDF")
    parser.add_argument("--quick", action="store_true", help="only 5 questions (fast latency check)")
    parser.add_argument("--only", help="only these question numbers, e.g. --only 18,20 (saves free quota)")
    parser.add_argument("--delay", type=float, default=7.0, help="seconds between questions (free-tier limits)")
    args = parser.parse_args()

    try:
        httpx.get(f"{args.api}/health", timeout=5).raise_for_status()
    except httpx.HTTPError:
        print("Backend is not running. Start it first: uvicorn app.main:app --reload")
        return 1

    spec = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    questions = [dict(q, n=i) for i, q in enumerate(spec["questions"], start=1)]
    if args.quick:
        questions = [questions[i] for i in (0, 5, 10, 12, 15)]
    if args.only:
        wanted = {int(x) for x in args.only.split(",") if x.strip()}
        questions = [q for q in questions if q["n"] in wanted]
    report = ["# Benchmark results", "", f"Document: {spec['document']}  ·  Questions: {len(questions)}  ·  "
              f"{time.strftime('%Y-%m-%d %H:%M')}", ""]

    if args.upload_test:
        report += run_upload_test(args.api)
    else:
        print("Uploading the benchmark PDF (skipped if already indexed)...")
        doc_id = upload(args.api, HERE / spec["document"])
        if args.search_only:
            report += run_search_only(args.api, doc_id, questions)
        else:
            print(f"Asking {len(questions)} questions, this takes a few minutes...")
            warm_up(args.api, doc_id)
            lines, summary = run_full(args.api, doc_id, questions, args.delay)
            if lines is None:
                return 2
            report += lines
            (HERE / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    text = "\n".join(report)
    print("\n" + text)
    (HERE / "results.md").write_text(text + "\n", encoding="utf-8")
    print("\nSaved to benchmark/results.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
