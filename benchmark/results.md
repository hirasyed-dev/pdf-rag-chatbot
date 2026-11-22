# Benchmark results

Document: AI_PDF_Chatbot.pdf  ·  Questions: 20  ·  2026-11-22 22:20

## Results against the acceptance criteria

| Criterion | Target | Result | |
|---|---|---|---|
| Answers citing the exact page (15 present questions) | at least 85% | 15/15 (100%) | PASS |
| Individual citations on a correct page | for information | 18/20 (90%) | |
| Unanswerable questions give 'Not found in document' | at least 4 of 5 | 5/5 | PASS |
| Vector search time (slowest) | under 2 s | 264 ms | PASS |
| First answer token (median / slowest) | under 5 s | 2.6 s / 8.6 s | PASS |
| Requests that failed | 0 | 0 | PASS |

## Every question

| # | Question | Result | Expected | Cited |
|---|---|---|---|---|
| 1 | What is the main goal of the internship project? | OK | pages [1] | cited [1] |
| 2 | What is the maximum file size and number of pages for uploaded PDFs? | OK | pages [1, 2] | cited [2, 1] |
| 3 | Why is Ctrl+F not enough for finding information in a PDF? | OK | pages [1] | cited [1] |
| 4 | What problems come from sending a whole PDF directly to an LLM? | OK | pages [1] | cited [1] |
| 5 | Which library extracts text from the PDFs? | OK | pages [1, 3] | cited [3, 1] |
| 6 | Are scanned or image-only PDFs supported? | OK | pages [2] | cited [2] |
| 7 | What does upload validation have to check? | OK | pages [2] | cited [2] |
| 8 | What should the system answer when the document does not contain the information? | OK | pages [2] | cited [2, 4] |
| 9 | Which features are listed as should-have enhancements? | OK | pages [2] | cited [2] |
| 10 | Which bonus features are listed as could-have? | OK | pages [2] | cited [2] |
| 11 | Which database tracks the uploaded documents and their status? | OK | pages [3] | cited [3] |
| 12 | Which framework is used for the user interface? | OK | pages [2, 3] | cited [3, 2, 4] |
| 13 | What was built in Phase 2 of the timeline? | OK | pages [4] | cited [4] |
| 14 | How do citations get their exact page numbers in Phase 3? | OK | pages [4] | cited [4] |
| 15 | How long may processing a 50-page PDF take? | OK | pages [4] | cited [4] |
| 16 | Who is the CEO of Google? | OK | not in PDF | Not found |
| 17 | What is the monthly stipend of the intern? | OK | not in PDF | Not found |
| 18 | When was the Eiffel Tower built? | OK | not in PDF | Not found |
| 19 | What was the total budget of the project in dollars? | OK | not in PDF | Not found |
| 20 | Which mobile operating systems does the app support? | OK | not in PDF | Not found |
