# `gmsjourney/s09_reports/` — Reports

Reading a campaign: the HTML report, case-by-case pages and the one-page verdict.

The full explanation, with the theory and diagrams, is chapter 9 of the guide: [docs/guide/09-reports.md](../../docs/guide/09-reports.md).

```
gmsjourney/s09_reports/
├── __init__.py  Reports: reading a campaign's results
├── answer_sheet.py  The answer sheet: every generated case and what a correct bot must do, written after step 5
├── cases.py     Case by case: what happened in each run, expected beside actual
├── report.py    The campaign report: one self-contained HTML file (light and dark themes)
└── verdict.py   The one-page verdict of a finished run: is the bot ready, what are the risks, can anyone reproduce it, and, case
```

## `__init__.py`

Reports: reading a campaign's results.

## `answer_sheet.py`

The answer sheet of a test run: every generated case and what a correct bot must do, written right after the cases
and their rewordings are proved (campaign step 5, before any bot is run).

| name | kind | what it is |
|---|---|---|
| `write_answer_sheet` | def | Write ``answer_sheet.html`` in a run folder: each case turn by turn (the customer's words and meaning, what a correct bot must do, the state's rules), how the call must end, and every reworded version with its proof verdict. |

## `cases.py`

Case by case: what happened in each run, expected beside actual.

Parts:

* PART 1 - Plain-English case summaries for the answer sheet: what happened, turn by turn, and why the call ended
* PART 2 - Turn-by-turn: what the oracle says the bot should do, next to what the bot did.
* PART 3 - Expected vs actual, case by case: one HTML page per campaign, in the answer sheet's format.
* PART 4 - Look inside a base (seed) case and any of its enriched variants, turn by turn.

| name | kind | what it is |
|---|---|---|
| `graph_edge` | def | The policy graph's leads_to triple for one hop with the same states AND trigger, or None (a hop whose trigger differs from every graph transition between those states is not passed off as one of them). |
| `turn_hops` | def | Split a turn's ordered steps (("act", act) / ("tool", event)) into dialog-state hops: {src, on, tools, dst, act}. The customer's meaning triggers the first hop; the bot's own moves are 'auto'. |
| `tool_line` | def | One tool call as a line: ``tool(arg=value, ...) -> status``. |
| `case_facts` | def | turns: [{turn, state, customer_says, customer_means, steps:[(kind, x)], bot_says:[str]}] -> fact sheet. |
| `template_summary` | def | Deterministic account from the fact sheet (no model). |
| `LLMNarrator` | class | Model-written summaries from the fact sheet, checked against it; falls back to the template. |
| `describe_act` | def | A bot act as a short readable string: PRESENT_TXN(T2: Starbucks $6.25). |
| `describe_tool` | def | One tool call as a line, keeping only its key arguments: ``tool(key=value) -> status``. |
| `normalize_acts` | def | The decision content of a bot reply: (act, txn, reason) in order, without optional acknowledgements, without a re-presentation of the same charge right after a clarification / context / policy answer, and without immediate repeats. |
| `segment` | def | Cut a run into (opening bot acts, per customer turn: prompt state, customer, tools, bot acts/text). |
| `speech_by_turn` | def | What the speech check said about the bot's words: (opening, one entry per customer turn). |
| `turn_by_turn` | def | Expected (oracle) vs actual (bot) for one scenario, one row per customer turn. |
| `show_turns` | def | Print one case turn by turn (expected vs actual tools, bot acts, whether words were grounded) and return the table and its summary. |
| `export_turn_by_turn` | def | dataset/turn_by_turn.csv (one row per turn per scenario) + turn_by_turn_summary.csv. |
| `judges_block` | def | The two LLM judges beside the rules' verdict, for a case page. |
| `read_judges` | def | dataset/llm_judges.jsonl as {case_id: judges}; empty when the run had no LLM judges. |
| `write_case_comparison` | def | Write ``case_comparison.html`` in a run folder: every case side by side with the expected and actual turns, from the run's dataset files. |
| `variants_of` | def | The reworded versions of one base case. |
| `pick` | def | (seed item, enriched scenario) by base qid (e.g. 'J015', 'kb-exact_recall-01') or index, and variant index. |
| `compare_case` | def | Seed vs enriched, turn by turn, in canonical form. |
| `frame_hrt` | def | A customer turn as triples: (turn, in_state, state), (turn, has_act, act), (turn, has_<slot>, value). The card is canonicalised in its dialog state (a bare yes/no read the way the state reads it). |
| `case_hrt` | def | Head / Relation / Tail for the seed turn and the enriched turn. |
| `show_case` | def | Print the seed/enriched comparison and the H/R/T tables for one selection. |

## `report.py`

The campaign report: one self-contained HTML file (light and dark themes).

Parts:

* PART 1 - Self-contained HTML campaign report (one file, light and dark themes).
* PART 2 - The HTML report of a campaign on a compact policy graph (one self-contained file).

| name | kind | what it is |
|---|---|---|
| `esc` | def | HTML-escaped text. |
| `pct` | def | A fraction as a percentage with one decimal ("–" for none). |
| `tile` | def | One headline tile: a big value over a small label. |
| `table` | def | An HTML table from a header and rows (columns in ``num_cols`` right-aligned). |
| `hbar` | def | Horizontal single-series bar chart: per-bar hover titles, direct labels placed clear of any interval whisker, round tick values, and no upscaling of text on wide screens. |
| `pfmt` | def | A p-value for display (scientific below 0.001). |
| `render_html` | def | The charge-review campaign report (HTML) from the campaign's results dict. |
| `failure_reasons` | def | Plain-English reasons a run failed, read off its graded row (nothing re-judged here). |
| `render_graph_html` | def | The campaign report (HTML) for a pack whose cases were mined from its policy graph. |

## `verdict.py`

The one-page verdict of a finished run: is the bot ready, what are the risks, can anyone reproduce it, and, case

| name | kind | what it is |
|---|---|---|
| `provenance` | def | What a run can be reproduced from, taken when it starts: the code commit (and whether the working tree had uncommitted edits) and the hash of the journey's graph file. |
| `build` | def | The verdict of a finished run: ``path`` is a console run folder (with job.json and out/) or a campaign's output folder (out_dir, with report.json). |
| `render_html` | def | The verdict as a standalone page (out/verdict.html): page one is the one-pager, every case follows. |
| `write` | def | Build the verdict of a finished campaign and write out/verdict.json and out/verdict.html; returns a summary. |
