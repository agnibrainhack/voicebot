# turnval: turn-validator dataset and training

`turnval` builds a labelled dataset of Amex Account Protection bot turns. Each row is a step summary, the customer's utterance, the bot's response and a label: 1 if the response is admissible under the AOP, 0 if not. It then trains gpt-oss-20b (LoRA, contrastive) to classify those turns and reports results overall and per category. It covers User Identification and Charge Verification, each validated as its own AOP.

```
design (factors.yaml) → covering array → simulate the correct call (flow + HRT graph)
  → inject one error, continue 2 turns → write the words (gpt-oss / gateway) → rows.csv
  → stratified split → train.csv / val.csv / test.csv → LoRA + contrastive training → reports
```

## Inputs (kept out of git)

| File | What it is |
|---|---|
| `turnval/data/aop_flow.json` | The AOP flow: sections, then nodes (USE, IF, WAIT UNTIL, speech, GOTO, SWITCH…) and edges |
| `turnval/data/hrt_graph.json` | The HRT graph: golden boxes, then branches, then MUST_CONVEY / MUST_NOT_CONVEY / METADATA_CHECK / LOCAL_GUARDRAIL, plus the global policies |

The HRT-graph export has a known fault: 12 condition strings lose a quote. The loader (`graph.load_json`) repairs this.

## Blocks

| Block | Module | What it does | Check |
|---|---|---|---|
| 1. Graph | `graph.py` | `Walker` re-enters the AOP each customer turn (UI starts at Routing, which follows `checkpoint`; CV starts at Fraud Charges Routing). It follows the tools and IFs to the first speech, and collects every speaking box up to the customer's answer as one response (complex states). Speech already given and answered is skipped. `HRTGraph.resolve` returns the branches and global policies that apply, as atomic HRTs. | Replays the real call: UI_108 → UI_119 → UI_250 → UI_250 |
| 2. Factors | `factors.py`, `config/factors.yaml` | 42 journey factors: every flow decision (the audit shows each flow outcome is driven by some level), data shape, customer behaviour, and the error. `when` gives relevance and `constraints` exclude impossible rows. Greedy covering array: every feasible pair (9k+), plus 3-way coverage for (aop × error family × position / interjection / register). | `python -m turnval.factors audit` → `{}` |
| 3. Simulator | `scenario.py` | Outcome first. A row fixes each tool's outcomes; the customer's intent is derived from them; the walker gives the state the bot must be in. Also synthesises the profile, cards and transactions, OTP and verification questions. | 48 states reached; no failures over the design |
| 4. Errors | `errors.py` | 15 error families (transition, hand-off, content, metadata, order, local/global guardrail, context, wrapper misread, hallucination). Each is placed early, middle or late, is subtle or blatant, and either recovers or stays off for two more turns. Adds the correct counterpart at the error turn, plus side negatives. | The fallback family is recorded as `family_effective` |
| 5. Text | `textgen.py` | Customer, bot (correct or flawed), step summary, and an optional LLM quality gate. Uses `gmsjourney.llm`: gpt-oss on vLLM (`backend: local`) or the Amex gateway, with the cache and pacer. `TemplateRenderer` runs without a model. | `tests/test_turnval.py` fake-model test |
| 6. Dataset | `generate.py` | Builds conversations in parallel → `conversations.jsonl`, `rows.csv`, `generation_report.json`. Rows sharing a context (`group_id`) form one contrastive group. Out-of-place negatives are added (a correct response from another state). | — |
| 7. Split | `split.py` | Iterative stratification plus swap refinement, over every factor level, error family, AOP and length band → `train/val/test.csv` and `split_report.json` (per-level shares and maximum deviation). | `max_deviation` in the report |
| 8. Training | `train/` | gpt-oss-20b (MXFP4 dequantised to bf16) with LoRA on attention (experts optional). Score head on the last token plus a projection. Loss = BCE + pairwise margin within each context group + supervised contrastive (classes: correct / each error family). Keeps the best checkpoint on val PR-AUC, fits a temperature on val, and sets thresholds on val. | Tiny-model run in tests |
| 9. Evaluation | `evaluate.py`, `logs.py` | Overall metrics: accuracy, balanced accuracy, MCC, error and pass precision/recall/F1, ROC-AUC, PR-AUC, Brier, ECE. Pairwise accuracy per error family. Thresholds where flags or passes are right ≥ 95% of the time (with a Wilson interval). Per category: error family, operator, AOP, state, phase, variant, position, length, and every journey factor. `logs.py` turns real logs into rows to hand-label. | — |

## Run

```bash
# 0. data files in turnval/data/ ; a gpt-oss server for the text
vllm serve openai/gpt-oss-20b --port 8000            # or set text.backend: gateway (Amex gateway, gcloud token)

# 1-2. look at the design (optional)
python -m turnval.factors audit
python -m turnval.factors design --out /tmp/design.csv --replicates 3

# 3-7. dataset: a 20-conversation pilot first, then everything
python -m turnval.generate --limit 20 --out turnval/out/pilot      # read conversations.jsonl before scaling up
python -m turnval.generate                                         # config: turnval/config/default.yaml
#    dry run without a model: --renderer template

# 8-9. train and report (GPU host: pip install -r turnval/requirements-train.txt)
python -m turnval.train.trainer --config turnval/train/config.yaml
#    overrides: --set train.epochs=2 loss.supcon=0 model.head=lm_token

# real test set: logs -> rows to label -> add a label column (1/0) -> data.real_test in train/config.yaml
python -m turnval.logs --in call1.json call2.txt --out turnval/out/real_to_label.csv --renderer llm
python -m turnval.evaluate --pred turnval/runs/gptoss20b_lora/real_test_predictions.csv --out turnval/runs/gptoss20b_lora/real_report \
       --thresholds turnval/runs/gptoss20b_lora/thresholds.json
```

The `vllm_url` is the server's `/v1` base (e.g. `http://host:8000/v1`); `GMSJOURNEY_VLLM_URL` also works.

## Knobs

| Where | Knob | Effect |
|---|---|---|
| `factors.yaml` | levels, `when`, `constraints`, `strength`, `triples` | What the covering array guarantees |
| `default.yaml: design` | `replicates`, `extra_random` | Dataset size. The array is about 480 rows; 3 replicates give about 1,500 conversations, 12k rows, ~4k contrastive groups |
| `default.yaml: simulation` | `side_negatives_per_conversation`, `out_of_place_per_conversation`, `include_handoff_turns` | Negatives per context |
| `default.yaml: text` | temperatures, `verify` (quality gate), `max_tries`, `reasoning_effort`, `mask_customer_digits`, `workers` | Text quality, cost and speed |
| `default.yaml: split` | `ratios`, `group_by` (`conv_id`: best balance; `design_id`: replicates kept together, stricter) | — |
| `train/config.yaml` | LoRA r / targets, `head`, loss weights, `supcon_label`, `max_length`, lr, epochs, `precision_target` | — |

## Rules

- **Labels follow the AOP, not ideal service.** A turn is correct when it is in the state the flow requires and says what that state's HRTs require.
- **Turns after an error are judged against the correct path.** A bot that returns to the correct state is labelled correct.
- **No retry limits are imposed.** Re-asks happen as the wrapper tools' outcomes dictate.
- **Hand-off turns** (transfer, call end, card replacement) are validated with two generic HRTs: say the caller is being transferred, and do not carry on the flow. Their words belong to another AOP.

## Hardware

- **Generation:** one gpt-oss-20b vLLM server (16 GB+ GPU memory with MXFP4).
- **Training:** gpt-oss-20b in bf16 is about 42 GB of weights. LoRA with gradient checkpointing at `max_length` 1536 fits one 80 GB GPU, or two 40–48 GB GPUs with `device_map: auto`.
