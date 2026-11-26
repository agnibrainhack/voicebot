# Turn validator for the Amex Account Protection voice bot

This repository builds the dataset for the third validator, V3, and trains it. V3 is a classifier that reads three things:
- a step summary of the call so far;
- the customer's latest turn;
- the bot's response.

It decides whether that response is admissible under the AOP (1) or not (0). It covers User Identification and Charge Verification, each validated as its own AOP.

The labels come from the AOP itself:
1. The AOP flow and the HRT graph give the state the bot must be in after each customer turn, and what that state must and must not say.
2. A covering array over the journey factors decides which calls are simulated.
3. One bot error is injected per call, and the call continues for two more turns.
4. gpt-oss-20b, on vLLM or through the Microsoft gateway, writes the words.

gpt-oss-20b is then trained with LoRA and a classification head, on BCE plus a pairwise margin within each context plus supervised contrastive loss. The report covers results overall and per error type, state and journey factor.

| Where | What |
|---|---|
| `turnval/README.md` | the nine blocks, commands, knobs and rules |
| `turnval/notebooks/gpt_oss_vllm_server.ipynb` | start gpt-oss-20b on vLLM (the text model) |
| `turnval/notebooks/turnval_end_to_end.ipynb` | the whole run: data check → design → pilot → dataset → split → GPU plan → training → reports → real logs |
| `turnval/data/` | `aop_flow.json` and `hrt_graph.json` (client material, not in git) |

```bash
pip install -r requirements.txt                       # generation
pip install -r turnval/requirements-train.txt         # training (GPU host)
python -m pytest -q tests                             # needs turnval/data/*.json
TURNVAL_SMOKE=1 jupyter nbconvert --to notebook --execute turnval/notebooks/turnval_end_to_end.ipynb   # no GPU / server
```
