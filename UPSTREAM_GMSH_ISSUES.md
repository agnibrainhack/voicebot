# Issues to raise with knowlytix (gmsh 1.5.0 harness)

Found while comparing this framework with `knowlytix/gmsh` at commit `2920587` ("Release gmsh v1.5.0").
Only the Apache-2.0 packages were read (`knowlytix-harness`, `knowlytix-benchmark`, `knowlytix-knowledge`).
knowlytix-core was not opened.

How to confirm each issue before filing:

```
python tools/gmsh_upstream_repro.py                     # against the installed knowlytix
python tools/gmsh_upstream_repro.py --root ~/src/gmsh   # against a checkout
```

The script reads the source files as text. For each issue it reports whether the construct
that causes the defect is still present. Where the open module imports cleanly, it also runs
a short demonstration. **File only the issues it reports as `present`.**

All 15 constructs were present at `2920587`, and each was read in the source by hand. The effects
listed for G12, G13 and G15 (zero questions, zero counts) were inferred from the code; they were
not executed.

| id | where (gmsh 1.5.0) | defect | effect | suggested fix |
|---|---|---|---|---|
| G1 | `harness/testing/oracles.py:374` | `ExactOutcomeOracle`: `expected in actual or actual in expected` | An empty answer, or any substring of the truth, passes. | Normalise both sides, then require equality or a typed comparison. Reject empty output. |
| G2 | `harness/testing/multiturn.py:667` | Cross-turn consistency computes `tension_energy(entity, entity)` | The check compares each entity with itself, so it can never fail. | Compare what each turn asserted about the entity, e.g. the (relation, tail) claimed per turn. |
| G3 | `harness/testing/gateway.py:262-269` | `allowed = self.policy_engine.check(...)`. A `PolicyDecision` dataclass without `__bool__` is always truthy. | A `PolicyEngine` attached to `ToolGateway` never blocks anything. | Use `decision.allowed`. Pass a real context, not `run_id`. |
| G4 | `harness/testing/policy.py:461-467` | A `requires` rule of severity `error` always blocks | Whether the requirement is met is never evaluated, so a satisfied requirement still blocks. | Evaluate the requirement against the context or state before blocking. |
| G5 | `harness/testing/oracles.py:835-842` | `TraceOracle._check_holonomy` passes two lists to `check_holonomy(path, direct)`, and the exception is swallowed | The holonomy part of trace checking always returns `None`. | Pass a relation path and the direct relation. Surface errors. |
| G6 | `benchmark/eval/generators/base.py:83` | `QuestionGenerator(geometric=False)` by default | The two-way admission is off unless the caller turns it on. | Default to `True` when a cap-trained store is present, or log loudly when it is off. |
| G7 | `harness/testing/bridge.py:196-205` | The default bridge instantiates generators without `geometric=True` | Default campaigns admit symbolic-only questions. | Pass the flag from the config. |
| G8 | `harness/suite/emit.py:31-33` | `_default_materialize` is the identity | Factor levels are recorded but never applied to the text. | Ship a template realiser as the default, or fail when none is configured. |
| G9 | `harness/suite/evaluate.py:68` | `evaluate.run` sends `scn.base.query` | Suite attribution analyses factors the system under test never saw. | Send the materialised text and record it in the result row. |
| G10 | `harness/graphdoe/question_rephraser.py:575-586` | `_with_nl` changes only `natural_language` | About 9 factors never reach `llm_prompt`, which is what the SUT receives. | Apply the transform to `llm_prompt` too, or score on `natural_language`. |
| G11 | `harness/graphdoe/question_rephraser.py:1066-1084, 1088-1108` | The LLM rephraser prompt contains the ground truth. The output is checked only for JSON and length. | Rephrasings can leak or bend toward the answer, and nothing verifies the meaning was kept. | Keep the ground truth out of the prompt. Re-derive the answer from the rephrased text before admission. |
| G12 | `harness/testing/cross_document.py:656-659` | `CrossDocumentGenerator.__init__` never calls `super().__init__()` | `self.geometric` is missing, the `AttributeError` is swallowed in `_validate`, and every question is rejected. | Call `super().__init__(geometric=...)`. |
| G13 | `harness/testing/test_generators.py:125-128, 168-174` | `NegativeGenerator` answer functions return `True` while the stated ground truth is `False` | The symbolic check rejects every question, so the generator yields 0 questions. | Align the answer function with the ground truth. |
| G14 | `harness/core.py:499-500` with `harness/testing/gateway.py` | An injected error is returned through `before_tool` | The agent sees "Tool call skipped by policy hook." instead of the error. `on_tool_result` never fires, so the fault is not in `SUTTrace.tool_calls`. | Return the configured error as the tool result and record it in the trace. |
| G15 | `harness/testing/campaign.py:483-486` | The campaign summary reads `n_questions`, `n_correct` and `results` | `DOEBenchmarkResult` has `total_questions`, `total_correct` and `questions`, so the counts come out as 0. | Read the real field names. |

Lower-priority observations (not in the script):

- **Stale data ignores arguments.** `ToolGateway` stale-data returns the last result for the tool name whatever the arguments were (`gateway.py:343-355`).
- **Error injection is unseeded.** It uses Python's unseeded global `random` (`gateway.py:279-288`).
- **The harness session store persists between runs** (`core.py:189-194`). Adapters reset the message list but not the store.
- **Operating points are on the training data.** `GMSJudge.calibrate` cuts every threshold at P(correct) = 0.5 on training data, with no holdout (`judge.py:53-82`, `773-774`). The fitted `path_transport` model is never used.
- **The plausibility gate fails open.** `GovernedToolGateway` does so when a call cannot be scored, and θ defaults to 1.5 unless passed (`governed_gateway.py:270-273`, `383`).
- **Per-class FAR/FRR is computed over the whole cohort** in `policy_validation.py:215-240`, so every class reports the same number.
- **The analysis reference level is alphabetical.** `DOEAnalyzer.run_logistic` uses the alphabetically first level as the odds-ratio reference and flags significance at raw p < 0.05 (`analysis.py:101`, `174`, `187`).
- **Hybrid ingest can write LLM-extracted values into ENM.** In hybrid mode, numeric tails from LLM-extracted prose triples are written into ENM (`benchmark/ingest.py:712-718`). This contradicts the module's statement that values never bypass the deterministic path.
