"""turnval: graph walk, factors, simulation, errors, text, split, evaluation and a tiny training run.

The AOP data (turnval/data/*.json) is client material kept out of git: these tests skip when it is absent.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

DATA = Path(__file__).resolve().parents[1] / "turnval" / "data"
pytestmark = pytest.mark.skipif(not (DATA / "aop_flow.json").exists() or not (DATA / "hrt_graph.json").exists(),
                                reason="turnval/data (AOP flow + HRT graph) not present")


@pytest.fixture(scope="module")
def graphs():
    from turnval.graph import Flow, HRTGraph
    return Flow(), HRTGraph()


@pytest.fixture(scope="module")
def design():
    from turnval.factors import Factors
    fx = Factors.load()
    rows, rep = fx.design(seed=7, log=lambda *a: None)
    return fx, rows, rep


def test_walker_replays_the_real_call(graphs):
    """The logged call: SSN ask -> card ending in X -> (No) card number -> (yes it is) card number again."""
    from turnval.graph import DictTurn, Walker
    flow, _ = graphs
    md = {"checkpoint": "check4SSN", "channel_type": "voice", "journey": "Fraud"}
    w = Walker(flow, "User Identification", md)
    resp = "the user responds confirming whether or not they are asking about the card you inquired about"
    conf = "the user confirms they are asking about the card ending in {{ @set_4ssn_metadata.last5AccountNumber }}"
    got = [w.run(DictTurn({"set_4ssn_metadata": {"match": "NO_USER_RESPONSE"}})).state,
           w.run(DictTurn({"set_4ssn_metadata": {"match": "PROFILE_SINGLE_MATCH"}})).state,
           w.run(DictTurn({"set_4ssn_metadata": {"match": "PROFILE_SINGLE_MATCH"},
                           "set_cm15_metadata": {"match": "NO_USER_RESPONSE"}}, {resp: True, conf: False})).state,
           w.run(DictTurn({"set_cm15_metadata": {"match": "NO_USER_RESPONSE"}})).state]
    assert got == ["UI_CLAUSE_108", "UI_CLAUSE_119", "UI_CLAUSE_250", "UI_CLAUSE_250"]
    assert md["checkpoint"] == "checkCM15"


def test_cv_readout_is_one_assembled_response(graphs):
    from turnval.graph import DictTurn, Walker
    flow, hrt = graphs
    w = Walker(flow, "Charge Verification", {"num_transactions": 2})
    st = w.run(DictTurn({"fetch_fraud_charges": {"action": "SELF_SERVICE_ELIGIBLE"},
                         "transaction_readout": {"started": "False"}}))
    assert st.speech[0] == "CV_CLAUSE_032" and st.speech[-1] == "CV_CLAUSE_044" and len(st.speech) == 9


def test_hrt_resolution_and_branch_conditions(graphs):
    from turnval.graph import Context
    _, hrt = graphs
    r = hrt.resolve("Charge Verification", ["CV_CLAUSE_056"],
                    Context(txn={"transactionStatus": "approved", "merchantAddress": "1 Main St"}))
    assert set(r["branches"]) == {"CV_CLAUSE_056/A", "CV_CLAUSE_056/C", "CV_CLAUSE_056/E"}
    r = hrt.resolve("Charge Verification", ["CV_CLAUSE_056"], Context(txn={"transactionStatus": "declined"}))
    assert "CV_CLAUSE_056/D" in r["branches"] and "CV_CLAUSE_056/C" not in r["branches"]
    assert not hrt.unknown_conditions


def test_factors_cover_every_flow_outcome_and_all_pairs(design):
    fx, rows, rep = design
    assert fx.audit()["not_driven_by_a_factor"] == {}
    assert rep["uncovered"] == 0 and rep["rows"] > 100
    assert all(fx.valid({k: v for k, v in r.items() if k not in ("design_id", "replicate")}) for r in rows)


def test_simulation_and_errors_for_every_design_row(graphs, design):
    from turnval.errors import build_script
    from turnval.scenario import simulate
    flow, hrt = graphs
    _, rows, _ = design
    labels = set()
    for i, row in enumerate(rows[:120]):
        conv, sim = simulate(flow, hrt, row, f"c{i}", seed=i)
        sc = build_script(sim, conv, row, seed=i)
        assert conv.turns, row
        for st in sc.turns:
            labels.add(st.bot.label)
            if st.phase == "error":
                assert st.bot.label == 0 and st.variants and st.variants[0].label == 1   # its correct counterpart
            if st.phase in ("pre", "clean"):
                assert st.bot.label == 1
    assert labels == {0, 1}


def test_template_generation_split_and_report(tmp_path):
    from turnval.generate import main as gen_main
    from turnval.evaluate import full_report
    assert gen_main(["--renderer", "template", "--limit", "60", "--out", str(tmp_path)]) == 0
    for s in ("train", "val", "test"):
        assert (tmp_path / f"{s}.csv").exists()
    rep = json.loads((tmp_path / "split_report.json").read_text())
    assert sum(rep["rows"].values()) == len(pd.read_csv(tmp_path / "rows.csv"))
    df = pd.read_csv(tmp_path / "test.csv", keep_default_na=False)
    df["p"] = np.where(df.label == 1, 0.8, 0.2)
    out = full_report(df)
    assert out["overall@0.5"]["accuracy"] == 1.0 and out["pairwise"]["accuracy"] in (1.0, None)


def test_llm_renderer_prompts_with_a_fake_model(graphs, design):
    """The LLM path end to end with a stand-in chat model (no network)."""
    from turnval.errors import build_script
    from turnval.scenario import simulate
    from turnval.textgen import LLMRenderer, facts_for

    class Fake:
        def invoke(self, msgs):
            sys_ = msgs[0].content
            key = "utterance" if "caller says" in sys_ else "summary" if "summarise" in sys_ else \
                "checks" if "requirements" in sys_.lower() and "check" in sys_.lower() else "response"
            val = {"1": "ok"} if key == "checks" else "text for " + key
            return SimpleNamespace(content=json.dumps({key: val, "defect_present": True}))

    r = LLMRenderer({"backend": "gateway", "model": "fake", "cache_dir": "/tmp/turnval_test_cache", "verify": True})
    for m in (r.cust, r.botm, r.errm, r.summ, r.judge):
        m.model = Fake()
    flow, hrt = graphs
    row = next(x for x in design[1] if x["error_family"] == "content_missing")
    conv, sim = simulate(flow, hrt, row, "fake", seed=1)
    sc = build_script(sim, conv, row, seed=1)
    st = sc.turns[0]
    facts = facts_for(st.gold, conv.profile)
    assert r.customer(st.customer, conv.profile, [("bot", "hi")], ["plain"]) == "text for utterance"
    assert r.bot(st.bot, facts, [("bot", "hi")], "yes", []) == "text for response"
    assert r.check(st.bot, facts, "x")["accepted"]


def test_tiny_training_run(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from turnval.generate import main as gen_main
    from turnval.train.trainer import main as train_main
    d = tmp_path / "data"
    gen_main(["--renderer", "template", "--limit", "40", "--out", str(d)])
    rc = train_main(["--set", "model.backbone=tiny-random", "model.gradient_checkpointing=false",
                     f"data.train={d / 'train.csv'}", f"data.val={d / 'val.csv'}", f"data.test={d / 'test.csv'}",
                     "data.max_length=192", "train.epochs=1", "train.grad_accum=1", f"train.out_dir={tmp_path / 'run'}",
                     "model.proj_dim=16"])
    assert rc == 0
    s = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert "test" in s and "thresholds" in s and (tmp_path / "run" / "test_report.md").exists()


def test_losses():
    torch = pytest.importorskip("torch")
    from turnval.train.losses import pairwise, supcon
    s = torch.tensor([3.0, -3.0, 0.0])
    y = torch.tensor([1.0, 0.0, 1.0])
    g = torch.tensor([0, 0, 1])
    assert float(pairwise(s, y, g, 1.0)) < 0.01                       # the correct one already wins by 6
    assert float(pairwise(-s, y, g, 1.0)) > 5                        # and the reverse is punished
    z = torch.nn.functional.normalize(torch.tensor([[1.0, 0], [1.0, 0.01], [0, 1.0]]), dim=-1)
    assert float(supcon(z, torch.tensor([0, 0, 1]))) < float(supcon(z, torch.tensor([0, 1, 0])))


def test_api_through_a_vllm_style_server(tmp_path):
    """The notebook's calls over HTTP: config -> check the server -> pilot generation with the LLM renderer."""
    from turnval import api
    from turnval.mock_vllm import start
    url = start()
    cfg = api.config(backend="local", vllm_url=url, vllm_model="openai/gpt-oss-20b", out_dir=str(tmp_path),
                     replicates=1, workers=2, cache_dir=None, **{"text.cache_dir": str(tmp_path / "cache")})
    assert api.check_data(cfg)["replay_ok"]
    assert api.check_llm(cfg)["ok"]
    res = api.generate(cfg, limit=6, out_dir=str(tmp_path / "pilot"))
    assert res["report"]["failed"] == 0 and res["report"]["rows"] > 0
    assert Path(res["train"]).exists()
    with pytest.raises(RuntimeError):
        api.check_llm(api.config(backend="local", vllm_url=url, vllm_model="some/other-model"))


def test_gpu_plan():
    from turnval.train.hardware import plan
    p = plan(1, "H100 80GB")
    assert p["overrides"]["model.max_memory"] == {0: "70GiB"} and p["overrides"]["train.batch_rows"] >= 4
    assert plan(2, "L40S")["overrides"]["data.max_length"] == 1536
    with pytest.raises(ValueError):
        plan(1, "A10G")                      # 24 GB cannot hold 42 GB of weights
    with pytest.raises(ValueError):
        plan(2, "V100 32GB")                 # no bf16
    with pytest.raises(ValueError):
        plan(1, "Mystery GPU")               # unknown without memory_gb
