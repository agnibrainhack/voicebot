"""One campaign in its own process: ``python -m gmsjourney.app.worker <run_dir>``.

Reads ``<run_dir>/job.json`` (the journey, the device and the campaign settings), runs ``pipeline.run_campaign``
and appends one JSON event per line to ``<run_dir>/events.jsonl`` as it goes:

    start        the job as given
    step_start   a step began (step, title)
    step_done    a step finished, with what it produced (summaries.summarize: metrics, tables, notes)
    log          a line the pipeline logged
    done         the run finished (headline numbers); the outputs are under <run_dir>/out
    error        the run stopped (the traceback)

A separate process keeps the console responsive, lets a run be cancelled, and frees the GPU when it ends.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

OPTIONS = ("max_cases", "runs_per_base", "seed", "run_resilience", "run_mutation", "run_power", "run_governance",
           "speech_controls", "calibration_bootstrap", "negative_controls_per_type", "llm_arm", "llm_mode", "llm_model",
           "rebuild_store", "profile")


def main(run_dir: str) -> int:
    run = Path(run_dir)
    job = json.loads((run / "job.json").read_text())
    if job.get("device") and job["device"] != "auto":
        os.environ["GMSJOURNEY_DEVICE"] = job["device"]
    events = open(run / "events.jsonl", "a", buffering=1, encoding="utf-8")

    def emit(**e) -> None:
        e["t"] = round(time.time(), 3)
        events.write(json.dumps(e, default=str) + "\n")

    emit(type="start", job=job, pid=os.getpid())
    try:
        from ..knowledge.kx_store import device as _device
        dev = _device()                            # what "auto" resolved to on this machine
        name = None
        if dev.startswith("cuda"):
            import torch
            idx = int(dev.split(":")[1]) if ":" in dev else torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
        emit(type="device", device=dev, name=name)
    except Exception as e:                         # noqa: BLE001 - the run reports it and carries on
        emit(type="log", text=f"could not resolve the device: {type(e).__name__}: {e}")
    try:
        from ..pipeline import CampaignConfig, run_campaign
        from . import summaries as S
        sent: set[str] = set()

        def finished(R, before: str | None = None) -> None:
            # a later step's keys can appear early (step 7 already holds the release gate): only steps before the one
            # starting count as done, and every step counts once the campaign has returned
            lim = int(before) if before else 99
            for k in S.done_steps(R):
                if k not in sent and int(k) < lim:
                    sent.add(k)
                    emit(type="step_done", step=k, details=S.summarize(k, R))

        def on_step(step: str, title: str, R) -> None:
            finished(R, re.match(r"\d+", step).group(0))     # everything before this step is done
            emit(type="step_start", step=step, title=title)

        def log(msg, *a) -> None:
            emit(type="log", text=" ".join(str(x) for x in (msg, *a)))

        opts = {k: job[k] for k in OPTIONS if job.get(k) is not None}
        sut = job.get("sut") or {}
        if (run / "sut.json").exists():            # the full prompt and attachments (runs.Runs.start)
            sut = json.loads((run / "sut.json").read_text(encoding="utf-8"))
        fac = None
        if sut.get("kind") == "llm":               # the tester's LLM bot instead of the journey's reference bot
            from ..domain import set_current
            from ..sut.llm_bot import CRITIC_VERSION
            from .sut import make_factory
            set_current(job["pack"])
            fac, name, _ = make_factory(sut)
            (run / "out").mkdir(parents=True, exist_ok=True)
            (run / "out" / "sut_prompt.txt").write_text(fac.system_prompt, encoding="utf-8")
            opts.update(bot_factory=fac, bot_name=name)
            refs = [r.get("name") for r in sut.get("references") or [] if r.get("text")]
            log(f"bot under test: {name}; prompt {len(fac.system_prompt):,} characters"
                + (f", with {', '.join(refs)} attached" if refs else "") + " (out/sut_prompt.txt); "
                + ("critique layer on: a second call reviews every step and may correct it (out/sut_critic_prompt.txt); "
                   if fac.critique else "")
                + 
                "planted defects (step 10) need the reference bot and are skipped")
        cfg = CampaignConfig(domain=job["pack"], out_dir=str(run / "out"), device=job.get("device") or None,
                             on_step=on_step, progress=False, **opts)
        try:
            R = run_campaign(cfg, log=log)
        finally:
            if fac is not None:                    # what the model was asked and answered, even when the run stops
                (run / "out" / "sut_llm.json").write_text(json.dumps(
                    {"bot": fac.stats.as_dict(), "llm": fac.llm.report(),
                     "critique": {"on": fac.critique, "version": CRITIC_VERSION, "revisions": fac.stats.revisions()}},
                    indent=1, default=str))
                if fac.critique:
                    (run / "out" / "sut_critic_prompt.txt").write_text(fac.critic_prompt, encoding="utf-8")
        finished(R)
        head = S.headline(R)
        head["bot"] = (R.get("results") or {}).get("bot")
        (run / "headline.json").write_text(json.dumps(head, default=str))
        emit(type="done", headline=head)
        return 0
    except Exception:                              # noqa: BLE001
        emit(type="error", text=traceback.format_exc())
        return 1
    finally:
        events.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
