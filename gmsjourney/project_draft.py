"""A starting project file for any domain pack, drafted from its graph.

``python -m gmsjourney.project_draft <pack> [--write]`` prints (or writes to ``<pack>/project.yaml``) a project
whose every name comes from the pack's graph:

* **fixture**: the credential and each tool's default result (the pack's tool specs);
* **customer**: one script per prompt state, answering with the first act the state accepts and the first value
  of each required slot, sticking to its last answer when asked again;
* **journey factors**, each about one decision point (``anchor``) and relevant only when a case reaches it
  (``relevant_when: reached``):
    - every tool: success, each failure status (UNAVAILABLE included), a retry that succeeds and the failures up
      to the tool's limit for a tool whose retries are counted, a result value for each value a failure returns
      (an ineligibility reason), and a success echoing another value for a tool that echoes;
    - every prompt state: each value of its slot (one act with one slot), or each thing the customer can do there;
    - the always-available behaviours: a person asked for at a prompt, an unclear answer (once, or past the
      clarification limit), a policy question (``role: questions``), the turn limit (``role: turn_target``);
* **categories and tags** from those factors, **generator** settings with the graph's policy questions as padding,
  and the built-in **evaluation** and **speech** settings.

Edit the result in the studio (``python -m gmsjourney.studio --pack <pack>``): rename, merge, drop or add levels.
"""
from __future__ import annotations

import argparse
import re
from typing import Any

SIGNALS = {
    "condition_openers": ["if", "when", "unless", "in case"],
    "recording": ["recorded", "i've noted", "i have noted", "noted that", "noted it", "noted the", "saved", "logged",
                  "marked", "captured", "i've put", "i have put"],
    "completion": ["has been", "have been", "is done", "that's done", "it's done", "all done", "done\\b", "completed",
                   "is complete", "submitted", "i've (?:secured|locked|replaced|frozen|blocked|ordered|sent|submitted)",
                   "i have (?:secured|locked|replaced|frozen|blocked|ordered|sent|submitted)", "is now"],
    "offer": ["shall i", "should i", "would you like", "do you want", "can i", "may i", "i can", "want me to"],
    "fee": ["fee", "fees", "cost", "costs", "price", "pay", "charge for", "charges? you"],
    "free": ["free", "no fee", "no charge", "at no cost"],
}


def _slug(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")


def _frame(act: str, **slots) -> str:
    inner = ", ".join(f"{k}={v}" for k, v in slots.items() if v is not None)
    return f"{act}({inner})"


def _lc(text: str) -> str:
    """A sentence as a clause: 'Verifies the caller' -> 'verifies the caller' (an acronym keeps its case)."""
    return text[:1].lower() + text[1:] if text[1:2].islower() else text


def tool_about(tool: str, text: str | None) -> str:
    """A tool factor's description: what varies (the tool's answer), with the graph's own text about the tool. A
    text copied from a procedure table ('tool | when used | returns | rule') is read as its columns."""
    if not text:
        return f"what {tool} returns"
    cols = [c.strip().rstrip(".;") for c in text.split("|")]
    if len(cols) >= 3:
        if cols[0] == tool:
            cols = cols[1:]
        when, returns = cols[0], cols[1] if len(cols) > 1 else ""
        return f"what {tool} returns ({_lc(when)}: {returns})" if returns else f"what {tool} returns ({_lc(when)})"
    return f"what {tool} returns ({_lc(text.rstrip('.'))})"


def draft(dom=None) -> dict[str, Any]:
    from .domain import current
    from .names import GraphNames
    dom = dom or current()
    g = dom.graph
    if g is None:
        raise ValueError(f"{dom.name}: drafting needs a compact policy graph (schema gmsjourney.policy_graph/2)")
    G = g.glob
    human, question, unclear, tl = G.get("human"), G.get("question"), G.get("unclear"), G.get("turn_limit")
    glob_acts = {x["act"] for x in (human, question, unclear) if x}
    cred = str(dom.get("fixtures.credential", "0000"))

    # ---------------------------------------------------------------- the customer
    def default_answer(st: str) -> str:
        act = next(a for a in g.states[st]["on"] if a not in glob_acts)
        slots = {}
        for sl, sp in g.slot_specs(act).items():
            if sp.get("optional"):
                continue
            if sp["set"] in g.values:
                slots[sl] = g.members(sp["set"])[0]
            else:
                slots[sl] = "$credential"
        return _frame(act, **slots)

    prompts = list(g.prompt_states)
    script = {st: _slug(st) for st in prompts}
    scripts = {}
    not_repeated = [a for a in (question and question["act"], unclear and unclear["act"]) if a]
    loops = g.loops()
    in_loop = {st for lp in loops for st in lp["prompts"]}     # asked once per item: one queue per item
    for st in prompts:
        if st in in_loop:
            scripts[script[st]] = {"per_item": True, "item_script": [default_answer(st)],
                                   "unknown_item": default_answer(st),
                                   "exhausted": {"repeat_last_answer": True, "not_repeated": not_repeated}}
        else:
            scripts[script[st]] = {"default": default_answer(st),
                                   "exhausted": {"repeat_last_answer": True, "not_repeated": not_repeated}}
    every = lambda st: {"item": "all"} if st in in_loop else {}          # noqa: E731  a decision for every item
    gate = next((t for t, sp in g.tools.items() if sp.get("role") == "auth"), None)
    # where policy questions are asked: the first prompt after the auth tool on the default path
    reached = {s: _tools_before(g, s) for s in prompts}
    q_state = next((s for s in prompts if reached[s] is not None and gate in reached[s]), None) or \
        (prompts[1] if len(prompts) > 1 else prompts[0])

    # ---------------------------------------------------------------- factors
    factors: list[dict[str, Any]] = []
    for tool, sp in g.tools.items():
        if sp.get("role") == "handoff":
            continue
        key = f"tool:{tool}"
        levels: dict[str, Any] = {"ok": {}}
        statuses = list(sp.get("fail", [])) + ["UNAVAILABLE"]
        value_fields = {f: vs for f, vs in (sp.get("returns") or {}).items() if isinstance(vs, str) and vs in g.values}
        counted = _counted_limit(g, tool)
        said = {stt for stt in statuses for f in value_fields if _says_field(g, tool, stt, f)}
        for stt in statuses:
            if value_fields and stt in said:
                for f, vs in value_fields.items():
                    for m in g.members(vs):
                        levels[_slug(f"{stt}_{m}")] = {"fixture": {key: [{"status": stt, f: m}]}}
                continue
            if counted:
                lim = g.limit_value(counted)
                levels[_slug(f"retry_{sp['ok']}")] = {"fixture": {key: [stt] + [sp["ok"]]}}
                levels[_slug(f"{stt}_{lim}x")] = {"fixture": {key: [stt] * lim}, "boundary": True}
                counted = None
                continue
            levels[_slug(stt)] = {"fixture": {key: [stt]}}
        for f, typ in (sp.get("returns") or {}).items():   # some members of a value set: offer only one of them
            if isinstance(typ, dict) and "subset_of" in typ and len(g.members(typ["subset_of"])) > 1:
                for m in g.members(typ["subset_of"]):
                    levels[_slug(f"only_{m}")] = {"fixture": {key: [{"status": sp["ok"], f: [m]}]}}
        if sp.get("echo"):
            levels["echo_mismatch"] = {"fixture": {key: [{"status": sp["ok"], sp["echo"]: "@other"}]}}
        factors.append({"name": _slug(tool), "about": tool_about(tool, sp.get("text")),
                        "default": "ok", "anchor": {"tool": tool}, "relevant_when": "reached", "levels": levels})
    for st in prompts:
        acts = [a for a in g.states[st]["on"] if a not in glob_acts]
        slotted = [a for a in acts if any(not sp.get("optional") and sp["set"] in g.values
                                          for sp in g.slot_specs(a).values())]
        if len(acts) == 1 and len(slotted) == 1:
            act = acts[0]
            sl, sp = next((k, v) for k, v in g.slot_specs(act).items() if v["set"] in g.values)
            vals = g.members(sp["set"])
            levels = {_slug(vals[0]): {}}
            for v in vals[1:]:
                levels[_slug(v)] = {"customer": [{"script": script[st], "say": [_frame(act, **{sl: v})],
                                                  "position": "replace", **every(st)}]}
            factors.append({"name": _slug(f"{st}_{sl}"), "about": f"the {sl.replace('_', ' ')} the customer gives at {st}",
                            "default": _slug(vals[0]), "anchor": {"state": st}, "relevant_when": "reached",
                            "levels": levels})
        elif len(acts) > 1:
            first = default_answer(st)
            levels = {_slug(re.sub(r"\(.*", "", first)): {}}
            for a in acts:
                specs = {k: v for k, v in g.slot_specs(a).items() if v["set"] in g.values and not v.get("optional")}
                options = [_frame(a)] if not specs else \
                    [_frame(a, **{k: m}) for k, v in specs.items() for m in g.members(v["set"])]
                for fr in options:
                    if fr == first:
                        continue
                    # another answer, then the default one if the prompt is asked again (a switch, then a yes)
                    levels[_slug(fr)] = {"customer": [{"script": script[st], "say": [fr, first], "position": "replace",
                                                       **every(st)}]}
            if len(levels) > 1:
                factors.append({"name": _slug(st), "about": f"what the customer does at {st}",
                                "default": next(iter(levels)), "anchor": {"state": st}, "relevant_when": "reached",
                                "levels": levels})
    constraints: list[str] = []
    for lp in loops:                                  # the loop: how many items, the window's edges, mixed answers
        constraints += _loop_factors(g, lp, factors, script, default_answer, glob_acts)
    for st in prompts:                                # what the customer adds to an answer (optional slots)
        act = next((a for a in g.states[st]["on"] if a not in glob_acts), None)
        if act is None:
            continue
        for sl, sp in g.slot_specs(act).items():
            if not sp.get("optional") or sp.get("correction_of") or sp["set"] not in g.values:
                continue
            levels = {"none": {}}
            for m in g.members(sp["set"]):
                extra = {need: vals[0] for need, vals in (sp.get("with") or {}).items()}
                slots = {**extra, sl: m}
                if st in in_loop:                     # layered onto every item's answer
                    op = {"script": script[st], "add_slots": slots, "item": "all"}
                else:
                    base = default_answer(st)
                    op = {"script": script[st], "say": [base[:-1] + (", " if base[-2] != "(" else "")
                                                        + ", ".join(f"{k}={v}" for k, v in slots.items()) + ")"],
                          "position": "replace"}
                levels[_slug(m)] = {"customer": [op]}
            factors.append({"name": _slug(f"{st}_{sl}"), "about": f"the {sl.replace('_', ' ')} the customer adds at {st}",
                            "default": "none", "anchor": {"state": st}, "relevant_when": "reached", "levels": levels})
    if human:
        levels = {"none": {}}
        for st in prompts:
            levels[_slug(f"at_{st}")] = {"customer": [{"script": script[st], "say": [_frame(human["act"])],
                                                       "position": "start"}]}
        factors.append({"name": "human_request", "about": "the customer asks for a person at a prompt", "default": "none",
                        "relevant_when": "reached", "levels": levels})
    if unclear:
        levels = {"none": {}}
        for st in prompts:
            u = _frame(unclear["act"])
            levels[_slug(f"once_at_{st}")] = {"customer": [{"script": script[st], "say": [u], "position": "start"}]}
            levels[_slug(f"persists_at_{st}")] = {"customer": [{"script": script[st], "say": [u, u], "position": "start"}]}
        factors.append({"name": "unclear_answer",
                        "about": "an answer that carries no decision, once (recovered) or past the clarification limit",
                        "default": "none", "relevant_when": "reached", "levels": levels})
    if question:
        factors.append({"name": "policy_question", "about": f"the customer asks a policy question at {q_state}",
                        "default": "none", "role": "questions", "anchor": {"state": q_state}, "relevant_when": "reached",
                        "levels": {"none": {}, "ask": {"questions": 1}}})
    if tl:
        reason = tl["else"][1:]
        factors.append({"name": "turn_pressure", "about": "how close the conversation runs to the turn limit",
                        "default": "normal", "role": "turn_target", "anchor": {"state": q_state}, "relevant_when": "reached",
                        "levels": {"normal": {},
                                   "at_limit": {"turn_target": f"limit('{tl['limit']}')", "exact": True,
                                                "avoid_transfer": reason, "boundary": True},
                                   "over_limit": {"turn_target": f"limit('{tl['limit']}') + 1", "transfer": reason,
                                                  "boundary": True}}})

    # ---------------------------------------------------------------- categories, tags
    names = [f["name"] for f in factors]
    tool_f = [f for f in factors if f.get("anchor", {}).get("tool")]
    failing = " or ".join(f"level('{f['name']}') != 'ok'" for f in tool_f) or "False"
    interrupts = " or ".join(x for x in (
        "level('unclear_answer') != 'none'" if "unclear_answer" in names else "",) if x) or "False"
    gov = " or ".join(x for x in ("level('human_request') != 'none'" if "human_request" in names else "",) if x) or "False"
    bound_levels = [f"level('{f['name']}') in {tuple(k for k, v in f['levels'].items() if v.get('boundary'))!r}"
                    for f in factors if f.get("role") == "loop" and any(v.get("boundary") for v in f["levels"].values())]
    bound = " or ".join(x for x in ("level('turn_pressure') != 'normal'" if "turn_pressure" in names else "",
                                     *bound_levels) if x) or "False"
    categories = [{"name": "recovery", "when": f"{failing} or {interrupts}"},
                  {"name": "boundary", "when": bound},
                  {"name": "governance_policy", "when": gov},
                  {"name": "procedural_ordering", "when": "True"}]
    tags = [{"name": "procedural_ordering", "when": "True"}] + [c for c in categories[:-1]]

    # ---------------------------------------------------------------- generator
    from .graph.kb import kb_frames
    pad = [_frame("ASK_KB", **dict(fr.slots)) for fr in kb_frames(g)][:6]
    gen = {"policy_question_case": {"policy_question": "ask"} if question else {}, "padding_questions": pad,
           "fault_only_rules": []}
    from .evaluation.severity import DEFAULT_SEVERITY, DEFAULT_TIER_ESCALATION, DEFAULT_PROFILES
    from dataclasses import asdict
    evaluation = {"severity": dict(DEFAULT_SEVERITY), "tier_escalation": DEFAULT_TIER_ESCALATION,
                  "release_gate": {t: {k: v for k, v in asdict(p).items() if k != "tier"} for t, p in DEFAULT_PROFILES.items()}}
    speech = {"also_allowed": {"any": [x for x in ((unclear or {}).get("clarify"), g.end_state) if x]}, "signals": SIGNALS}
    return {
        "project": {"name": g.title},
        "fixture": {"credential": cred, "defaults": {}},
        "customer": {"answers": {st: script[st] for st in prompts}, "scripts": scripts,
                     "questions": {"script": script[q_state], "position": "start"}},
        "factors": [{k: v for k, v in f.items() if k != "role" or v != "loop"} for f in factors],
        "constraints": constraints,
        "categories": categories,
        "tags": tags,
        "generator": gen,
        "evaluation": evaluation,
        "speech": speech,
    }


def _loop_factors(g, lp: dict[str, Any], factors: list[dict[str, Any]], script: dict[str, str], default_answer,
                  glob_acts: set[str]) -> list[str]:
    """Factors for a loop over a returned list, and the constraints that keep them meaningful: how many items the
    list holds (one, two, at the cap, over it), where the last item sits against the window (inside, on its edge, one
    more just outside), and the customer answering the second item differently from the first."""
    tool, fld = lp["tool"], lp["field"]
    name = _slug(fld)
    levels: dict[str, Any] = {"one": {}, "two": {"fixture": {"items": 2}}}
    if lp.get("cap"):
        cap = int(g.limit_value(lp["cap"]))
        if cap > 2:
            levels[f"at_cap_{cap}"] = {"fixture": {"items": cap}, "boundary": True}
        levels[f"over_cap_{cap + 1}"] = {"fixture": {"items": cap + 1}, "boundary": True}
    added = [{"name": f"{name}_count", "about": f"how many {fld.replace('_', ' ')} {tool} returns (the loop handles "
              f"them one at a time{', up to ' + lp['cap'] if lp.get('cap') else ''})", "default": "one",
              "anchor": {"tool": tool}, "relevant_when": "reached", "role": "loop", "levels": levels}]
    age = lp.get("age")
    lim = (lp["where"].get(age) or {}).get("max") if age else None
    if lim:
        w = int(g.limit_value(lim))
        added.append({"name": f"{name}_window", "about": f"where the items' {age.replace('_', ' ')} sit against {lim}",
                      "default": "inside", "anchor": {"tool": tool}, "relevant_when": "reached", "role": "loop",
                      "levels": {"inside": {}, f"last_at_{w}": {"fixture": {"last_item_age": w}, "boundary": True},
                                 f"one_more_at_{w + 1}": {"fixture": {"extra_item_age": w + 1}, "boundary": True}}})
    first = next((st for st in [lp["do"], *lp["prompts"]] if st in lp["prompts"]), None)
    if first:
        acts = [a for a in g.states[first]["on"] if a not in glob_acts]
        slot = next(((a, sl, sp["set"]) for a in acts for sl, sp in g.slot_specs(a).items()
                     if not sp.get("optional") and sp["set"] in g.values), None)
        if slot:
            act, sl, vs = slot
            others = [m for m in g.members(vs) if _frame(act, **{sl: m}) != default_answer(first)]
            if others:
                added.append({"name": f"{name}_answers", "about": f"whether the customer answers {first} the same way "
                              f"for every item (the second item answered differently)", "default": "same",
                              "anchor": {"state": first}, "relevant_when": "reached", "role": "loop",
                              "levels": {"same": {}, _slug(f"second_{others[0]}"): {"customer": [
                                  {"script": script[first], "item": 1, "say": [_frame(act, **{sl: others[0]})],
                                   "position": "replace"}]}}})
    factors.extend(added)
    tool_f = next((f for f in factors if f.get("anchor", {}).get("tool") == tool and f.get("role") != "loop"), None)
    out = []
    if tool_f:                                        # item lists only matter when the tool returns them
        for f in added:
            if f["anchor"].get("tool") == tool:
                out.append(f"level('{tool_f['name']}') == '{tool_f['default']}' or level('{f['name']}') == '{f['default']}'")
    return out


def _tools_before(g, state: str) -> list[str] | None:
    """Tools called on the default path before a prompt state is first reached (first act, first value, success);
    None when the default path does not reach it."""
    seen, out, cur = set(), [], g.start
    for _ in range(200):
        if cur == state:
            return out
        if cur in seen:
            break
        seen.add(cur)
        k = g.kind(cur)
        s = g.states[cur]
        if k == "say":
            nxt = s["go"]
        elif k == "call":
            out.append(s["call"])
            nxt = s["on"].get(g.tools[s["call"]]["ok"])
        elif k == "ask":
            nxt = next(iter(s["on"].values()))
        elif k == "each":                            # a loop: its first item
            nxt = s["do"]
        else:
            break
        while isinstance(nxt, (list, dict)):
            nxt = nxt[-1] if isinstance(nxt, list) else nxt.get("then")
        if not isinstance(nxt, str) or nxt.startswith(("!", "@")):
            break
        cur = nxt
    return out if cur == state else None


def _says_field(g, tool: str, status: str, field: str) -> bool:
    """Does the state a status leads to say the tool's returned field (``$tool.field``)?"""
    for s in g.states.values():
        if s.get("call") != tool:
            continue
        for t, _ in g.alternatives(s["on"].get(status, [])) if status in s["on"] else []:
            st = g.states.get(t) or {}
            if any(ref == f"${tool}.{field}" for ref in (st.get("say") or {}).values()):
                return True
    return False


def _counted_limit(g, tool: str) -> str | None:
    """The limit a tool's failures are counted against (a ``below`` target in its call state), if any."""
    for s in g.states.values():
        if s.get("call") == tool:
            for t in (s.get("on") or {}).values():
                if isinstance(t, dict) and "below" in t:
                    return t["below"]
    return None


HEADER = """# Project configuration of the {name} pack, drafted from its policy graph (python -m gmsjourney.project_draft).
# Every name here is the graph's; check it with: python -m gmsjourney.project check {path}
# Edit it in the studio:  python -m gmsjourney.studio --pack {pack}
#
# A factor with relevant_when: reached is relevant when a case reaches its anchor (a tool call or a prompt), decided
# by walking the case through the graph. Fixture keys tool:<name> list what that tool returns on its 1st, 2nd, ...
# call (a status, or a result object); the last result repeats.
"""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m gmsjourney.project_draft")
    ap.add_argument("pack")
    ap.add_argument("--write", action="store_true", help="write <pack>/project.yaml (an existing file is kept as .bak)")
    a = ap.parse_args(argv)
    from .domain import set_current
    from .project import dump_yaml
    dom = set_current(a.pack)
    raw = draft(dom)
    text = HEADER.format(name=raw["project"]["name"], path=dom.project_path, pack=a.pack) + "\n" + dump_yaml(raw)
    if a.write:
        if dom.project_path.exists():
            dom.project_path.with_suffix(".yaml.bak").write_text(dom.project_path.read_text())
        dom.project_path.write_text(text)
        print(f"wrote {dom.project_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
