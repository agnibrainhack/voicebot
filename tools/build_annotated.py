"""Build ANNOTATED.html — a line-by-line annotated listing of ontology.py, oracle.py and
enrich/reader.py, plus how the three connect in the invariance proof.

The code is read from the source files verbatim; the notes are line ranges supplied below.
The script prints any line not covered by a note, so "line by line" is checkable.
"""
import argparse
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

E = html.escape

# --------------------------------------------------------------------------- notes
# (first_line, last_line, note) — note is placed immediately before first_line.
ONTOLOGY = [
    (1, 21, "The contract this file exists to make true: <i>if two utterances map to the same frame, they are the same test input, whatever their surface form.</i> Three parts — the domain individuals read from the store, the closed set of dialog acts, and SemanticFrame itself. The last paragraph is the promise the invariance proof later cashes: ground truth is a function of the frame sequence and the fixture, and nothing else."),
    (22, 27, "Standard-library only: <code>hashlib</code> for the two signature functions, <code>json</code> for canonical form and for Turtle literals."),
    (29, 29, "The IRI namespace the exported evidence uses."),
    (32, 43, "The closed vocabulary of customer meanings. Ten acts, no more — this is what makes the reader's output checkable: a frame is either one of these or it is not a meaning at all. Note <code>UNCLEAR</code>: the policy needs a name for <i>content-level</i> ambiguity (the customer said something, but no decision is recoverable), because R22/R23 require the bot to ask for a clarification rather than guess."),
    (45, 50, "The bot's side of the conversation, listed for the ontology export (the acts the oracle emits). These are not gating anywhere — the invariance proof only constrains what the <i>customer</i> meant."),
    (54, 74, "The 18 policy-question meanings and the slots each one carries. The last two are the hallucination probes: <code>unknown_option</code> (a near-miss name such as “priority delivery”) and <code>unstated_fact</code> — questions whose correct answer is <i>the policy says nothing</i>, so the only passing behaviour is to abstain."),
    (78, 81, "<code>frozen=True</code> is load-bearing: frames are compared and hashed everywhere, so they must be immutable. Slots are a tuple of pairs, not a dict, for the same reason."),
    (83, 86, "<code>make()</code> is the only constructor used in practice: it drops unset slots (<code>None</code>) and sorts them, so two frames built in different orders are the same frame. This is what lets the oracle and the reader construct frames independently and still compare equal."),
    (88, 89, "Slot lookup, with a default so callers can ask <code>f.get(\"corrected_from\")</code> without guarding."),
    (91, 93, "<code>canonical()</code> is the string form that gets hashed: act plus sorted slot pairs as JSON. Everything downstream that compares meanings compares this string."),
    (95, 96, "<code>signature()</code> — a 16-hex-character fingerprint of one frame. Used for evidence and for the exported RDF, not for equality (equality goes through the ontology, which is state-aware)."),
    (98, 100, "The debugging and report form: <code>ACT(slot=value, ...)</code>. Every trace line, every CSV cell and every entry in the answer sheet shows frames in this shape — which is why a reader of the outputs is looking at exactly what the proof compared, not a summary of it."),
    (103, 108, "<code>_freeze</code> makes values hashable by turning lists into tuples, recursively. A slot value that is a tuple (the <code>options</code> of an mcq question) survives the round trip."),
    (111, 114, "<code>_thaw</code> is its inverse, applied when a frame is written to a report or to RDF."),
    (117, 128, "<b>The conversation signature.</b> SHA-256 over the ordered sequence of (dialog state, canonical card) pairs, truncated to 20 hex characters. The state is folded into the hash with a separator (<code>\\x1f</code>) before each card, and cards are separated by <code>\\x1e</code> — so “the same card in a different state” and “the same cards in a different order” are both different signatures. This single function is what the invariance proof compares when it says the enriched conversation <i>is</i> the base conversation, and what <code>ontology.ttl</code> exports as evidence."),
    (131, 153, "The state-aware judge. The docstring lists the four rules that make state-awareness real: a value is canonicalised <i>within the value set its slot draws from</i> (“regular” means standard delivery only where delivery is being discussed); a bare yes/no is read the way the state reads it; a card is admissible only if the state accepts its act and every slot and value is declared; and equivalence means same canonical card <i>and</i> admissible there. Without a graph attached (<code>kg is None</code>) it degrades gracefully to the state-free mapping."),
    (155, 178, "<code>from_store</code> builds three lookup tables from the trained store: every entity mapped to its class (for the RDF export), a global alias map (alias → canonical, including the canonical itself and its spaced form), and — the important one — <b>a per-value-set map</b>, so a word is resolved inside the vocabulary of the slot it fills rather than globally."),
    (180, 183, "<code>canon</code>: resolve a word inside a value set when we know which one applies, else fall back to the global alias map."),
    (186, 189, "<code>slot_value_set</code>: which value set a given slot of a given act draws from, straight from the policy graph's slot schema. This is the link that makes canonicalisation slot-aware."),
    (191, 201, "<b>The bare yes/no rule.</b> A frame that is only CONFIRM or DENY with no slots is not a meaning on its own — it means whatever the state asked for. In PRESENT_TXN that turns “yes” into <code>INFORM_DISPOSITION(disposition=RECOGNIZED)</code>; in CONFIRM_ACTION it stays <code>CONFIRM</code>; in ASK_MORE_HELP “no” becomes <code>END</code>. The wrap into INFORM_DISPOSITION when the target is a disposition value is what makes the substitution type-correct."),
    (203, 221, "<code>admissible</code> — the legitimacy test, and the reason a proof about text alone would be insufficient. Four checks: the state accepts this act at all; every slot is declared for the act; every value belongs to the slot's value set; every required slot is present. It returns a reason, which is what you see quoted in the answer sheet when a card is rejected."),
    (223, 233, "<code>card</code> — the canonical, human-readable form of a meaning: state, stage, act, slots, and whether it is admissible (with the reason if not). Every artifact in this project shows meanings in this shape, which is why a reader of the CSV or the answer sheet is looking at exactly what the proof compared."),
    (235, 239, "<code>disjoint</code>: two values are mutually exclusive when they belong to the same value set (RECOGNIZED vs UNRECOGNIZED, standard vs expedited). Disjointness is what lets <code>frame_distance</code> grade a swap as a contradiction rather than a typo."),
    (241, 246, "<code>canonicalize</code> — the two-step normalisation applied before any comparison: read a bare yes/no in its state, then resolve every string slot value inside its own value set. Every equality in the proof goes through this function."),
    (248, 252, "<b>The test the whole method turns on.</b> Two frames are equivalent when their canonical forms are identical <i>and</i> the read-back card is admissible in the state where it was said. Not “similar”, not “close enough” — equal, and legal there. L2 of the invariance proof is this call, once per customer turn."),
    (254, 272, "<code>frame_distance</code> — the graded version, used as evidence rather than as a gate. 0 for identical; 1.0 when the acts differ; 0.5 for a missing or extra slot; for two values that are disjoint siblings, half the store's <code>tension_energy</code> (so RECOGNIZED→UNRECOGNIZED lands around 1.0 — a contradiction, not a wording change); 0.75 otherwise. The max over slots is taken, so one bad slot is enough."),
    (275, 333, "<code>to_turtle</code> — the evidence export. It writes the journey ontology (classes, properties, the ten acts, the KB intents, every individual with its <code>skos:altLabel</code> aliases), the <code>owl:AllDifferent</code> declarations that encode disjointness, the exact numeric register with the first 12 characters of each integrity hash, the rules with their stance and text, the asserted graph — and finally, appended under <code># ---- invariance evidence</code>, the per-case triples the validator passes in."),
]

ORACLE = [
    (1, 14, "What the oracle is: an <i>executable reading</i> of the policy, not an annotated answer key. The last paragraph is the hinge for invariance — it consumes semantic frames only and never reads text, so its answer depends on what the customer <b>meant</b>, which is exactly what the proof shows enrichment leaves unchanged. If the oracle read text, the proof would be circular."),
    (15, 24, "Imports: the journey machinery (spec, user simulator, fixture builder), the knowledge-base truth function, the frame type, the mock tool environment, and the trace it fills in."),
    (27, 32, "Two control-flow exceptions. <code>_Stop</code> is normal termination (the call ended); <code>_TurnLimit</code> is raised by <code>_ask</code> when the turn budget is spent and is caught once, in <code>_journey</code>, where it is turned into the policy's prescribed behaviour."),
    (35, 53, "<code>Limits</code> — every number the journey depends on, read by key from the store's exact numeric register (<code>lookup_enm</code>), never parsed from prose. This is why a limit can be quoted with provenance: <code>max_turns</code> is 20 because the register says so, and the register says so because a policy triple with an evidence quote says so."),
    (56, 60, "Construction: the store, the limits, and the required opening line."),
    (62, 81, "<b><code>run()</code> — the entry point, and the seam the invariance proof plugs into.</b> <code>frames_override</code> replaces the simulated customer with a fixed list of frames: L3 re-derives the answer key from the frames the <i>reader</i> recovered, which is how “the messy text still leads to the same call” is demonstrated rather than asserted. Note the return: a <code>Trace</code> carrying the outcome signature, per-stage components, the rules fired, the workflow edges and the full event list."),
    (84, 85, "Records a bot act. Everything the oracle does is appended to <code>Trace.events</code>, which is what the turn-by-turn export and the answer sheet read."),
    (87, 103, "<b><code>_ask</code> — the turn gate, and the tightest point in the whole system.</b> Before asking, it checks the turn budget and raises <code>_TurnLimit</code> at the cap (so the 20-turn rule is enforced by construction, not by a check someone might forget). Then it emits the bot's acts and takes the customer's reply — from the replay list when re-deriving, otherwise from the simulator. Silence ends the call. The turn counter and <code>user_frames</code> are updated here: <code>user_frames</code> is precisely the sequence the invariance proof compares and replays."),
    (105, 113, "<code>_call</code> — every tool call is recorded with its arguments and its result. A <code>ToolError</code> is recorded too and then re-raised, because a failed call is part of the expected trajectory: the bot that transfers on an unavailable tool is behaving correctly, and the record must show the fault."),
    (115, 128, "<code>_transfer</code>: summarise what was recorded so far (R19), emit TRANSFER with the reason, make the transfer tool call, and stop. The reason string is part of the ground truth signature, so “transferred for the wrong reason” is a failing outcome, not a detail."),
    (130, 131, "<code>_recorded_claims</code> — the running record of what has been recorded so far, keyed by transaction. It is what gets summarised on a transfer (R19) and what the summary act carries, so the bot's summary is checked against the same dictionary the outcome is built from."),
    (133, 137, "<code>_answer_kb</code> — the answer to a policy question comes from <code>kb_answer(store, frame)</code>: computed from the store (exact register, aliases, arithmetic), never from text. The trace marks it correct by construction, and R25/R27 fire. This is why a bot that states a different figure is <i>measurably</i> wrong rather than merely different."),
    (140, 150, "<code>_journey</code> — the three phases in order. The turn-limit handler is the interesting part: if the limit is hit during the finish phase the call simply ends with a summary, and otherwise the bot must transfer with reason <code>turn_limit</code> under R28. Two different correct behaviours for the same exception, decided by which phase the call had reached."),
    (152, 181, "<code>_authenticate</code> — the loop that runs before anything account-specific. Human request → transfer (R20); a policy question → answer it and re-ask; anything else → a clarification (R22), and too many clarifications → transfer (R23); a credential → call the tool; a tool error → transfer (R5); success → R3; the attempt cap → transfer (R4). Every branch ends in either a return or a transfer, so the loop cannot spin."),
    (183, 205, "<code>_retrieve</code> — fetch the charges, then apply the window (R30 fires when anything falls outside the 30 days <i>or</i> sits exactly on the boundary — the boundary case is deliberate). No reviewable charges at all → explain, ask if anything else is needed, end (R8), with a KB question allowed in between."),
    (207, 249, "<code>_review</code> — at most four charges per session. Record before moving on (R12), and only claim success when the tool says SUCCESS <i>and</i> echoes the same transaction (R13/R33) — the echo check is what catches a tool that silently records the wrong row. UNSURE ends in a transfer (R11). An unrecognised charge opens the card-action path once. If more charges exist than the cap allows, the bot must explain the limit and transfer (R31). Otherwise: summarise, ask if anything else is needed, end."),
    (251, 281, "<code>_capture</code> — the disposition loop for one charge. A policy question is answered and the charge re-presented; a human request transfers; INFORM_DISPOSITION with <code>corrected_from</code> set fires R21 (the latest clear answer rule); RECOGNIZED/UNRECOGNIZED returns the disposition plus any card slots; UNSURE re-presents the context (R10) and after too many tries is recorded as UNSURE; anything else asks for a clarification."),
    (283, 330, "<code>_card_action</code> — the card path. The options are fetched (R15) and an action the account does not offer is refused rather than attempted. Replacing the card with expedited delivery must disclose the $15 fee first (R26), then confirm (R16). A different request mid-confirmation replaces the action. A decline is acknowledged, not argued with. The submission must succeed <i>and</i> echo the action that was confirmed (R17/R29) — and R18 fires when other charges remain to be reviewed."),
]

READER = [
    (1, 19, "What the reader is: <b>the instrument of the invariance proof.</b> The docstring is the design contract — it must read an utterance the way a careful human agent would, <i>independent of how the realizer built it</i>. Five commitments follow: aliases come from the store rather than from templates; typos are repaired by edit distance to the ontology's own vocabulary; speech-recognition confusions are undone by an inverse homophone map; spoken digits and numbers are normalised; reported speech is not the customer's answer; and R21 decides between a false start and its correction. The final paragraph fixes the precedence order when a turn carries several cues."),
    (20, 26, "Imports, plus the number-word table from the noise module — the same table the realizer uses to <i>write</i> spoken numbers, used here in reverse to read them."),
    (28, 31, "Three number maps. <code>_DIGIT_WORDS</code> deliberately includes the homophones a speech recogniser produces — “won” for one, “to”/“too” for two, “for” for four, “ate” for eight — because a phone bot must survive them. <code>_STRICT_DIGITS</code> keeps the unambiguous ones, which is how the code detector avoids reading “for” as a 4 in ordinary speech."),
    (33, 39, "The <code>ASR_REPAIR</code> table: inverse-homophone fixes applied before any analysis (cart→card, humane→human, fee's→fee, their→there). Deterministic and auditable — a reader that silently “fixed” more than this would be inventing meaning."),
    (40, 42, "Fillers are removed rather than analysed (they carry no meaning); <code>REPAIR_MARK</code> is what turns an earlier answer into a <i>correction</i> rather than a mistake; <code>REPORTED</code> marks reported speech."),
    (44, 56, "The cue patterns. Each is a request to the policy's own vocabulary — there is no free-text interpretation anywhere. Read them as the definition of what the reader can hear: an unsure answer, an unrecognised charge (in many phrasings), a recognised one, a card to secure or replace, standard or expedited delivery, a request for a human, an end, and agreement or refusal."),
    (57, 64, "The probe machinery. <code>OPTION_NAME</code> captures an option name before the word “delivery”; <code>NAME_STOP</code> lists the words that are not part of a name; <code>NEAR_MISS_WORDS</code> lists the qualifiers a caller may use that name no option of this policy — listed explicitly so the typo repair can restore them, because the reader <i>must</i> read a near-miss name back exactly for the hallucination probe to work."),
    (65, 68, "<code>QWORD</code> — what makes a clause a question, including questions phrased as statements or requests (“tell me …”, “I want to know …”). Without this, an imperative question would be read as a card request."),
    (71, 87, "<b>The guard against inventing meaning.</b> A few hundred ordinary English words that must never be “corrected” into a cue word by the typo repairer. Without this list, “mine” could be repaired from “mind”, or “card” from “care” — and the reader would be manufacturing the decision it is supposed to be detecting."),
    (90, 103, "<code>_osa</code> — optimal string alignment distance (Levenshtein with transposition counted as one edit), with an early exit above the cap so it stays cheap. Transposition-as-one matters for real typos (“hte” for “the”)."),
    (106, 128, "Construction: build the vocabulary the repairer may use, from three sources — the store's aliases (so every synonym the graph knows is readable), the cue patterns themselves, and a hand-written list of journey words. <code>domain_vocab</code> is kept separately from <code>vocab</code>: it is the tie-breaker that prefers a domain word over an ordinary one. <code>kg</code> is the policy graph, kept for the dialog-state checks in <code>read()</code>."),
    (130, 143, "<b><code>normalize</code> — every turn goes through this before anything is decided.</b> Lower-case and unify quotes; protect decimal points so “45.67” is not split into a clause; drop apostrophes; strip fillers; apply the ASR repairs; repair each token against the vocabulary; normalise numbers; and turn dashes and ellipses into an explicit clause break (<code>|</code>). The result is what the rest of the file — and every stored artifact — actually sees. It is cached (50,000 entries) because the proof normalises the same text repeatedly."),
    (145, 164, "<code>_repair_token</code>: refuse to touch short words, known words, and ordinary English. Otherwise look for a vocabulary word with the same first letter and a length within the cap (1 edit up to 7 characters, 2 beyond), and prefer — in order — a smaller distance, a domain word over a common word, a longer word, then alphabetical order. The tie-break rules are what make the repair deterministic and explainable rather than a nearest-neighbour guess."),
    (166, 208, "<b><code>_numbers</code> — spoken numbers to digits, and the reason “four eight two one” is readable at all.</b> A run of three or more digit words containing at least two unambiguous ones is a code (that is how “for ate two one” becomes 4821 without reading “for” as a 4 in ordinary speech); tens plus a unit compound (“thirty one”); a lone digit word only in front of a unit (“five dollars”, “two business days”). Everything else is left alone, because a reader that turns every “one” into a digit would corrupt the text it is meant to be reading faithfully."),
    (210, 233, "<code>clauses</code> — split the normalised text into clauses (keeping whether each was a question), then <i>drop reported speech</i>: from a marker like “told me” onwards, up to nine words are removed, so “the last agent told me I wouldn't need to verify” does not become the customer's own claim that verification is unnecessary."),
    (235, 240, "<code>cue_counts</code> — how many disposition and delivery cues an utterance carries. Not used by the reader itself: the negative-control generator uses it to only build an edit where the sentence has exactly one cue of the kind being edited, so the edited sentence is a coherent opposite rather than a self-contradiction."),
    (242, 260, "<b><code>read</code> — the public entry point.</b> Four-digit numbers are the subtle part: a code is read as a credential only where the state asks for one (the policy graph decides), so a charge amount can never be mistaken for a card. If nothing else was said, a bare code is still a credential — that is the “just the digits” turn. Note the comment: how a <i>bare yes/no</i> is read in its state is deliberately the ontology's job, not the reader's, so that the same words can mean different things in different states without the reader knowing anything about the journey."),
    (262, 281, "<b><code>_read_rest</code> — the precedence order</b>, which the docstring at the top of the file promises: a question clause is tried as a policy question first; then a disposition (with any card request attached); then a card request on its own; then an end; then the <i>latest</i> confirm-or-deny cue; and only then UNCLEAR. The order encodes the policy: a turn that contains both a decision and a question is read as the decision, because that is what a careful agent would act on."),
    (283, 295, "<code>_disposition_matches</code> — find every disposition cue and drop overlaps, keeping them in order of appearance. Simple, but it is what makes the false-start rule implementable: you need <i>all</i> the cues, in order, before you can decide which one was final."),
    (297, 307, "<b><code>_disposition</code> — R21 in eight lines.</b> The final clear answer wins (“use the latest clear answer”). An earlier, <i>different</i> answer is recorded as <code>corrected_from</code> only when a repair marker sits between the two — so “no, sorry, yes” is a correction, while “no thanks, that's all” is not. The repair marker is the difference between a customer changing their mind and a customer saying two unrelated things."),
    (309, 321, "<code>_card</code> — securing versus replacing the card (whichever is mentioned last wins, matching R21's spirit), and the delivery option from the last delivery word. The card slots attach to whatever disposition was read, which is why the frame can carry both a decision and a request in one turn."),
    (324, 330, "<code>_money</code> — dollar amounts, “N dollars”, and the whole “free / no charge / doesn't cost anything” family mapped to 0. Fee questions are answered by comparing against the register, so reading $0 correctly matters as much as reading $15."),
    (332, 395, "<b><code>_kb</code> — the intent ladder.</b> Ordered from the most specific cue to the most general, because the first match wins: refund, fraud, skipping verification, the procedure, a window check carrying a day count, the lookback limit, the attempts limit, the transaction cap, counting the options, the three-part expedited question, which option needs confirmation, the fee difference, the fastest option and its fee, then cheaper. Only if none of those matched does it consider the probes — an unstated fee on securing the card, or an option name the policy does not list — and finally the plain value questions (how much, how long) and multi-option fee lists. The ordering is the file's real content: it is a decision procedure over the policy's question vocabulary, and every branch names a meaning the ground-truth function knows how to answer."),
]

FILES = [
    ("ontology.py", ROOT / "gmsjourney" / "ontology.py", ONTOLOGY,
     "The judge", "Turns meanings into canonical form, decides equivalence in a dialog state, and defines "
     "the conversation signature."),
    ("oracle.py", ROOT / "gmsjourney" / "oracle.py", ORACLE,
     "The answer key", "Plays the call from meaning cards and reads every limit from exact memory — the "
     "ground truth, computed rather than annotated."),
    ("enrich/reader.py", ROOT / "gmsjourney" / "enrich" / "reader.py", READER,
     "The instrument", "Reads an utterance back to a meaning card, independent of how it was written."),
]


def listing(path: Path, notes) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    by_start = {n[0]: n for n in notes}
    covered = set()
    for a, b, _ in notes:
        covered.update(range(a, b + 1))
    # blank lines, comments and decorators are carried by the note above them
    def _trivial(i):
        t = lines[i - 1].strip()
        return t == "" or t.startswith("#") or t.startswith("@")
    missing = [i for i in range(1, len(lines) + 1) if i not in covered and not _trivial(i)]
    if missing:
        print(f"  !! {path.name}: {len(missing)} uncovered line(s): {missing[:12]}")
    out = []
    for i, line in enumerate(lines, 1):
        if i in by_start:
            out.append(f'<div class="note">{by_start[i][2]}</div>')
        out.append(f'<div class="l"><span class="n">{i}</span><span class="c">{E(line) or "&nbsp;"}</span></div>')
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Annotated listing of the invariance-critical source files.")
    ap.add_argument("--out", default=str(ROOT / "docs" / "ANNOTATED.html"))
    ap.add_argument("--trace", default=None,
                    help="a captured trace to embed; omitted renders a placeholder")
    args = ap.parse_args()
    out_path = Path(args.out)
    chain_path = Path(args.trace) if args.trace else None

    parts = []
    for name, path, notes, role, blurb in FILES:
        print(f"annotating {name} …")
        body = listing(path, notes)
        anchor = name.replace("/", "-").replace(".py", "")
        parts.append(f"""
<section class="file" id="{anchor}">
  <h2>{E(name)} <span class="role">{E(role)}</span></h2>
  <p class="blurb">{blurb} <span class="small">{len(path.read_text(encoding='utf-8').splitlines())} lines.</span></p>
  <div class="listing">{body}</div>
</section>""")

    chain = (chain_path.read_text(encoding="utf-8", errors="replace")
             if chain_path is not None and chain_path.exists() else "(no --trace given)")
    # keep the trace readable: drop the store-loading chatter
    chain = "\n".join(l for l in chain.splitlines() if not l.startswith("[1/12]"))

    doc = f"""<title>Frames, Reader, Oracle</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{{--bg:#eef1f4;--surface:#fff;--ink:#131c26;--muted:#5a6b7d;--line:#d3dbe3;--line-soft:#e3e9ee;
--accent:#a8500a;--accent-soft:#f7e9db;--llm:#0d6a72;--code-bg:#f4f6f9;--note-bg:#fbf7f1;--gut:#c9d3dc;
--display:"IBM Plex Sans Condensed",system-ui,sans-serif;--body:"IBM Plex Sans",system-ui,sans-serif;
--mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#10161d;--surface:#161e27;--ink:#e6edf3;
--muted:#93a3b3;--line:#2b3743;--line-soft:#222c36;--accent:#f0a35e;--accent-soft:#33261a;--llm:#63c9cf;
--code-bg:#131a22;--note-bg:#1d1a15;--gut:#3a4650;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#10161d;--surface:#161e27;--ink:#e6edf3;--muted:#93a3b3;--line:#2b3743;
--line-soft:#222c36;--accent:#f0a35e;--accent-soft:#33261a;--llm:#63c9cf;--code-bg:#131a22;--note-bg:#1d1a15;--gut:#3a4650;color-scheme:dark}}
body{{background:var(--bg);color:var(--ink);font-family:var(--body);font-size:16px;line-height:1.62;padding-inline:16px}}
.wrap{{max-width:1120px;margin:0 auto;padding-block:clamp(26px,5vw,60px)}}
h1,h2,h3{{font-family:var(--display);line-height:1.15;margin:0;text-wrap:balance}}
h1{{font-size:clamp(1.9rem,5vw,2.9rem);font-weight:700}}
h2{{font-size:1.5rem;font-weight:700;margin-block:1.8rem .35rem}}
h3{{font-size:1.05rem;font-weight:600}}
p{{margin:.6rem 0}}
.lede{{color:var(--muted);font-size:1.06rem;max-width:70ch}}
.kicker{{display:inline-flex;align-items:center;gap:.5rem;font-family:var(--mono);font-size:.72rem;text-transform:uppercase;letter-spacing:.16em;color:var(--accent)}}
.kicker::before{{content:"";width:22px;height:2px;background:var(--accent)}}
code{{font-family:var(--mono);font-size:.86em;background:var(--code-bg);border:1px solid var(--line-soft);border-radius:4px;padding:.05em .3em;word-break:break-word}}
.role{{font-family:var(--mono);font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;color:var(--accent);background:var(--accent-soft);padding:.12rem .45rem;border-radius:4px;vertical-align:.2em}}
.blurb{{color:var(--muted);max-width:78ch}}
.small{{font-size:.84rem}}
.listing{{border:1px solid var(--line);border-radius:9px;background:var(--surface);overflow-x:auto;margin:.7rem 0 1.6rem}}
.listing .l{{display:grid;grid-template-columns:3.4rem minmax(0,1fr);font-family:var(--mono);font-size:.785rem;line-height:1.55}}
.listing .l .n{{color:var(--muted);text-align:right;padding:.05rem .55rem .05rem .3rem;border-right:1px solid var(--line-soft);user-select:none;font-size:.72rem}}
.listing .l .c{{padding:.05rem .8rem;white-space:pre;color:var(--ink)}}
.listing .note{{background:var(--note-bg);border-left:3px solid var(--accent);padding:.5rem .85rem;margin:.35rem 0;
  font-size:.9rem;line-height:1.55;border-top:1px solid var(--line-soft);border-bottom:1px solid var(--line-soft)}}
.listing .note b{{font-weight:600}}
figure{{margin:1.2rem 0}}
figcaption{{font-size:.84rem;color:var(--muted);margin-top:.5rem;max-width:72ch}}
svg{{max-width:100%;height:auto;display:block}}
.figscroll{{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--surface);padding:12px}}
.console{{background:var(--code-bg);border:1px solid var(--line);border-left:3px solid var(--llm);border-radius:6px;
  padding:.7rem .85rem;overflow-x:auto;font-family:var(--mono);font-size:.78rem;line-height:1.5;white-space:pre;color:var(--ink)}}
.steps{{counter-reset:s;display:grid;gap:10px;margin-top:.8rem}}
.sbox{{background:var(--surface);border:1px solid var(--line);border-radius:9px;padding:.8rem .95rem;min-width:0}}
.sbox h4{{display:flex;gap:.55rem;align-items:baseline;font-family:var(--display);font-size:1rem;margin:0}}
.sbox h4::before{{counter-increment:s;content:counter(s);font-family:var(--mono);font-size:.7rem;color:var(--accent);
  border:1px solid var(--accent);border-radius:50%;width:1.35rem;height:1.35rem;display:grid;place-items:center;flex:none}}
.sbox p{{font-size:.93rem;margin:.35rem 0 0}}
table{{border-collapse:collapse;width:100%;font-size:.88rem}}
th,td{{text-align:left;vertical-align:top;padding:.45rem .6rem;border-bottom:1px solid var(--line-soft)}}
th{{font-family:var(--display);font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted)}}
.tscroll{{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--surface)}}
footer{{margin-top:2.5rem;padding-top:1rem;border-top:1px solid var(--line);font-size:.85rem;color:var(--muted)}}
</style>
<div class="wrap">
<span class="kicker">Annotated source · the three files the invariance proof rests on</span>
<h1>Frames, Reader, Oracle</h1>
<p class="lede">Every line of <code>ontology.py</code>, <code>oracle.py</code> and <code>enrich/reader.py</code>,
explained in order — and then joined up: how these three files, and nothing else, decide whether an enriched
test is the same test as its base case.</p>

<p><b>How to read this.</b> The code below is quoted verbatim from the files, with real line numbers. Notes sit
between the lines they explain; a note covers every line under it until the next note. Nothing is skipped — if a
line is in the file, it is here.</p>

<h2>The chain in one picture</h2>
<div class="figscroll">
<figure>
<svg viewBox="0 0 880 330" role="img" aria-label="A meaning card is realised into text by the realizer; the reader turns the text back into a card; the ontology canonicalises that card in its dialog state and decides equivalence; the oracle replays the recovered cards to re-derive the outcome; the sequence signature and the truth hash are compared.">
  <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>
  <g font-family="IBM Plex Sans, system-ui, sans-serif" font-size="13" fill="currentColor">
    <rect x="14" y="26" width="150" height="54" rx="7" fill="none" stroke="currentColor" stroke-opacity=".45"/>
    <text x="89" y="48" text-anchor="middle" font-weight="600">meaning card</text>
    <text x="89" y="66" text-anchor="middle" font-size="11" fill-opacity=".7">the intended frame</text>
    <line x1="168" y1="53" x2="214" y2="53" stroke="currentColor" stroke-opacity=".6" marker-end="url(#a)"/>
    <text x="191" y="45" text-anchor="middle" font-size="10" fill-opacity=".75">realizer</text>
    <rect x="218" y="26" width="150" height="54" rx="7" fill="none" stroke="currentColor" stroke-opacity=".45"/>
    <text x="293" y="48" text-anchor="middle" font-weight="600">messy text</text>
    <text x="293" y="66" text-anchor="middle" font-size="11" fill-opacity=".7">typos, fillers, asides</text>
    <line x1="372" y1="53" x2="418" y2="53" stroke="currentColor" stroke-opacity=".6" marker-end="url(#a)"/>
    <text x="395" y="45" text-anchor="middle" font-size="10" fill-opacity=".75">reader.read</text>
    <rect x="422" y="26" width="170" height="54" rx="7" fill="none" stroke="var(--accent)" stroke-width="1.6"/>
    <text x="507" y="48" text-anchor="middle" font-weight="600">recovered card</text>
    <text x="507" y="66" text-anchor="middle" font-size="11" fill-opacity=".7">in its dialog state</text>
    <line x1="596" y1="53" x2="642" y2="53" stroke="currentColor" stroke-opacity=".6" marker-end="url(#a)"/>
    <text x="619" y="45" text-anchor="middle" font-size="10" fill-opacity=".75">ontology</text>
    <rect x="646" y="26" width="220" height="54" rx="7" fill="none" stroke="currentColor" stroke-opacity=".45"/>
    <text x="756" y="48" text-anchor="middle" font-weight="600">equivalent? admissible?</text>
    <text x="756" y="66" text-anchor="middle" font-size="11" fill-opacity=".7">yes → the turn is proved</text>

    <line x1="507" y1="84" x2="507" y2="126" stroke="currentColor" stroke-opacity=".6" marker-end="url(#a)"/>
    <text x="540" y="108" font-size="10" fill-opacity=".75">all turns</text>
    <rect x="330" y="130" width="354" height="54" rx="7" fill="none" stroke="currentColor" stroke-opacity=".45"/>
    <text x="507" y="152" text-anchor="middle" font-weight="600">sequence of recovered cards</text>
    <text x="507" y="170" text-anchor="middle" font-size="11" fill-opacity=".7">the same list the oracle replays</text>
    <line x1="330" y1="157" x2="272" y2="157" stroke="currentColor" stroke-opacity=".6" marker-end="url(#a)"/>
    <text x="301" y="149" text-anchor="middle" font-size="10" fill-opacity=".75">frames_override</text>
    <rect x="66" y="130" width="202" height="54" rx="7" fill="none" stroke="currentColor" stroke-opacity=".45"/>
    <text x="167" y="152" text-anchor="middle" font-weight="600">oracle re-derives</text>
    <text x="167" y="170" text-anchor="middle" font-size="11" fill-opacity=".7">outcome + every stage</text>

    <line x1="167" y1="188" x2="167" y2="230" stroke="currentColor" stroke-opacity=".6" marker-end="url(#a)"/>
    <rect x="40" y="234" width="256" height="54" rx="7" fill="none" stroke="currentColor" stroke-opacity=".45"/>
    <text x="168" y="256" text-anchor="middle" font-weight="600">same as the base case?</text>
    <text x="168" y="274" text-anchor="middle" font-size="11" fill-opacity=".7">outcome signature · stages · truth hash</text>
    <line x1="300" y1="261" x2="346" y2="261" stroke="currentColor" stroke-opacity=".6" marker-end="url(#a)"/>
    <rect x="350" y="234" width="256" height="54" rx="7" fill="none" stroke="currentColor" stroke-opacity=".45"/>
    <text x="478" y="256" text-anchor="middle" font-weight="600">same conversation?</text>
    <text x="478" y="274" text-anchor="middle" font-size="11" fill-opacity=".7">sequence_signature equal</text>
    <line x1="610" y1="261" x2="656" y2="261" stroke="currentColor" stroke-opacity=".6" marker-end="url(#a)"/>
    <rect x="660" y="234" width="206" height="54" rx="7" fill="none" stroke="var(--accent)" stroke-width="1.6"/>
    <text x="763" y="256" text-anchor="middle" font-weight="600">proved invariant</text>
    <text x="763" y="274" text-anchor="middle" font-size="11" fill-opacity=".7">the turn and the scenario</text>
    <text x="14" y="318" font-size="11" fill-opacity=".7">Grey boxes are data; the amber box is the state-aware step that no string comparison could do.</text>
  </g>
</svg>
<figcaption>The reader never sees the intended card, and the ontology never sees the text — each step is blind to
what the previous one knew, which is why agreement between them is evidence rather than tautology.</figcaption>
</figure>
</div>

<h2>How they connect, turn by turn</h2>
<p>The whole proof is five calls per customer turn, and each one lives in one of these three files:</p>
<div class="tscroll"><table>
<thead><tr><th style="width:38%">Call</th><th style="width:14%">In file</th><th>What it establishes</th></tr></thead>
<tbody>
<tr><td><code>reader.read(text, state)</code></td><td>reader.py</td><td>The messy sentence is turned back into a meaning card — the reader's opinion of what was said.</td></tr>
<tr><td><code>ont.canonicalize(g, state)</code></td><td>ontology.py</td><td>That card is normalised in its state: a bare yes/no becomes what the state asked, and every value is resolved inside its own value set.</td></tr>
<tr><td><code>ont.equivalent(frame, g, state)</code></td><td>ontology.py</td><td>The normalised read-back is identical to the intended card <i>and</i> legal where it was said. This is L2.</td></tr>
<tr><td><code>sequence_signature(read_frames, states)</code></td><td>ontology.py</td><td>The whole conversation, reduced to one string — compared against the base conversation's string.</td></tr>
<tr><td><code>oracle.run(spec, fixture, frames_override=read_frames)</code></td><td>oracle.py</td><td>The oracle replays the cards the reader recovered and must reach the same outcome and stages. This is L3 — and it is only sound because the oracle reads frames and never text.</td></tr>
</tbody>
</table></div>

<h2>One turn through all three files</h2>
<p>The turn below is real: the false-start disposition of scenario <code>kb-probe-near_miss-02~0003</code>, read in
<code>PRESENT_TXN</code>. Everything printed was produced by running the three files on the trained store.</p>
<div class="console">{E(chain)}</div>
<p>Read the trace in order and the division of labour becomes visible. <code>normalize</code> strips the fillers
and turns the dash into a clause break, so the false start (“no”) and what follows it (“sorry, yes, mine”)
become separate clauses; the asides — including “you can skip writing anything down”, an instruction-conflict
factor written to tempt the bot — contain no cue and are simply not read as anything. <code>_disposition</code>
then finds exactly <b>one</b> disposition cue, the “mine”, and returns <code>RECOGNIZED</code> with
<code>corrected_from=None</code>. That is the rule behaving correctly rather than merely firing: a correction is
recorded only when two <i>different</i> disposition answers are separated by a repair marker, and here “no” is a
repair, not a disposition. The ontology then canonicalises the card to itself — an explicit disposition is not
re-interpreted by the state — and declares it admissible, so the turn is proved with distance 0.0.</p>
<p>Now the last three lines. The <i>same words</i> answered in ASK_AUTH, CONFIRM_ACTION or ASK_MORE_HELP are read
to the <i>same card</i> — and rejected anyway, because that card is not admissible there. No string comparison
could make that distinction; it takes the policy graph. That is why a <code>state</code> parameter runs through
all three files, and why the proof is about meaning <i>in a dialog state</i> rather than about text.</p>

{''.join(parts)}

<footer>Annotated from <code>gmsjourney/ontology.py</code>, <code>gmsjourney/oracle.py</code> and
<code>gmsjourney/enrich/reader.py</code> at branch <code>gmsh-parity-0.4</code>; the trace was captured by running
them against the campaign store. Companions: <a href="EXPLAINER.html">Inside gmsjourney</a> ·
<a href="INVARIANCE.html">How Invariance Is Proven</a> · <a href="ANSWER_SHEET.html">Answer Sheet</a>.</footer>
</div>
"""
    out_path.write_text(doc, encoding="utf-8")
    print(f"wrote {out_path} ({len(doc)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
