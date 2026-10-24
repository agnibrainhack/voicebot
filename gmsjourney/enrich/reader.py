"""The frame reader: utterance -> semantic frame, grounded in the ontology.

This is the instrument of the invariance proof. It must read an utterance the
way a careful human agent would, independent of how the realizer built it:

* value words are bound to canonical individuals through the store's alias
  edges (skos:altLabel), not through realizer templates;
* typos are repaired by optimal-string-alignment distance to the ontology
  vocabulary, and speech-recognition confusions by an inverse homophone map;
* spoken numbers and digit runs are normalised ("for ate two one" -> 4821);
* reported speech ("the last agent told me ...") is not the customer's answer;
* the policy rule "use the latest clear answer" (R21) decides between a false
  start and its correction, and an explicit repair marker ("wait, actually")
  is what makes an earlier answer a correction.

Precedence when several acts are present: human request, credential, policy
question, disposition (+ card request), card request, end, confirm/deny,
otherwise UNCLEAR.
"""
from __future__ import annotations

import re
from functools import lru_cache

from ..ontology import SemanticFrame as F
from .lexicon import Lexicon
from .noise import NUMBER_WORDS

_WORD_NUM = {w: n for n, w in NUMBER_WORDS.items()}
_DIGIT_WORDS = {"zero": 0, "oh": 0, "one": 1, "won": 1, "two": 2, "to": 2, "too": 2, "three": 3,
                "four": 4, "for": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "ate": 8, "nine": 9}
_STRICT_DIGITS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}

FILLER_RE = re.compile(r"\b(um|uh|hmm|you know|okay so|like|so),\s*|\b(um|uh|hmm|you know|okay so)\b\s*")
REPAIR_MARK = r"(?:\bwait\b|\bactually\b|\bi mean\b|\bsorry\b|\bon second thought\b)"
REPORTED = re.compile(r"\b(told me|told us|said|says|according to)\b")

# value / act cues, speech repairs, the domain vocabulary and near-miss words come from the
# domain lexicon (domain.yaml -> lexicon); the lists below are language-level (English).
NAME_STOP = set("""the a an does do is for of your my this that which whichever any pick choose use get with by via
what whats how much long if i about on go want take cost costs fee price many business days option options would
should and or to it its please tell me know wondering was were be have has can could will""".split())
_QWORD_START = r"^(how|hows|what|whats|which|when|where|why|is|does|do|can|will|am|are|could|would|if|for)\b"
_QWORD_ANY = ["i want to know", "tell me", "i was wondering", "walk me through", "is it true", "am i right", "can we",
              "can you", "could you", "do you", "do i", "will i", "is there", "is it", "is a", "does an", "does a",
              "how", "what", "whats", "which"]


# Valid English words are never "repaired" into cue words (a reader must not invent meaning).
COMMON_WORDS = set("""
about above after again against along already also always another answer anyone anything anyway around asked
asking because been before being below better between both bring call called calling came cannot card care case
cause certain change changed check checked checking come coming could cost costs count days deal does doing done
down each early earlier either else even ever every everyone exactly feel find fine first follow found friend
from give given going gone good great happen happened hard have having hear heard help helps here hold home hope
hour hours however idea into just keep kind know known last later least left less like likely little live long
look looked looking lost made make makes many maybe mean means might money month more morning most much must
need needs never next nice none note nothing notice number often okay once only open order other over part
past pick place plan please point policy pretty problem quick quite rather read ready real really reason right
said same says second seem seems seen send sent should show side since small some someone something soon sorry
start stay still such sure take taken talk tell than thank thanks that their them then there these they thing
things think this those though thought through time today told took tried true turn under until upon used using
usual usually very wait want wanted wants week well went were what when where whether which while whole will
wish with within without wonder wondering word words work worried would wrong year years your yours
""".split())


def _osa(a: str, b: str, cap: int = 3) -> int:
    """Optimal string alignment distance (transposition = 1), early exit above cap."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[-1]


class FrameReader:
    def __init__(self, store, lexicon: Lexicon | None = None):
        self.store = store
        self.lx = lx = lexicon or Lexicon.current()
        self.value_pat = lx.value_patterns                    # value -> cue regex
        self.act_pat = lx.act_patterns                        # act -> cue regex
        self.reading = lx.get("reading", {})
        kbq = lx.kb
        self.option_name = lx.option_name_regex
        self.qword = re.compile(_QWORD_START + "|" + r"\b(" + "|".join(_QWORD_ANY + lx.get("question_cues", [])) + r")\b")
        vocab = set()
        for canon, als in store.aliases.items():
            for a in als + [canon.replace("_", " ")]:
                vocab.update(re.findall(r"[a-z]+", a.lower().replace("'", "")))
        for pat in list(self.value_pat.values()) + list(self.act_pat.values()):
            vocab.update(w for w in re.findall(r"[a-z]{4,}", pat))
        vocab.update(lx.vocabulary)
        self.domain_vocab = {w for w in vocab if len(w) >= 4}
        self.kg = getattr(store, "kg", None)                 # policy graph: dialog states
        vocab.update(kbq.get("near_miss_words", []))
        vocab.update(COMMON_WORDS)
        vocab.update(w for w in NUMBER_WORDS.values() if len(w) >= 4)
        self.vocab = {w for w in vocab if len(w) >= 4}
        code_act = self.reading.get("code")
        self.code_act = code_act
        self.code_spec = lx.acts[code_act]["code"] if code_act else None

    # ------------------------------------------------------------ normalise
    @lru_cache(maxsize=50000)
    def normalize(self, text: str) -> str:
        t = text.lower().replace("’", "'").replace("‘", "'")
        t = re.sub(r"(\d)\.(\d)", r"\1<dot>\2", t)
        t = t.replace("'", "")
        t = re.sub(r"\s+", " ", FILLER_RE.sub(" ", t)).strip()
        for pat, rep in self.lx.speech_repairs:
            t = re.sub(pat, rep, t)
        t = " ".join(self._repair_token(w) for w in t.split(" "))
        t = self._numbers(t)
        t = re.sub(r"\s*(—|–|--|\.\.\.)\s*", " | ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _repair_token(self, w: str) -> str:
        m = re.match(r"^([a-z]+)(\W*)$", w)
        if not m:
            return w
        core, tail = m.groups()
        if len(core) < 4 or core in self.vocab or core in COMMON_WORDS:
            return w
        cap = 1 if len(core) <= 7 else 2
        best_key, best = None, None
        for v in self.vocab:
            if v[0] != core[0] or abs(len(v) - len(core)) > cap or (len(core) == 4 and len(v) < 5):
                continue
            d = _osa(core, v, cap)
            if d > cap:
                continue
            # ties: prefer domain cue words over common words, then longer words, then alphabetical
            key = (d, v in COMMON_WORDS and v not in self.domain_vocab, -len(v), v)
            if best_key is None or key < best_key:
                best_key, best = key, v
        return (best + tail) if best is not None else w

    @staticmethod
    def _numbers(t: str) -> str:
        """Spoken numbers -> digits.

        * a run of >= 3 digit words with at least two unambiguous ones is a code
          ("four eight two one", "for ate two one" -> 4821); homophones such as
          "for"/"to" only count inside such a run;
        * tens + unit compounds ("thirty one", "thirty won") and teens/tens;
        * a lone digit word only before a unit ("five dollars", "two business days").
        """
        toks = t.split(" ")
        word = [re.sub(r"[^a-z]", "", x) for x in toks]
        tail = [re.sub(r"^[a-z]+", "", x) if word[i] else "" for i, x in enumerate(toks)]
        out: list[str] = []
        i = 0
        while i < len(toks):
            w = word[i]
            if w in _DIGIT_WORDS:
                j = i
                while j < len(toks) and word[j] in _DIGIT_WORDS and (j == i or not tail[j - 1]):
                    j += 1
                run = word[i:j]
                strict = sum(x in _STRICT_DIGITS for x in run)
                if (len(run) >= 3 and strict >= 2) or len(run) >= 4:
                    out.append("".join(str(_DIGIT_WORDS[x]) for x in run) + tail[j - 1])
                    i = j
                    continue
            if w in _WORD_NUM and _WORD_NUM[w] >= 20 and _WORD_NUM[w] % 10 == 0 and not tail[i] \
                    and i + 1 < len(toks) and word[i + 1] in _DIGIT_WORDS and _DIGIT_WORDS[word[i + 1]] > 0:
                out.append(str(_WORD_NUM[w] + _DIGIT_WORDS[word[i + 1]]) + tail[i + 1])
                i += 2
                continue
            if w in _WORD_NUM and _WORD_NUM[w] >= 10:
                out.append(str(_WORD_NUM[w]) + tail[i])
                i += 1
                continue
            if w in _STRICT_DIGITS and i + 1 < len(toks) and re.match(r"(dollar|day|business)", word[i + 1]):
                out.append(str(_DIGIT_WORDS[w]) + tail[i])
                i += 1
                continue
            out.append(toks[i])
            i += 1
        return " ".join(out)

    def clauses(self, text: str) -> list[tuple[str, bool]]:
        t = self.normalize(text)
        parts = re.split(r"(\?|\.|!|;|\|)", t)
        out: list[tuple[str, bool]] = []
        buf = ""
        for p in parts:
            if p in ("?", ".", "!", ";", "|"):
                if buf.strip():
                    out.append((buf.strip(), p == "?"))
                buf = ""
            else:
                buf += p
        if buf.strip():
            out.append((buf.strip(), False))
        clean = []
        for c, q in out:
            c = c.replace("<dot>", ".")
            m = REPORTED.search(c)
            if m:                                  # drop reported speech: marker + up to 9 words
                head, tail = c[:m.start()], c[m.end():].split()
                c = (head + " " + " ".join(tail[9:])).strip()
            if c:
                clean.append((c, q))
        return clean

    def cue_counts(self, text: str) -> dict[str, int]:
        """How many decision cues an utterance carries, per slot (used to keep negative controls
        unambiguous). ``<slot>_values``: how many DIFFERENT values are cued ("it's mine, I made that
        purchase" is one decision said twice, not two competing ones)."""
        j = " | ".join(c for c, _ in self.clauses(text))
        out = {}
        for slot in self.lx.slots:
            m = self._matches(slot, j)
            out[slot] = len(m)
            out[f"{slot}_values"] = len({x[2] for x in m})
        return out

    # ------------------------------------------------------------ read
    def read(self, text: str, state: str | None = None) -> F:
        """Text -> meaning card. With the dialog state (policy graph attached), a code (e.g. a
        4-digit credential) is read as such first only where the state asks for one; elsewhere
        the other cues win and the number is a code only if nothing else is said.
        (How a bare yes/no is read in the state is the ontology's job: Ontology.canonicalize.)"""
        cl = self.clauses(text)
        joined = " | ".join(c for c, _ in cl)
        for act in self.reading.get("first", []):
            if re.search(self.act_pat[act], joined):
                return F.make(act)
        codes = re.findall(self.code_spec["pattern"], joined) if self.code_spec else []
        code_state = (self.kg is None or state is None or state not in self.kg.prompt_states
                      or self.kg.accepts(state, self.code_act))
        if codes and code_state:
            return F.make(self.code_act, **{self.code_spec["slot"]: codes[-1]})
        f = self._read_rest(cl, joined)
        if codes and f.act == "UNCLEAR":
            return F.make(self.code_act, **{self.code_spec["slot"]: codes[-1]})
        return f

    def _read_rest(self, cl, joined: str) -> F:
        for c, q in cl:
            if q or self.qword.search(c):
                kb = self._kb(c)
                if kb is not None:
                    return kb
        acts = self.lx.acts
        value_acts = [a for a, sp in acts.items() if "slot" in sp]
        request_acts = [a for a, sp in acts.items() if "slots" in sp]
        for a in value_acts:
            sp = acts[a]
            v = self._pick(sp["slot"], joined)
            if v is not None:
                val, corrected = v
                req = self._request_fields(acts[sp["with_request"]], joined) if sp.get("with_request") else {}
                return F.make(a, **{sp["slot"]: val}, corrected_from=corrected, **req)
        for a in request_acts:
            req = self._request_fields(acts[a], joined)
            if req:
                return F.make(a, **req)
        for a in self.reading.get("then", []):
            if re.search(self.act_pat[a], joined):
                return F.make(a)
        cues = [(m.start(), a) for a in self.reading.get("last_cue", []) for m in re.finditer(self.act_pat[a], joined)]
        if cues:
            return F.make(max(cues)[1])
        return F.make("UNCLEAR")

    # ------------------------------------------------------------ slot resolution
    def _matches(self, slot: str, joined: str) -> list[tuple[int, int, str]]:
        """Non-overlapping cue matches of a slot's values (earliest, then longest, first)."""
        found = []
        for label in self.lx.slots[slot]["order"]:
            for m in re.finditer(self.value_pat[label], joined):
                found.append((m.start(), m.end(), label))
        found.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        kept, last_end = [], -1
        for s, e, lab in found:
            if s >= last_end:
                kept.append((s, e, lab))
                last_end = e
        return kept

    def _pick(self, slot: str, joined: str) -> tuple[str, str | None] | None:
        """The slot's value per its rule; returns (value, corrected_from)."""
        spec = self.lx.slots[slot]
        rule, order = spec["pick"], spec["order"]
        if rule == "final_with_correction":       # the latest clear answer wins (R21); a repair
            kept = self._matches(slot, joined)    # marker makes the earlier one a correction
            if not kept:
                return None
            final = kept[-1]
            corrected = None
            for s, e, lab in reversed(kept[:-1]):
                if lab != final[2] and re.search(REPAIR_MARK, joined[e:final[0]]):
                    corrected = lab
                    break
            return final[2], corrected
        cands = []
        for i, v in enumerate(order):
            ms = list(re.finditer(self.value_pat[v], joined))
            if ms:
                pos = ms[0].start() if rule == "latest_first_mention" else ms[-1].start()
                cands.append((pos, i, v))
        return (max(cands)[2], None) if cands else None

    def _request_fields(self, spec: dict, joined: str) -> dict:
        out = {}
        for role in ("action", "option"):
            slot = spec["slots"][role]
            cond = self.lx.slots[slot].get("only_if") or {}
            if any(out.get(k) != v for k, v in cond.items()):
                continue
            if role == "option" and "action" in spec["slots"] and spec["slots"]["action"] not in out:
                continue
            v = self._pick(slot, joined)
            if v is not None:
                out[slot] = v[0]
        return out

    # ------------------------------------------------------------ KB intents
    @staticmethod
    def _money(c: str) -> list[int]:
        vals = [int(float(x)) for x in re.findall(r"\$(\d+(?:\.\d+)?)", c)]
        vals += [int(float(x)) for x in re.findall(r"\b(\d+(?:\.\d+)?) dollars\b", c)]
        if re.search(r"\b(free|doesnt cost anything|no cost|costs nothing|no charge)\b", c):
            vals.append(0)
        return vals

    def _kb(self, c: str) -> F | None:
        has = lambda p: re.search(p, c) is not None
        ents = set()
        for v in self.lx.slots[self.lx.kb["entity_slot"]]["order"]:
            if has(self.value_pat[v]):
                ents.add(v)
        ent = next(iter(ents)) if len(ents) == 1 else None
        money = self._money(c)
        fee_cue = has(r"\b(how much|cost|costs|fee|price|pay)\b")
        days_cue = has(r"\b(how long|business days|days|arrive|take|timing|when would)\b")
        conf_cue = has(r"\b(confirm|confirmation|approve)\b")
        K = lambda intent, **kw: F.make("ASK_KB", intent=intent, **kw)
        if has(r"\b(refund|money back|get back|reimburs)"):
            return K("refund")
        if has(r"\bfraud\b"):
            return K("fraud")
        if has(r"\bskip\b.{0,30}\b(verif|check|security)|\bwithout\b.{0,25}\b(verif|security)|\bhave to verify\b"):
            return K("skip_auth")
        if has(r"\b(steps|process|how this works|step by step)\b"):
            return K("procedure")
        m = re.search(r"\b(\d+) days? (ago|old|back)\b", c)
        if m:
            return K("window_check", days=int(m.group(1)))
        if has(r"\b(how far back|how old can|time window|lookback|review window)\b"):
            return K("limit", key="lookback_days")
        if has(r"\b(attempts|tries|try to verify|times can i try)\b"):
            return K("limit", key="max_auth_attempts")
        if has(r"\b(how many|most|max|maximum|limit|cap)\b.{0,30}\b(transactions?|charges?)\b"):
            return K("limit", key="max_transactions_per_session")
        if (has(r"\b(how many|number of)\b.{0,40}\b(delivery|shipping|ways|choices|options)\b") and not days_cue) \
                or has(r"choices do i have"):
            return K("count_options")
        if ents == {"expedited_delivery"} and (has(r"\b(total|overall|everything)\b") or fee_cue + days_cue + conf_cue >= 2):
            return K("expedited_total")
        if conf_cue and has(r"\bwhich\b"):
            return K("needs_confirmation")
        if has(r"\b(difference|how much more|how much extra|more than|compared|versus)\b"):
            return K("fee_difference")
        fast = has(r"\b(faster|fastest|quickest|quicker|sooner|quickly)\b")
        if fast and fee_cue:
            return K("faster_option_fee")
        if fast:
            return K("faster")
        if has(r"\b(cheaper|cheapest|costs less|save money|less expensive)\b"):
            return K("cheaper")
        if not ents and (fee_cue or days_cue):
            # hallucination probes: the bot must abstain, so the meaning card must say what was asked
            if fee_cue and re.search(self.value_pat["SECURE_CARD"], c):
                return K("unstated_fact", attr="fee", subject="SECURE_CARD")
            m = self.option_name.search(c)
            if m:
                words = [w for w in re.split(r"[ -]", m.group(1)) if w and w not in NAME_STOP]
                if words:
                    return K("unknown_option", attr="fee" if fee_cue else "days", name="_".join(words + ["delivery"]))
        if ent and len(money) >= 3:
            return K("mcq", attr="fee", entity=ent, options=tuple(money))
        if ent and len(set(money)) == 1 and not has(r"\bhow much\b"):
            return K("plausibility", attr="fee", entity=ent, value=str(money[0]))
        if ent and fee_cue:
            return K("value", attr="fee", entity=ent)
        if ent and days_cue:
            return K("value", attr="days", entity=ent)
        return None
