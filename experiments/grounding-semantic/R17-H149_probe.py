"""R17-H149 stage 1 - held-out BARE-ASSERTION prose probe.

Each pair is (positive assertion, negative twin) over one prose passage:

  positive  - a single-sentence assertion restating one proposition of the
              passage in NEW words (voice recast + verb synonym; never a
              verbatim substring of the passage)
  negative  - the SAME sentence with exactly ONE proposition-level substitution

Three families, each verifiable only against the passage's meaning - no family
can be decided by matching a numeral or a string against an adjacent verbatim
span (the R17-H148 lesson):

  role_swap      - the two arguments of an asymmetric relation exchange roles;
                   positive and negative are token-multiset identical
  direction_flip - the relation's direction word is replaced by its antonym,
                   both drawn from one closed antonym pair so each word appears
                   as often in positives as in negatives
  entity_swap    - the object is replaced by another entity of the SAME passage
                   that fills the object role of a DIFFERENT relation; emitted
                   in mirrored couples so each entity appears once positive and
                   once negative

No LLM is used: positives are produced by dependency-parse extraction plus a
curated synonym/voice recast table, so the realized paraphrase depth is
structural (voice, tense-preserving verb synonym, determiner normalisation)
with content words retained.  That depth is MEASURED and reported, not claimed.

CPU only:
  uv run python experiments/grounding-semantic/R17-H149_probe.py
"""
import collections
import json
import pathlib
import random
import re

import numpy as np
import polars as pl
import spacy

HERE = pathlib.Path(__file__).parent
PASSAGES = HERE / "R17-H149_passages.parquet"
OUT = HERE / "R17-H149_probe.parquet"
MANIFEST = HERE / "R17-H149_probe_manifest.json"
AUDIT = HERE / "R17-H149_audit_sample.parquet"

SEED = 20260811
N_FOLDS = 5
AUDIT_N = 100
TARGET_PER_FAMILY = 700
PAIRS_PER_PASSAGE = 3
PAIRS_PER_DOC = 400   # faa-amt is 3 handbooks; the passage cap is the real one

# --------------------------------------------------------------------------- #
# lexicon
# --------------------------------------------------------------------------- #
# lemma -> (base, 3sg, past, past-participle) of a SYNONYM (never the lemma itself)
ASYM = {
    "cause": ("bring about", "brings about", "brought about", "brought about"),
    "induce": ("trigger", "triggers", "triggered", "triggered"),
    "trigger": ("set off", "sets off", "set off", "set off"),
    "produce": ("generate", "generates", "generated", "generated"),
    "generate": ("produce", "produces", "produced", "produced"),
    "create": ("make", "makes", "made", "made"),
    "regulate": ("control", "controls", "controlled", "controlled"),
    "control": ("regulate", "regulates", "regulated", "regulated"),
    "affect": ("influence", "influences", "influenced", "influenced"),
    "influence": ("affect", "affects", "affected", "affected"),
    "determine": ("dictate", "dictates", "dictated", "dictated"),
    "predict": ("forecast", "forecasts", "forecast", "forecast"),
    "contain": ("hold", "holds", "held", "held"),
    "provide": ("supply", "supplies", "supplied", "supplied"),
    "supply": ("provide", "provides", "provided", "provided"),
    "allow": ("permit", "permits", "permitted", "permitted"),
    "permit": ("allow", "allows", "allowed", "allowed"),
    "prevent": ("block", "blocks", "blocked", "blocked"),
    "block": ("prevent", "prevents", "prevented", "prevented"),
    "target": ("act on", "acts on", "acted on", "acted on"),
    "carry": ("transmit", "transmits", "transmitted", "transmitted"),
    "transmit": ("carry", "carries", "carried", "carried"),
    "absorb": ("take up", "takes up", "took up", "taken up"),
    "deliver": ("supply", "supplies", "supplied", "supplied"),
    "drive": ("power", "powers", "powered", "powered"),
    "power": ("drive", "drives", "drove", "driven"),
    "actuate": ("operate", "operates", "operated", "operated"),
    "operate": ("actuate", "actuates", "actuated", "actuated"),
    "support": ("back", "backs", "backed", "backed"),
    "protect": ("shield", "shields", "shielded", "shielded"),
    "replace": ("supersede", "supersedes", "superseded", "superseded"),
    "exceed": ("surpass", "surpasses", "surpassed", "surpassed"),
    "remove": ("eliminate", "eliminates", "eliminated", "eliminated"),
    "eliminate": ("remove", "removes", "removed", "removed"),
    "encode": ("specify", "specifies", "specified", "specified"),
    "require": ("need", "needs", "needed", "needed"),
    "use": ("employ", "employs", "employed", "employed"),
    "employ": ("use", "uses", "used", "used"),
    "release": ("give off", "gives off", "gave off", "given off"),
    "maintain": ("keep", "keeps", "kept", "kept"),
    "detect": ("identify", "identifies", "identified", "identified"),
    "identify": ("detect", "detects", "detected", "detected"),
    "measure": ("gauge", "gauges", "gauged", "gauged"),
    "seal": ("close off", "closes off", "closed off", "closed off"),
    "convert": ("change", "changes", "changed", "changed"),
    "transfer": ("move", "moves", "moved", "moved"),
    "direct": ("route", "routes", "routed", "routed"),
    "hold": ("retain", "retains", "retained", "retained"),
    "retain": ("hold", "holds", "held", "held"),
    "govern": ("control", "controls", "controlled", "controlled"),
    "house": ("contain", "contains", "contained", "contained"),
    "feed": ("supply", "supplies", "supplied", "supplied"),
    "follow": ("come after", "comes after", "came after", "come after"),
    "precede": ("come before", "comes before", "came before", "come before"),
    "outweigh": ("exceed", "exceeds", "exceeded", "exceeded"),
    "damage": ("harm", "harms", "harmed", "harmed"),
    "destroy": ("wreck", "wrecks", "wrecked", "wrecked"),
    "kill": ("destroy", "destroys", "destroyed", "destroyed"),
    "heat": ("warm", "warms", "warmed", "warmed"),
    "express": ("produce", "produces", "produced", "produced"),
    "secrete": ("release", "releases", "released", "released"),
    "recruit": ("attract", "attracts", "attracted", "attracted"),
    "mediate": ("bring about", "brings about", "brought about", "brought about"),
    "modulate": ("adjust", "adjusts", "adjusted", "adjusted"),
    "disrupt": ("break up", "breaks up", "broke up", "broken up"),
    "restore": ("bring back", "brings back", "brought back", "brought back"),
    "initiate": ("start", "starts", "started", "started"),
    "terminate": ("end", "ends", "ended", "ended"),
    "sustain": ("uphold", "upholds", "upheld", "upheld"),
    "confer": ("grant", "grants", "granted", "granted"),
    "consume": ("use up", "uses up", "used up", "used up"),
    "emit": ("give off", "gives off", "gave off", "given off"),
    "displace": ("push out", "pushes out", "pushed out", "pushed out"),
    "occupy": ("take up", "takes up", "took up", "taken up"),
    "penetrate": ("pass through", "passes through", "passed through", "passed through"),
    "withstand": ("resist", "resists", "resisted", "resisted"),
    "resist": ("withstand", "withstands", "withstood", "withstood"),
    "lubricate": ("oil", "oils", "oiled", "oiled"),
    "compress": ("squeeze", "squeezes", "squeezed", "squeezed"),
    "rotate": ("turn", "turns", "turned", "turned"),
    "monitor": ("track", "tracks", "tracked", "tracked"),
    "indicate": ("show", "shows", "showed", "shown"),
    "trap": ("catch", "catches", "caught", "caught"),
    "guide": ("steer", "steers", "steered", "steered"),
    "cool": ("chill", "chills", "chilled", "chilled"),
    "dissolve": ("break down", "breaks down", "broke down", "broken down"),
    "activate": ("switch on", "switches on", "switched on", "switched on"),
    "encounter": ("meet", "meets", "met", "met"),
    "cover": ("coat", "coats", "coated", "coated"),
    "load": ("stress", "stresses", "stressed", "stressed"),
}

UP = {"increase", "elevate", "raise", "enhance", "improve", "boost", "promote",
      "stimulate", "accelerate", "strengthen", "amplify", "augment", "rise", "grow",
      "intensify", "escalate", "worsen", "aggravate", "exacerbate"}
DOWN = {"reduce", "decrease", "lower", "diminish", "inhibit", "suppress", "impair",
        "weaken", "restrict", "limit", "slow", "delay", "attenuate", "minimize",
        "fall", "drop", "decline", "dampen",
        "alleviate", "mitigate", "blunt"}

# closed antonym pairs; both members of a pair are generic magnitude verbs, so the
# positive verb and the negative verb are drawn from the SAME pair (surface parity)
# ONE antonym pair per construction kind: with several pairs in play the per-pair
# up/down balance (which is what keeps a direction word from marking the label)
# has to be struck inside each pair, and the smallest bucket then governs supply
ANT_PAIRS = [
    (("raise", "raises", "raised", "raised"), ("lower", "lowers", "lowered", "lowered")),
]

# intransitive antonym pairs, for 'X rose' / 'X fell' style propositions
INTRANS_PAIRS = [
    (("rise", "rises", "rose", "risen"), ("fall", "falls", "fell", "fallen")),
]

# source verbs whose direction is about QUALITY, not magnitude - pairing them with
# a magnitude antonym ("the outcome was boosted") reads wrong, so they get their own
QUALITY = {"improve", "worsen", "aggravate", "exacerbate", "impair"}
QUALITY_PAIR = (("enhance", "enhances", "enhanced", "enhanced"),
                ("worsen", "worsens", "worsened", "worsened"))
# verbs that genuinely take an intransitive change-of-magnitude reading; a
# directional transitive parsed without an object is a mis-parse, not an intransitive
INTRANS_OK = {"increase", "decrease", "rise", "fall", "grow", "decline", "drop",
              "improve", "worsen", "halve",
              "accelerate", "intensify", "escalate", "diminish", "lengthen",
              "widen", "contract"}
# a finite clause head; participles and infinitival / relative heads mis-assign roles
# structured-abstract section headings sit inline in SciFact sentences and get
# swept into a span ("CONCLUSIONS enhanced.") - any span carrying one is refused
HEADERS = {"CONCLUSION", "CONCLUSIONS", "RESULT", "RESULTS", "METHOD", "METHODS",
           "BACKGROUND", "OBJECTIVE", "OBJECTIVES", "PURPOSE", "CONTEXT", "DESIGN",
           "SETTING", "PATIENTS", "INTERVENTION", "INTERVENTIONS", "MEASUREMENTS",
           "IMPORTANCE", "SIGNIFICANCE", "KEY", "POINTS", "RECENT", "ADVANCES",
           "CRITICAL", "ISSUES", "FUTURE", "DIRECTIONS", "SUMMARY", "ABSTRACT",
           "INTRODUCTION", "DISCUSSION", "AIM", "AIMS", "RATIONALE", "MAIN",
           "OUTCOME", "MEASURES", "UNLABELLED", "TRIAL", "REGISTRATION",
           "STATEMENT", "FINDINGS", "INTERPRETATION", "FUNDING"}
BAD_VERB_DEP = {"advcl", "xcomp", "acl", "relcl", "pcomp", "csubj", "csubjpass", "dep"}

MODALS = {"may", "might", "could", "would", "should", "can", "must", "will", "shall"}
# a sentence carrying any of these is refused outright - its proposition is
# hedged, negated, conditional or attributed, so a flat assertion may be untrue
SENT_VETO = {"if", "whether", "unless", "although", "though", "not", "no", "never",
             "when", "while", "whereas", "until", "once", "compared", "versus",
             "vs", "relative", "except", "despite", "than",
             "rarely", "seldom", "without", "unclear", "unknown", "unlikely", "nor",
             "suggest", "suggests", "suggested", "hypothesize", "hypothesized",
             "assume", "assumed", "propose", "proposed", "appear", "appears",
             "appeared", "seem", "seems", "seemed", "believe", "believed",
             "possibly", "perhaps", "maybe", "either", "neither", "unable",
             "cannot", "n't", "?"} | MODALS
BAD_DET = {"this", "that", "these", "those", "its", "their", "his", "her", "our",
           "your", "my", "such", "another", "other", "same", "which", "what",
           "some", "any", "each", "every", "most", "many", "few", "both", "all",
           "no", "several", "various"}
LIGHT = {"it", "they", "them", "one", "ones", "thing", "things", "part", "parts",
         "type", "types", "kind", "kinds", "number", "amount", "way", "ways",
         "case", "cases", "time", "times", "result", "results", "study", "studies",
         "example", "examples", "use", "uses", "figure", "table", "section",
         "chapter", "paragraph", "author", "authors", "data", "information",
         "addition", "order", "term", "terms", "fact", "reason", "point",
         "person", "people", "man", "men", "woman", "women", "group", "groups"}
WORD = re.compile(r"[A-Za-z][A-Za-z-]+")
NUM = re.compile(r"\d")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def toks(s):
    return [w.lower() for w in WORD.findall(s)]


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def max_ngram_overlap(claim, passage):
    """Longest contiguous word n-gram the claim shares with the passage."""
    a, b = toks(claim), toks(passage)
    bs = set()
    best, n = 0, 1
    while n <= len(a):
        grams = {" ".join(b[i:i + n]) for i in range(len(b) - n + 1)}
        hit = any(" ".join(a[i:i + n]) in grams for i in range(len(a) - n + 1))
        if not hit:
            break
        best, n = n, n + 1
        bs = grams
    del bs
    return best


BAD_DEP = {"conj", "cc", "appos", "relcl", "acl", "advcl", "parataxis", "dep"}
BAD_POS = {"PUNCT", "VERB", "PRON", "SYM"}
MAX_SPAN = 12
HYPHEN = {"-", "\u2013", "/"}
# deverbal / abstract heads too vague to pin a proposition on when they stand alone
VAGUE = {"activation", "formation", "presence", "effect", "level", "levels",
         "expression", "delivery", "administration", "treatment", "exposure",
         "activity", "function", "process", "system", "condition", "conditions",
         "increase", "decrease", "change", "changes", "response", "production",
         "development", "growth", "reduction", "induction", "inhibition",
         "association", "correlation", "mechanism", "model", "method", "approach",
         "analysis", "patient", "patients", "subject", "subjects", "control",
         "controls", "sample", "samples", "value", "values", "rate", "rates",
         "ratio", "factor", "factors", "role", "need", "state", "form", "size",
         "area", "region", "surface", "side", "end", "unit", "units", "line",
         "lines", "point", "points", "material", "component", "components"}


def symbolic(t):
    """A token that reads as a name/identifier rather than sentence-initial casing."""
    s = t.text
    if not re.search(r"[A-Za-z]", s):
        return False
    return bool(re.search(r"[A-Z0-9]", s[1:])) or "-" in s


def np_span(tok, chunks):
    """The full nominal subtree headed by `tok`, normalised; None if unusable.

    The whole subtree is taken (compounds, adjectives and prepositional
    modifiers included) so the span cannot state a proposition broader than the
    passage's; any coordination, relative clause or bracketed material makes the
    span untrustworthy and is refused outright.
    """
    if tok.pos_ not in ("NOUN", "PROPN"):
        return None
    if tok.lower_ in LIGHT or tok.lemma_.lower() in LIGHT:
        return None
    if tok.dep_ == "conj":            # 'blockade of Tim-1 and Tim-4' - the conjunct
        return None                   # alone is not the argument of the relation
    sub = sorted(tok.subtree, key=lambda t: t.i)
    if not (1 <= len(sub) <= MAX_SPAN):
        return None
    if sub[-1].i - sub[0].i + 1 != len(sub):
        return None
    for t in sub:
        if t is tok:
            continue
        if t.dep_ in BAD_DEP:
            return None
        if t.pos_ in BAD_POS and not (t.pos_ == "PUNCT" and t.text in HYPHEN):
            return None
        if t.pos_ == "ADV" or t.dep_ in ("advmod", "npadvmod", "quantmod", "predet"):
            return None
        if NUM.search(t.text) and not symbolic(t):
            return None
    if any(t.text.isupper() and t.text in HEADERS for t in sub):
        return None
    if any(t.dep_ in BAD_DEP for t in tok.children):
        return None
    words = list(sub)
    det = None
    if words[0].pos_ == "DET" or words[0].dep_ in ("det", "poss"):
        det = words[0].lower_
        words = words[1:]
    if det in BAD_DET or (det is not None and det not in ("a", "an", "the")):
        return None
    if not words:
        return None
    if any(w.lower_ in BAD_DET for w in words):
        return None
    if any(w.pos_ == "DET" for w in words):
        return None
    # a bare one-word common noun is too vague to pin a proposition on; names and
    # identifiers (CRP, IL-10, GPIIb-IIIa) are specific enough on their own
    if len(words) < 2 and not symbolic(tok):
        return None
    doc = tok.doc
    txt = doc[words[0].i:words[-1].i + 1].text.strip()
    if words[0].i == words[0].sent.start and words[0].pos_ != "PROPN" \
            and not symbolic(words[0]):
        txt = txt[0].lower() + txt[1:]
    if not (3 <= len(txt) <= 80) or not WORD.search(txt):
        return None
    if re.search(r"[()\[\]{}<>|/\\]|\s,", txt):
        return None
    article = "" if (tok.pos_ == "PROPN" or symbolic(tok)) else "the "
    plural = "Plur" in tok.morph.get("Number")
    return {"text": txt, "article": article, "plural": plural,
            "lemmas": {w.lemma_.lower() for w in words if w.pos_ in ("NOUN", "PROPN", "ADJ")},
            "root": tok.lemma_.lower()}


def sent_ok(sent):
    for t in sent:
        if t.lower_ in SENT_VETO or t.lemma_.lower() in SENT_VETO:
            return False
    if sent.text.strip().endswith("?") or '"' in sent.text:
        return False
    return True


def extract(doc):
    """(subject, verb, object) propositions of a parsed passage."""
    out = []
    chunks = {c.root.i: c for c in doc.noun_chunks}
    for ordinal, sent in enumerate(doc.sents):
        if not sent_ok(sent):
            continue
        for v in sent:
            if v.pos_ != "VERB":
                continue
            lem = v.lemma_.lower()
            direction = 1 if lem in UP else (-1 if lem in DOWN else 0)
            if lem not in ASYM and direction == 0:
                continue
            if any(c.dep_ == "neg" for c in v.children):
                continue
            if any(c.dep_ == "aux" and c.lower_ in MODALS for c in v.children):
                continue
            if v.dep_ in BAD_VERB_DEP:
                continue
            if v.tag_ in ("VBG", "VBN") and not any(c.dep_ in ("aux", "auxpass")
                                                    for c in v.children):
                continue
            subj = obj = None
            passive = agent_subj = False
            for c in v.children:
                if c.dep_ == "nsubj":
                    subj = c
                elif c.dep_ in ("dobj", "obj"):
                    obj = c
                elif c.dep_ == "nsubjpass":
                    obj, passive = c, True
                elif c.dep_ == "agent":
                    for g in c.children:
                        if g.dep_ == "pobj":
                            subj, agent_subj = g, True
            if any(c.dep_ in ("conj", "cc") for c in v.children):
                continue
            past = "Past" in v.morph.get("Tense") or (passive and v.tag_ == "VBN")
            if direction != 0 and (subj is None) != (obj is None):
                # one-argument directional proposition: 'X rose' / 'X was reduced'.
                # a bare object with no subject is an infinitival / participial
                # transitive - not an intransitive reading - and is refused
                if obj is not None and not passive:
                    continue
                if agent_subj:                       # 'inhibited by FTO' - FTO is the
                    continue                         # agent, not the thing that moved
                if any(c.dep_ == "prep" for c in v.children):
                    continue                         # 'falls in between', 'limited to'
                if subj is not None and lem not in INTRANS_OK:
                    continue                         # a transitive verb parsed without
                                                     # an object is a mis-parse
                if sum(1 for t in sent if t.pos_ == "VERB"
                       and t.dep_ in ("ROOT", "conj", "ccomp")) != 1:
                    continue                         # a coordinated VP moves the object

                lone = np_span(subj if subj is not None else obj, chunks)
                if lone is None:
                    continue
                out.append({"lemma": lem, "direction": direction, "a": lone, "b": None,
                            "kind": "passive_noagent" if passive else "intrans",
                            "past": bool(past), "sent": sent.text.strip(),
                            "sent_i": sent.start, "sent_ord": ordinal})
                continue
            if subj is None or obj is None:
                continue
            a, b = np_span(subj, chunks), np_span(obj, chunks)
            if a is None or b is None:
                continue
            if a["lemmas"] & b["lemmas"] or a["root"] == b["root"]:
                continue
            if a["text"].lower() in b["text"].lower() or b["text"].lower() in a["text"].lower():
                continue
            out.append({"lemma": lem, "direction": direction, "a": a, "b": b,
                        "kind": "transitive",
                        "past": bool(past), "sent": sent.text.strip(),
                        "sent_i": sent.start, "sent_ord": ordinal})
    return out


def active(a, b, forms, past):
    """'The A verbs the B.'"""
    v = forms[2] if past else (forms[0] if a["plural"] else forms[1])
    s = f"{a['article']}{a['text']} {v} {b['article']}{b['text']}."
    return s[0].upper() + s[1:]


def passive(a, b, forms, past):
    """'The B is/was verbed by the A.'"""
    aux = ("were" if b["plural"] else "was") if past else ("are" if b["plural"] else "is")
    s = f"{b['article']}{b['text']} {aux} {forms[3]} by {a['article']}{a['text']}."
    return s[0].upper() + s[1:]


# --------------------------------------------------------------------------- #
# family builders
# --------------------------------------------------------------------------- #
def build_role_swap(rec, voice):
    """Positive and negative are token-multiset identical; only the roles move."""
    forms = ASYM.get(rec["lemma"])
    if forms is None:
        return None
    a, b, past = rec["a"], rec["b"], rec["past"]
    if voice == "passive":                       # positive reverses passage order
        pos, neg = passive(a, b, forms, past), passive(b, a, forms, past)
    else:                                        # positive keeps passage order
        pos, neg = active(a, b, forms, past), active(b, a, forms, past)
    return {"claim_pos": pos, "claim_neg": neg, "direction": voice,
            "balance_key": voice,
            "subst_from": a["text"], "subst_to": b["text"]}


def intrans(a, forms, past):
    """'The A rises.'"""
    v = forms[2] if past else (forms[0] if a["plural"] else forms[1])
    s = f"{a['article']}{a['text']} {v}."
    return s[0].upper() + s[1:]


def be_pp(a, forms, past):
    """'The A was raised.' - a passive with no agent in the source."""
    aux = ("were" if a["plural"] else "was") if past else ("are" if a["plural"] else "is")
    s = f"{a['article']}{a['text']} {aux} {forms[3]}."
    return s[0].upper() + s[1:]


def build_direction_flip(rec, voice, pair_ix):
    up_down = rec["direction"] > 0
    kind = rec.get("kind", "transitive")
    if rec["lemma"] in QUALITY:
        up, down = QUALITY_PAIR
    elif kind == "intrans":
        up, down = INTRANS_PAIRS[pair_ix % len(INTRANS_PAIRS)]
    else:
        up, down = ANT_PAIRS[pair_ix % len(ANT_PAIRS)]
    same, anti = (up, down) if up_down else (down, up)
    if same[0] == rec["lemma"] or anti[0] == rec["lemma"]:
        return None
    pair_tag = f"{up[0]}/{down[0]}"
    a, b, past = rec["a"], rec["b"], rec["past"]
    if kind == "intrans":
        pos, neg, form = intrans(a, same, past), intrans(a, anti, past), "intrans"
    elif kind == "passive_noagent":
        pos, neg, form = be_pp(a, same, past), be_pp(a, anti, past), "passive_noagent"
    else:
        fn = passive if voice == "passive" else active
        pos, neg, form = fn(a, b, same, past), fn(a, b, anti, past), voice
    return {"claim_pos": pos, "claim_neg": neg,
            "direction": f"{'up' if up_down else 'down'}|{form}",
            # balancing on the ANTONYM PAIR as well as the source direction is what
            # keeps each direction word as frequent in positives as in negatives
            "balance_key": f"{pair_tag}|{'up' if up_down else 'down'}",
            "subst_from": same[0], "subst_to": anti[0]}


def build_entity_swap(rec, other, voice):
    """Object replaced by the object of a DIFFERENT relation in the same passage."""
    forms = ASYM.get(rec["lemma"])
    if forms is None:
        return None
    a, b, c, past = rec["a"], rec["b"], other["b"], rec["past"]
    fn = passive if voice == "passive" else active
    return {"claim_pos": fn(a, b, forms, past), "claim_neg": fn(a, c, forms, past),
            "direction": voice, "balance_key": voice,
            "subst_from": b["text"], "subst_to": c["text"]}


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def candidates(passages, nlp, rng):
    """Family-tagged candidate pairs, capped per passage/document."""
    cands = collections.defaultdict(list)
    texts = passages["text"].to_list()
    meta = passages.select(["corpus", "doc_id", "passage_id"]).rows(named=True)
    n_rel = n_pass_with_rel = 0
    voice_ix = 0
    for k, doc in enumerate(nlp.pipe(texts, batch_size=48, n_process=8)):
        m = meta[k]
        rels = extract(doc)
        n_rel += len(rels)
        if rels:
            n_pass_with_rel += 1
        base = dict(corpus=m["corpus"], doc_id=m["doc_id"], passage_id=m["passage_id"],
                    chunk=texts[k])

        for r in rels:
            voice = "passive" if voice_ix % 2 == 0 else "active"
            voice_ix += 1
            if r["direction"] != 0:
                spec = build_direction_flip(r, voice, (voice_ix // 2) % len(ANT_PAIRS))
                fam = "direction_flip"
            else:
                spec = build_role_swap(r, voice)
                fam = "role_swap"
            if spec:
                cands[fam].append({**base, **spec, "neg_family": fam,
                                   "source_sent": r["sent"], "lemma": r["lemma"]})

        # entity_swap (object replaced by another entity of the same passage) was
        # designed and then REFUSED: at 20 mirrored pairs it is unusable for a
        # per-family read, and the hand audit found its negatives can be entailed
        # by the passage's own causal chain
    return cands, {"relations": n_rel, "passages_with_relations": n_pass_with_rel}


def select(cands, rng):
    """Per-passage / per-document caps, direction balance, family targets."""
    rows, dropped = [], collections.Counter()
    for fam, items in cands.items():
        rng.shuffle(items)
        # entity_swap couples must survive or die together
        if fam == "entity_swap":
            by_mirror = collections.defaultdict(list)
            for it in items:
                by_mirror[it["mirror"]].append(it)
            units = [v for v in by_mirror.values() if len(v) == 2]
            dropped[f"{fam}_unmirrored"] = sum(1 for v in by_mirror.values() if len(v) != 2)
            rng.shuffle(units)
        else:
            units = [[it] for it in items]

        # balance the two sides of each balance group (voice for role_swap; up vs
        # down INSIDE each antonym pair for direction_flip) before trimming
        groups = collections.defaultdict(lambda: collections.defaultdict(list))
        for u in units:
            g, _, side = u[0]["balance_key"].rpartition("|")
            groups[g][side].append(u)
        balanced = []
        for g in sorted(groups):
            sides = groups[g]
            n = min(len(v) for v in sides.values()) if len(sides) > 1 else 0
            for k in sorted(sides):
                balanced += sides[k][:n]
                dropped[f"{fam}_direction_balance"] += len(sides[k]) - n
        rng.shuffle(balanced)

        per_pass, per_doc, kept = collections.Counter(), collections.Counter(), []
        for u in balanced:
            p, d = u[0]["passage_id"], u[0]["doc_id"]
            if per_pass[p] + len(u) > PAIRS_PER_PASSAGE or per_doc[d] + len(u) > PAIRS_PER_DOC:
                dropped[f"{fam}_cap"] += len(u)
                continue
            per_pass[p] += len(u)
            per_doc[d] += len(u)
            kept.append(u)
        # trim to target, keeping the direction balance
        by_dir = collections.defaultdict(list)
        for u in kept:
            by_dir[u[0]["balance_key"]].append(u)
        quota = TARGET_PER_FAMILY // max(len(by_dir), 1)
        final = []
        for k in sorted(by_dir):
            final += by_dir[k][:quota]
        for u in final:
            rows += u
    return rows, dict(dropped)


def emit(units):
    rows = []
    for i, u in enumerate(units):
        pid = f"h149-{i:06d}"
        for lab, claim in ((1, u["claim_pos"]), (0, u["claim_neg"])):
            seq = u["passage_id"].rsplit(":", 1)[-1]
            fold_doc = (f'{u["doc_id"]}#{int(seq) // 50}' if seq.isdigit()
                        and u["corpus"] != "scifact" else u["doc_id"])
            rows.append({"pair_id": pid, "label": lab, "claim": claim, "chunk": u["chunk"],
                         "fold_doc": fold_doc,
                         "neg_family": u["neg_family"], "direction": u["direction"],
                         "corpus": u["corpus"], "doc_id": u["doc_id"],
                         "passage_id": u["passage_id"], "source_sent": u["source_sent"],
                         "lemma": u["lemma"], "subst_from": u["subst_from"],
                         "subst_to": u["subst_to"],
                         "mirror": u.get("mirror", "")})
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def verify(df, rng):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    out = {}
    claims, labels = df["claim"].to_list(), df["label"].to_list()

    doc_key = {d: k for d, k in df.filter(pl.col("label") == 1)
               .group_by("fold_doc")
               .agg((pl.col("neg_family") + ":" + pl.col("direction")).first()).iter_rows()}
    strata = collections.defaultdict(list)
    for d in sorted(doc_key):
        strata[doc_key[d]].append(d)
    fold_of, i = {}, 0
    for k in sorted(strata):
        ds = strata[k]
        rng.shuffle(ds)
        for d in ds:
            fold_of[d] = i % N_FOLDS
            i += 1
    folds = np.array([fold_of[d] for d in df["fold_doc"].to_list()])
    score = np.zeros(len(df))
    idx = np.arange(len(df))
    for f in range(N_FOLDS):
        tr_i, te_i = idx[folds != f], idx[folds == f]
        if not len(te_i):
            continue
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                              max_features=300_000, sublinear_tf=True)
        Xtr = vec.fit_transform([claims[j] for j in tr_i])
        Xte = vec.transform([claims[j] for j in te_i])
        clf = LogisticRegression(solver="liblinear", C=4.0, tol=1e-7, max_iter=3000)
        clf.fit(Xtr, [labels[j] for j in tr_i])
        score[te_i] = clf.decision_function(Xte)
    probe = auroc(labels, score)
    out["claim_only_tfidf_auroc"] = {
        "value": round(probe, 4), "bar": "< 0.55", "pass": bool(probe < 0.55),
        "scoring": f"{N_FOLDS}-fold document-disjoint, out of fold, "
                   "direction-stratified, liblinear tol 1e-7",
        "documents": len(fold_of), "rows": len(df)}

    scored = df.select(["pair_id", "label", "neg_family"]).with_columns(pl.Series("score", score))
    fam_acc, worst = {}, 0.0
    for fam, sub in scored.group_by("neg_family"):
        piv = sub.pivot(on="label", index="pair_id", values="score",
                        aggregate_function="first").drop_nulls()
        pos, neg = piv["1"].to_numpy(), piv["0"].to_numpy()
        acc = float(((pos > neg) + 0.5 * (pos == neg)).mean())
        fam_acc[fam[0]] = {"acc": round(acc, 4), "pairs": len(piv)}
        worst = max(worst, acc)
    out["within_pair_claim_only_accuracy"] = {
        "per_family": fam_acc, "worst": round(worst, 4), "bar": "< 0.60",
        "pass": bool(worst < 0.60)}

    # --- paraphrase depth, measured not claimed -----------------------------
    sub = df.filter(pl.col("label") == 1)
    verb_sub = df.filter(pl.col("label") == 0)
    ng = [max_ngram_overlap(c, ch) for c, ch in
          zip(sub["claim"].to_list()[:1500], sub["chunk"].to_list()[:1500])]
    jac = []
    for c, s in zip(sub["claim"].to_list(), sub["source_sent"].to_list()):
        a, b = set(toks(c)), set(toks(s))
        jac.append(len(a & b) / max(len(a | b), 1))
    verbatim_pos = [c.rstrip(".") in ch for c, ch in
                    zip(sub["claim"].to_list(), sub["chunk"].to_list())]
    verbatim_neg = [c.rstrip(".") in ch for c, ch in
                    zip(verb_sub["claim"].to_list(), verb_sub["chunk"].to_list())]
    out["paraphrase_depth"] = {
        "positive_verbatim_substring_rate": round(float(np.mean(verbatim_pos)), 6),
        "negative_verbatim_substring_rate": round(float(np.mean(verbatim_neg)), 6),
        "bar": "positive verbatim rate == 0", "pass": not any(verbatim_pos),
        "max_contiguous_ngram_with_passage": {
            "mean": round(float(np.mean(ng)), 3), "median": float(np.median(ng)),
            "p90": float(np.percentile(ng, 90)), "max": int(np.max(ng)),
            "sampled_positives": len(ng)},
        "token_jaccard_claim_vs_source_sentence": {
            "mean": round(float(np.mean(jac)), 4), "median": round(float(np.median(jac)), 4),
            "p90": round(float(np.percentile(jac, 90)), 4)},
        "note": "template + synonym + voice recast, no LLM; content words are "
                "retained by construction, so depth is structural/lexical-verb "
                "level, not full rewriting"}

    # --- surface parity ------------------------------------------------------
    piv = (df.select(["pair_id", "label", "claim", "neg_family"])
             .pivot(on="label", index=["pair_id", "neg_family"], values="claim",
                    aggregate_function="first").drop_nulls())
    same_multiset = [sorted(toks(p)) == sorted(toks(n))
                     for p, n in zip(piv["1"].to_list(), piv["0"].to_list())]
    piv = piv.with_columns(pl.Series("same_multiset", same_multiset))
    out["surface_parity"] = {
        "identical_token_multiset_rate_per_family": {
            f[0]: round(float(s["same_multiset"].mean()), 4)
            for f, s in piv.group_by("neg_family")},
        "claim_char_length_auroc": round(auroc(labels, [len(c) for c in claims]), 4),
        "claim_token_count_auroc": round(auroc(labels, [len(toks(c)) for c in claims]), 4),
        "numeral_presence_auroc": round(
            auroc(labels, [1.0 if NUM.search(c) else 0.0 for c in claims]), 4),
        "passage_token_overlap_auroc": round(auroc(labels, [
            len(set(toks(c)) & set(toks(ch))) / max(len(set(toks(c))), 1)
            for c, ch in zip(claims, df["chunk"].to_list())]), 4),
        "bar": "each AUROC in [0.45, 0.55]"}
    devs = [abs(out["surface_parity"][k] - 0.5) for k in
            ("claim_char_length_auroc", "claim_token_count_auroc",
             "numeral_presence_auroc", "passage_token_overlap_auroc")]
    out["surface_parity"]["pass"] = bool(max(devs) <= 0.05)

    # --- substituted-string balance (the H148 asymmetry fix) -----------------
    bal = {}
    for fam, s in df.group_by("neg_family"):
        if fam[0] == "role_swap":
            continue                       # both strings occur in both twins
        pos_words = collections.Counter()
        neg_words = collections.Counter()
        for lab, a, b in zip(s["label"].to_list(), s["subst_from"].to_list(),
                             s["subst_to"].to_list()):
            (pos_words if lab == 1 else neg_words)[a if lab == 1 else b] += 1
        keys = set(pos_words) | set(neg_words)
        tot = sum(pos_words.values()) + sum(neg_words.values())
        l1 = sum(abs(pos_words[k] - neg_words[k]) for k in keys) / max(tot, 1)
        bal[fam[0]] = round(float(l1), 4)
    out["substituted_string_balance_l1"] = {
        "per_family": bal, "bar": "report-only (0 = perfect balance)"}

    out["all_bars_pass"] = all(out[k]["pass"] for k in
                               ("claim_only_tfidf_auroc", "within_pair_claim_only_accuracy",
                                "paraphrase_depth", "surface_parity"))
    return out


def main():
    rng = random.Random(SEED)
    passages = pl.read_parquet(PASSAGES)
    passages = passages.filter(~pl.col("text").str.contains("Liberated Manuals"))
    print(f"passages: {passages.height}", flush=True)

    nlp = spacy.load("en_core_web_lg", disable=["ner"])

    cands, stats = candidates(passages, nlp, rng)
    print({k: len(v) for k, v in cands.items()}, stats, flush=True)

    units, dropped = select(cands, rng)
    # a positive that happens to be a verbatim substring of its passage would be
    # decidable by string match - the H148 triviality - so the pair is refused
    before = len(units)
    units = [u for u in units if u["claim_pos"].rstrip(".") not in u["chunk"]
             and u["claim_neg"].rstrip(".") not in u["chunk"]]
    dropped["verbatim_positive"] = before - len(units)
    df = emit(units)
    print(f"probe: {df.height} rows / {df['pair_id'].n_unique()} pairs", flush=True)

    ver = verify(df, rng)
    df.write_parquet(OUT)

    audit = (df.filter(pl.col("label") == 1).sample(n=min(AUDIT_N, df.height // 2), seed=SEED)
             .select(["pair_id", "neg_family", "corpus", "source_sent"])
             .join(df.filter(pl.col("label") == 1).select(["pair_id", "claim"])
                     .rename({"claim": "positive"}), on="pair_id")
             .join(df.filter(pl.col("label") == 0).select(["pair_id", "claim"])
                     .rename({"claim": "negative"}), on="pair_id"))
    audit.write_parquet(AUDIT)

    man = {
        "seed": SEED,
        "pairs": int(df["pair_id"].n_unique()), "rows": int(df.height),
        "documents": int(df["doc_id"].n_unique()),
        "passages": int(df["passage_id"].n_unique()),
        "families": {k: int(v) for k, v in df.group_by("neg_family").len().iter_rows()},
        "corpora": {k: int(v) for k, v in df.group_by("corpus").len().iter_rows()},
        "extraction": stats,
        "candidates": {k: len(v) for k, v in cands.items()},
        "dropped": dropped,
        "construction": {
            "positives": "voice recast + tense-preserving verb synonym over a "
                         "dependency-extracted (subject, relation, object) proposition",
            "role_swap": "arguments exchanged; token-multiset identical twin",
            "direction_flip": "direction verb replaced by its antonym from the same "
                              "closed pair; up/down source directions balanced 50/50",
            "entity_swap": "object replaced by the object of a different relation in "
                           "the same passage, emitted in mirrored couples",
            "caps": {"pairs_per_passage": PAIRS_PER_PASSAGE,
                     "pairs_per_document": PAIRS_PER_DOC,
                     "target_per_family": TARGET_PER_FAMILY},
            "sentence_veto": "negation / modality / hedging / conditionals refused"},
        "verify": ver,
    }
    MANIFEST.write_text(json.dumps(man, indent=2))
    print(json.dumps({k: man[k] for k in ("pairs", "rows", "documents", "families",
                                          "corpora", "dropped")}, indent=2), flush=True)
    print(json.dumps(ver, indent=2), flush=True)


if __name__ == "__main__":
    main()
