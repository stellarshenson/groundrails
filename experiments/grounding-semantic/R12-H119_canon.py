"""R12-H119 - numeric-surface canonicalizer (serving wrapper, pure python, no deps).

Registered in docs/experiments/semantic-grounding-experiments.md (round 12) and
experiments/grounding-semantic/R12_synthesis_full_field.md (R12-H119). An
idempotent, label-free, subset-blind text transform applied symmetrically to
claim and evidence immediately before tokenization; weights, read formula and
window geometry stay byte-identical.

Binding amendment 3 (skeptic 6): the accounting-parenthesis rule and whitespace
normalization are DROPPED (year/citation corruption 84.7% hotpotqa; whitespace
changes 100% of covidqa/techqa chunk tokenizations). At most three rules are
candidates here - thousands-separator, currency spacing, percent spacing - and
only rules with a non-zero measured ablation effect (R12-H119_surface_audit.py)
may ship.

Amendment 4: both directions run as separate deterministic reads. `strip`
removes thousands separators (claim/evidence agreement); `add` inserts them
(train-surface parity - RAGTruth carries 2,928 separated numbers).

Every rule is idempotent: canon(canon(x)) == canon(x), decanon(decanon(x)) ==
decanon(x). Asserted in self_test().
"""

import re

# --- rule (a) thousands separators -----------------------------------------
# Strict digit-group pattern only: 1..3 digits then one or more ",ddd" groups,
# not adjacent to another digit or comma on either side. "2020,1234" and
# "1,23" do not match; "1,234,567" does.
_THOUSANDS = re.compile(r"(?<![\d,])\d{1,3}(?:,\d{3})+(?![\d])")

# Bare integer runs for the "add" direction: not preceded by a digit, comma or
# dot (so the fractional part of 12.3456 is never separated), not followed by a
# digit or comma (so an already-separated number is left alone).
_BARE_INT = re.compile(r"(?<![\d.,])(\d{4,})(?![\d,])")

# --- rule (b) currency spacing ----------------------------------------------
_CURRENCY = re.compile(r"([$€£¥])[ \t]+(?=\d)")

# --- rule (c) percent spacing -----------------------------------------------
_PERCENT = re.compile(r"(\d)[ \t]+%")


def _strip_thousands(text):
    return _THOUSANDS.sub(lambda m: m.group(0).replace(",", ""), text)


def _add_thousands(text):
    """Insert separators into bare integers. 5+ digit runs always; 4-digit runs
    only outside [1900, 2099] (the year heuristic - a bare 2019 is a date far
    more often than a quantity)."""

    def rep(m):
        d = m.group(1)
        if len(d) == 4 and 1900 <= int(d) <= 2099:
            return d
        out = []
        for i, ch in enumerate(reversed(d)):
            if i and i % 3 == 0:
                out.append(",")
            out.append(ch)
        return "".join(reversed(out))

    return _BARE_INT.sub(rep, text)


def _currency_space(text):
    return _CURRENCY.sub(r"\1", text)


def _percent_space(text):
    return _PERCENT.sub(r"\1%", text)


# Candidate rule table. The audit ablates these one at a time; only survivors
# ship (see SHIPPED below / --rules on the read script).
RULES = {
    "thousands": _strip_thousands,
    "currency": _currency_space,
    "percent": _percent_space,
}

# Rules admitted by the CPU audit (R12-H119_audit_result.json).
#   thousands  +2.20 pts affix-inclusive, +8.29 pts bare-token (reproduces the
#              registered +8.3 reference exactly) - SHIPS
#   currency   +13.01 pts affix-inclusive, +0.00 bare - SHIPS. finqa evidence
#              writes every one of its 3,240 "$" with a following space while
#              claims write "$383,221"; the encoder tokenizes that affix, so the
#              affix-inclusive number is the honest metric and the bare-token
#              +0.00 is the sensitivity reading, not the gate
#   percent    +0.00 in both token definitions - DROPPED at the gate
SHIPPED = ("thousands", "currency")


def canon(text, rules=None):
    """Strip direction: apply the shipped rules in fixed order. Idempotent."""
    for name in RULES if rules is None else rules:
        if rules is None and name not in SHIPPED:
            continue
        text = RULES[name](text)
    return text


def decanon(text, rules=None):
    """Add direction: thousands separators inserted rather than stripped; the
    non-thousands rules are direction-free and applied identically."""
    names = SHIPPED if rules is None else rules
    for name in names:
        if name == "thousands":
            text = _add_thousands(text)
        else:
            text = RULES[name](text)
    return text


def transform(direction, rules=None):
    """Return the serving-path text function for a direction."""
    if direction == "strip":
        return lambda t: canon(t, rules)
    if direction == "add":
        return lambda t: decanon(t, rules)
    raise ValueError(f"unknown direction {direction!r}")


def self_test():
    cases = [
        "Revenue was 1,234,567 dollars in 2020, up 5 % from $ 1,000.",
        "no numbers here at all",
        "Section 4, item 12, page 1,024 of 2,048; 2020,1234 is not a group.",
        "12.3456 and 0.1234 and 123456789 and 1999 and 2100",
        "Costs: $ 1 500 and 45 % and € 2,500,000",
    ]
    for t in cases:
        for fn in (canon, decanon):
            once = fn(t)
            assert fn(once) == once, f"not idempotent: {fn.__name__}({t!r})"
        for name in RULES:
            once = canon(t, [name])
            assert canon(once, [name]) == once, f"not idempotent: rule {name} on {t!r}"
            once = decanon(t, [name])
            assert decanon(once, [name]) == once, f"not idempotent: add-rule {name} on {t!r}"
    assert canon("1,234,567", ["thousands"]) == "1234567"
    assert canon("2020,1234", ["thousands"]) == "2020,1234"
    assert canon("1,23", ["thousands"]) == "1,23"
    assert canon("$ 5", ["currency"]) == "$5"
    assert canon("5 %", ["percent"]) == "5%"
    assert decanon("1234567", ["thousands"]) == "1,234,567"
    assert decanon("2019", ["thousands"]) == "2019"
    assert decanon("1899", ["thousands"]) == "1,899"
    assert decanon("12.3456", ["thousands"]) == "12.3456"
    assert decanon("123456.78", ["thousands"]) == "123,456.78"
    print("R12-H119_canon self-test OK")


if __name__ == "__main__":
    self_test()
