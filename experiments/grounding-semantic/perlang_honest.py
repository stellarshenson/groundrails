"""Honest (leak-free) calibration: does a per-language or EN/non-EN operating point lift the
gold v3 aggregate macro-F1 when the thresholds are chosen out-of-fold?

R1-H3 lifted macro to 0.831 with per-language cuts, but those cuts were fit in-sample. This
settles it: OOF model probs (GroupKFold leave-one-source-out) AND nested leave-one-fold-out
threshold selection, so no row's verdict ever saw its own label - neither in the head nor in
the cut. Three schemes: a single global cut, an EN / non-EN pair (2 robust thresholds), and
per-language cuts. If even the honest version clears the ~0.014 noise band over the 0.809
frozen v1 head, that is a genuine fix.

Run:  python experiments/grounding-semantic/perlang_honest.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from joint_xlingual import JF, best_threshold, load, oof_grouped, v1_head_proba  # noqa: E402

MIN_LANG = 40


def honest_yhat(df, p, ev, scheme, langs, k=5):
    """Verdict array using thresholds chosen leave-one-fold-out (never on the scored row)."""
    y = df["label"].to_numpy(int)
    lang = df["lang_norm"].to_numpy()
    groups = df["group_id"].to_numpy()
    ev_pos = np.where(ev.to_numpy())[0]
    yhat = np.full(len(df), 0, int)
    for tr, te in GroupKFold(k).split(ev_pos, y[ev_pos], groups[ev_pos]):
        R, E = ev_pos[tr], ev_pos[te]
        _, Tg = best_threshold(y[R], p[R])  # fold-global fallback
        if scheme == "global":
            yhat[E] = (p[E] >= Tg).astype(int)
        elif scheme == "en_nonen":
            for is_en in (True, False):
                rs = R[(lang[R] == "en") == is_en]
                es = E[(lang[E] == "en") == is_en]
                T = best_threshold(y[rs], p[rs])[1] if len(rs) >= 30 and len(set(y[rs])) > 1 else Tg
                yhat[es] = (p[es] >= T).astype(int)
        elif scheme == "perlang":
            handled = np.zeros(len(E), bool)
            for lg in langs:
                rs = R[lang[R] == lg]
                em = lang[E] == lg
                es = E[em]
                if len(rs) >= 30 and len(set(y[rs])) > 1 and len(es) > 0:
                    T = best_threshold(y[rs], p[rs])[1]
                    yhat[es] = (p[es] >= T).astype(int)
                    handled |= em
            rem = E[~handled]
            yhat[rem] = (p[rem] >= Tg).astype(int)
    return yhat


def report(name, df, p, ev, en, nonen, langs):
    y = df["label"].to_numpy(int)
    m, em, nem = ev.to_numpy(), (en & ev).to_numpy(), nonen.to_numpy()
    print(f"\n{name}")
    print(f"  {'scheme':12} {'macro':>6} {'EN':>6} {'nonEN':>7}")
    for scheme in ("global", "en_nonen", "perlang"):
        yh = honest_yhat(df, p, ev, scheme, langs)
        ov = f1_score(y[m], yh[m], average="macro")
        e = f1_score(y[em], yh[em], average="macro")
        n = f1_score(y[nem], yh[nem], average="macro")
        print(f"  {scheme:12} {ov:6.3f} {e:6.3f} {n:7.3f}")


def main() -> None:
    df = load()
    ev = df["role"].eq("eval")
    en = df["lang_norm"].eq("en")
    nonen = ev & ~en
    langs = [c for c, n in df.loc[ev, "lang_norm"].value_counts().items() if n >= MIN_LANG]
    print(f"languages calibrated (n >= {MIN_LANG}): {langs}")

    report("frozen v1 head", df, v1_head_proba(df), ev, en, nonen, langs)
    report("retrained eval-only (OOF)", df, oof_grouped(df, JF, ("eval",)), ev, en, nonen, langs)


if __name__ == "__main__":
    main()
