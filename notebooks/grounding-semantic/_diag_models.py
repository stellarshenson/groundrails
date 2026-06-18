"""Per-model smoke test on CPU - prints a marker before each so a hard crash
(segfault) reveals the culprit. Writes raw, unbuffered."""
import grounding_models as gm

rec = [{"claim": "The torque is 25 Nm.",
        "chunks": ["Tighten the bolt to 25 Nm before use.", "Unrelated cleaning text."],
        "label": 1, "lang": "en"}]


def t(m):
    print(m, flush=True)


for cfg in gm.EMBEDDERS:
    t(f"TRY  embed  {cfg['name']}")
    try:
        s = gm.embed_scores(cfg, rec, "cpu", bs=8)
        t(f"OK   embed  {cfg['name']}  score={s[0]:.3f}")
    except Exception as e:
        t(f"FAIL embed  {cfg['name']} -> {type(e).__name__}: {str(e)[:140]}")

for cfg in gm.CROSS:
    t(f"TRY  {cfg['kind']:6} {cfg['name']}")
    try:
        s = gm.cross_scores(cfg, rec, "cpu", bs=8)
        t(f"OK   {cfg['kind']:6} {cfg['name']}  score={s[0]:.3f}")
    except Exception as e:
        t(f"FAIL {cfg['kind']:6} {cfg['name']} -> {type(e).__name__}: {str(e)[:140]}")

t("DONE")
