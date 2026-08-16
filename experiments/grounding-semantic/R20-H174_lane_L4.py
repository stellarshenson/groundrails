"""R20-H174 LANE L4 `path_bind` - bind_path_segment rider, build + verify, CPU.

Registered in docs/experiments/semantic-grounding-experiments.md, block "R20-H174
HAGRID/EMANUAL PORTFOLIO ARM": "L4 bind_path_segment rider (~5-15k rows, pure
generator)".  Rider only - the registration gives it no standalone bar.

THE DEFECT THE LANE TEACHES AGAINST
-----------------------------------
31 of 748 emanual sentences render an arrowed UI path; their mean model logit is
0.1703 against 1.5188 for the rest, and the 19 items carrying one read
within-stratum AUROC 0.5833 against 0.7107.  emanual item 15 transposes two path
levels while preserving the token multiset exactly and is ranked at the 73rd
model percentile against the 28th lexical percentile - the model reads which
segments are present, not which level each one sits at
(R19-H162_procedural_mechanisms.json, mechanism `bind_path_segment`).

CONSTRUCTION - a generator, no corpus
-------------------------------------
Each item is a synthetic manual page: a heading, an intro, and 3-5 numbered
procedures.  Every procedure states its navigation path as a BARE TOKEN RUN
inside a step line ("From the menu, go to Settings Display Picture Mode"), which
is the emanual rendering; the claim states the same path ARROWED
("Settings -> Display -> Picture Mode"), which is the response rendering.  The
pair is then:

  label 1   the claim restates the true path and the true value
  label 0   `path_transpose`      - two ADJACENT segments swapped; the token
                                    multiset is preserved EXACTLY, so no
                                    bag-of-tokens or containment feature can
                                    separate the legs
            `path_wrong_segment`  - one segment replaced by a sibling segment
                                    that the SAME page attests on a different
                                    path, so segment presence stays uninformative

NO GLOBAL HIERARCHY PRIOR
-------------------------
Level assignment is random from one shared segment pool, so the same name sits
at level 1 in one item and level 3 in another.  Without this a claim-only probe
learns the menu tree and settles every pair without reading the evidence; with
it the level of a segment is knowable only from the page.  The build enforces
the consequence rather than assuming it: no negative path may appear anywhere in
its own evidence, and every positive path must.

Contamination: CLEAR by construction - a generator has no source population.
The census runs anyway (R20-H174_lane_L4_census.json).

Run:  uv run python experiments/grounding-semantic/R20-H174_lane_L4.py
"""

import collections
import importlib.util as _ilu
import json
from pathlib import Path
import random
import sys

import polars as pl

_spec = _ilu.spec_from_file_location("h174common", Path(__file__).parent / "R20-H174_lane_common.py")
C = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(C)

HERE = Path(__file__).parent
OUT = HERE / "R20-H174_lane_L4.parquet"
MANIFEST = HERE / "R20-H174_lane_L4_manifest.json"

SEED = 4174
TAG = "path_bind"
N_PAIRS = 5_000                  # 10,000 rows, inside the registered 5-15k band
TRANSPOSE_SHARE = 0.60
DEPTHS = (3, 4, 5)
PROCEDURES = (3, 4, 5)           # procedure blocks per generated page
ARROW = " → "               # the arrowed render used in the claim

# --------------------------------------------------------------------------- #
# lexicons - written here, nothing crawled
# --------------------------------------------------------------------------- #
SEGMENTS = [
    "Settings", "General", "System", "Advanced", "Display", "Picture",
    "Picture Mode", "Picture Size", "Expert Settings", "Brightness", "Contrast",
    "Sharpness", "Colour", "Colour Tone", "Backlight", "Screen Timeout",
    "Auto Rotate", "Wallpaper", "Themes", "Fonts", "Text Size", "Zoom",
    "Colour Filters", "Sound", "Sound Mode", "Sound Output", "Volume",
    "Equalizer", "Balance", "Digital Output", "Audio Format", "Network",
    "Wi-Fi", "Hotspot", "VPN", "Proxy", "Airplane Mode", "Bluetooth",
    "Pairing", "Devices", "External Device Manager", "Remote and Accessories",
    "Input Signal", "Broadcasting", "Channel", "Auto Tuning", "Captions",
    "Caption Settings", "Accessibility", "Voice Assistant", "Gestures",
    "Shortcuts", "Privacy", "Permissions", "Location", "Camera Access",
    "Microphone Access", "Security", "Screen Lock", "PIN", "Fingerprint",
    "Face Unlock", "Storage", "Cache", "Downloads", "Backup", "Reset",
    "Software Update", "About", "Language", "Region", "Date and Time",
    "Time Zone", "Battery", "Power Saving", "Eco Solution", "Sleep Timer",
    "Screen Burn Protection", "Game Mode", "Multi Window", "Notifications",
    "Do Not Disturb", "Accounts", "Sync", "Data Usage", "Roaming", "Printers",
    "Scanners", "Paper Tray", "Print Quality", "Duplex", "Firmware",
    "Diagnostics", "Maintenance", "Calibration", "Sensors", "Motion Detection",
    "Recording", "Playback", "Timers", "Scheduling", "Profiles", "Presets",
    "Keyboard", "Input Method", "Autocorrect", "Clipboard", "Developer Options",
    "USB Configuration", "Ports", "Firewall", "Certificates",
]

DEVICES = [
    "the television", "the handset", "the router", "the printer", "the camera",
    "the receiver", "the monitor", "the tablet", "the set-top box",
    "the projector", "the speaker", "the thermostat", "the access point",
]

OPENERS = [
    "Press the Home button on the remote control.",
    "Unlock the screen and open the main menu.",
    "From the standby screen, press the Menu key.",
    "Open the launcher and scroll to the bottom of the list.",
    "Press and hold the function key until the menu appears.",
    "Tap the menu icon in the upper right corner.",
]

CLOSERS = [
    "Confirm the change and leave the menu.",
    "The change takes effect immediately.",
    "Press Back twice to return to the main screen.",
    "Wait for the confirmation message before continuing.",
    "The setting is stored until it is changed again.",
]

VALUES = [
    "Standard", "Dynamic", "Movie", "Natural", "On", "Off", "Automatic",
    "Manual", "High", "Medium", "Low", "45", "70", "12", "English",
    "30 seconds", "5 minutes", "Enabled", "Disabled", "Optimised",
]

VERBS = ["set", "change", "adjust", "configure", "switch"]

INTROS = [
    ("This section explains how to adjust {dev}. Read the safety information "
     "before you begin."),
    ("The following procedures apply to {dev}. Menu names may differ slightly "
     "between software versions."),
    ("Use the procedures below to configure {dev}. Some options are available "
     "only when the device is idle."),
    "This chapter covers the menu options available on {dev}.",
]

HEADINGS = [
    "Configuration and Setup", "Menu Reference", "Adjusting the Settings",
    "Operation", "Customising the Device", "Basic Configuration",
    "Advanced Options", "User Preferences",
]

CLAIM_TEMPLATES = [
    "Open {path} and {verb} it to {value}.",
    "To {verb} the {leaf}, go to {path} and select {value}.",
    "Navigate to {path}, then choose {value}.",
    "{path} is where you {verb} the {leaf} to {value}.",
    "You can {verb} the {leaf} to {value} from {path}.",
    "Select {value} under {path}.",
]


def render_path_bare(path):
    return " ".join(path)


def render_path_arrow(path):
    return ARROW.join(path)


def build_page(rng):
    """One synthetic manual page plus the paths it attests."""
    n_proc = rng.choice(PROCEDURES)
    pool = rng.sample(SEGMENTS, k=min(len(SEGMENTS), 4 * n_proc + 4))
    paths, blocks = [], []
    for i in range(n_proc):
        depth = rng.choice(DEPTHS)
        # levels drawn from ONE shared pool - no fixed hierarchy anywhere
        path = rng.sample(pool, k=depth)
        value = rng.choice(VALUES)
        leaf = path[-1]
        blocks.append(
            f"To {rng.choice(VERBS)} the {leaf}:\n"
            f"1. {rng.choice(OPENERS)}\n"
            f"2. From the menu, go to {render_path_bare(path)}.\n"
            f"3. Select {value} and confirm.\n"
            f"4. {rng.choice(CLOSERS)}"
        )
        paths.append({"path": path, "value": value, "leaf": leaf, "index": i})
    dev = rng.choice(DEVICES)
    text = (f"{rng.choice(HEADINGS)}\n\n{rng.choice(INTROS).format(dev=dev)}\n\n"
            + "\n\n".join(blocks))
    return text, paths, pool


def transpose(path, rng):
    """Swap two ADJACENT segments - the token multiset is preserved exactly."""
    i = rng.randrange(len(path) - 1)
    out = list(path)
    out[i], out[i + 1] = out[i + 1], out[i]
    return out, i


def wrong_segment(path, page_paths, rng):
    """Replace one segment with a sibling the SAME page attests elsewhere."""
    attested = {s for p in page_paths for s in p["path"]} - set(path)
    if not attested:
        return None, None
    i = rng.randrange(len(path))
    out = list(path)
    out[i] = rng.choice(sorted(attested))
    return out, i


def attested(path, text):
    return render_path_bare(path) in text


def already_built():
    """Idempotence: a lane whose parquet and manifest are on disk and whose own
    verify block passed is not rebuilt.  `--force` overrides."""
    if "--force" in sys.argv or not (OUT.exists() and MANIFEST.exists()):
        return False
    try:
        man = json.loads(MANIFEST.read_text())
        rows = pl.read_parquet(OUT).height
    except Exception:
        return False
    if man.get("verify", {}).get("all_bars_pass") and rows == man.get("rows"):
        print(f"{OUT.name}: {rows} rows already built and passing - skipping "
              f"(pass --force to rebuild)", flush=True)
        return True
    return False


def main():
    if already_built():
        return
    rng = random.Random(SEED)
    print(f"=== R20-H174 lane L4 ({TAG}) seed {SEED}", flush=True)
    rows, pid = [], 0
    built = collections.Counter()
    want = {"path_transpose": int(round(N_PAIRS * TRANSPOSE_SHARE))}
    want["path_wrong_segment"] = N_PAIRS - want["path_transpose"]
    tries = 0

    while pid < N_PAIRS and tries < N_PAIRS * 40:
        tries += 1
        text, paths, _pool = build_page(rng)
        spec = rng.choice(paths)
        path, value, leaf = spec["path"], spec["value"], spec["leaf"]
        if not attested(path, text):
            continue
        fam = ("path_transpose" if built["path_transpose"] < want["path_transpose"]
               else "path_wrong_segment")
        if built[fam] >= want[fam]:
            fam = "path_wrong_segment" if fam == "path_transpose" else "path_transpose"
        if fam == "path_transpose":
            bad, at = transpose(path, rng)
        else:
            bad, at = wrong_segment(path, paths, rng)
        if bad is None or bad == path or len(set(bad)) != len(bad) or attested(bad, text):
            continue
        ti = rng.randrange(len(CLAIM_TEMPLATES))
        tpl = CLAIM_TEMPLATES[ti]
        verb = rng.choice(VERBS)
        pos = tpl.format(path=render_path_arrow(path), verb=verb, value=value, leaf=leaf)
        neg = tpl.format(path=render_path_arrow(bad), verb=verb, value=value, leaf=leaf)
        base = {"chunk": text, "doc_id": f"gen{pid:06d}", "source": "generator",
                "tag": TAG, "neg_family": fam, "depth": len(path),
                "swap_index": at, "template_id": ti,
                "true_path": render_path_arrow(path),
                "wrong_path": render_path_arrow(bad),
                "value": value, "leaf": leaf,
                "procedures": len(paths)}
        rows.append(dict(pair_id=pid, label=1, claim=pos, **base))
        rows.append(dict(pair_id=pid, label=0, claim=neg, **base))
        built[fam] += 1
        pid += 1
        if pid % 1000 == 0:
            print(f"  {pid} pairs  {dict(built)}", flush=True)

    df = C.dedupe(pl.DataFrame(rows))
    df.write_parquet(OUT)
    print(f"{df.height} rows / {df['pair_id'].n_unique()} pairs -> {OUT.name}", flush=True)

    res = verify(df, rng)
    man = build_manifest(df, res)
    MANIFEST.write_text(json.dumps(man, indent=2))
    print(json.dumps({k: man[k] for k in
                      ("rows", "pairs", "label_balance", "families",
                       "window_census", "verify")}, indent=2), flush=True)
    ok = res["all_bars_pass"]
    print(f"=== R20-H174 LANE L4 {'BUILT' if ok else 'FAILED BARS'} ===", flush=True)
    raise SystemExit(0 if ok else 1)


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def verify(df, rng):
    out = {}
    out["pair_integrity"] = C.pair_integrity(df)

    # --- the path binding must be re-derivable from the evidence: the TRUE path
    # is attested as a bare run, the CORRUPTED one is not, on every row.
    errs = []
    for r in df.iter_rows(named=True):
        true_bare = r["true_path"].replace(ARROW, " ")
        bad_bare = r["wrong_path"].replace(ARROW, " ")
        if true_bare not in r["chunk"]:
            errs.append({"pair_id": r["pair_id"], "why": "true path not attested"})
        elif bad_bare in r["chunk"]:
            errs.append({"pair_id": r["pair_id"], "why": "corrupted path IS attested"})
    out["path_attestation_audit"] = {
        "rows": df.height, "errors": len(errs), "bar": "0 errors",
        "pass": not errs, "examples": errs[:5]}

    # --- the transposition family holds the claim's token multiset EXACTLY
    tr = df.filter(pl.col("neg_family") == "path_transpose")
    bad = 0
    for pid, sub in tr.group_by("pair_id"):
        a, b = sub.sort("label")["claim"].to_list()
        if sorted(C.tokens(a)) != sorted(C.tokens(b)):
            bad += 1
    out["transpose_multiset_identity"] = {
        "pairs": int(tr["pair_id"].n_unique()), "violations": bad,
        "bar": "0 violations - no bag-of-tokens feature may separate the legs",
        "pass": bad == 0}

    # --- every segment of a corrupted path is still attested on the page, so
    # segment PRESENCE cannot settle the pair either
    miss = 0
    for r in df.filter(pl.col("label") == 0).iter_rows(named=True):
        for seg in r["wrong_path"].split(ARROW):
            if seg not in r["chunk"]:
                miss += 1
                break
    out["negative_segment_presence"] = {
        "negatives": int(df.filter(pl.col("label") == 0).height),
        "rows_with_unattested_segment": miss, "bar": "0 rows",
        "pass": miss == 0}

    out["surface_parity"] = C.surface_parity(df)

    probe, score = C.claim_only_probe(df["claim"].to_list(), df["label"].to_list(),
                                      df["doc_id"].to_list(), rng)
    wp = C.within_pair_accuracy(df, score, by="neg_family")
    worst = max(v["acc"] for v in wp.values())
    out["claim_only_tfidf_auroc"] = {
        "value": round(probe, 4), "bar": "< 0.55", "pass": bool(probe < 0.55),
        "scoring": "5-fold item-disjoint, out of fold, liblinear tol 1e-7"}
    out["within_pair_claim_only_accuracy"] = {
        "per_family": wp, "worst": round(worst, 4), "bar": "< 0.60",
        "pass": bool(worst < 0.60)}

    out["all_bars_pass"] = all(
        out[k]["pass"] for k in
        ("pair_integrity", "path_attestation_audit", "transpose_multiset_identity",
         "negative_segment_presence", "surface_parity", "claim_only_tfidf_auroc",
         "within_pair_claim_only_accuracy"))
    return out


def build_manifest(df, res):
    y = df["label"].to_list()
    return dict(
        experiment="R20-H174 lane L4 - bind_path_segment rider (path_bind)",
        registration="docs/experiments/semantic-grounding-experiments.md, "
                     "block 'R20-H174 HAGRID/EMANUAL PORTFOLIO ARM'",
        tag=TAG,
        dann_group=TAG,
        mix_loader="drop-in for R18-H150_arm_run.make_build_mix - columns "
                   "claim / chunk / label / pair_id / neg_family; chunk is read "
                   "UNTRUNCATED and windowed 1500/750 by the loader",
        seed=SEED,
        rows=df.height,
        pairs=int(df["pair_id"].n_unique()),
        documents=int(df["doc_id"].n_unique()),
        label_balance={"label_1": int(sum(y)), "label_0": int(len(y) - sum(y)),
                       "positive_share": round(sum(y) / len(y), 4)},
        families={k: v for k, v in df.group_by("neg_family").len().iter_rows()},
        depths={str(k): v for k, v in df.group_by("depth").len().iter_rows()},
        sources={"generator": C.SOURCES["generator"]},
        generator=dict(n_pairs_target=N_PAIRS, transpose_share=TRANSPOSE_SHARE,
                       depths=list(DEPTHS), procedures_per_page=list(PROCEDURES),
                       arrow=ARROW, n_segments=len(SEGMENTS),
                       n_templates=len(CLAIM_TEMPLATES), n_values=len(VALUES),
                       level_assignment="random from one shared segment pool - "
                                        "no fixed hierarchy, so a claim-only "
                                        "probe cannot learn the tree"),
        char_stats=dict(claim=C.char_stats(df["claim"].to_list()),
                        chunk=C.char_stats(df["chunk"].to_list())),
        diversity=dict(distinct_claims=int(df["claim"].n_unique()),
                       distinct_chunks=int(df["chunk"].n_unique()),
                       distinct_true_paths=int(df["true_path"].n_unique())),
        window_census=C.window_census(df["chunk"].to_list()),
        verify=res)


if __name__ == "__main__":
    main()
