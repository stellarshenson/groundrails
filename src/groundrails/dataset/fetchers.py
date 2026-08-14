"""Bespoke fetchers - the sources a declarative manifest entry cannot express.

Three corpora need code rather than a url: TabFact ships only a retired loading
script, so its statement/table pairs are rebuilt from the GitHub repo that
script downloaded from; PubHealth's Hub repo is likewise script-only and the
data lives in the authors' Drive zip; FActScore's evidence is not shipped at all
and has to be pulled per topic from the Wikipedia API at a pinned revision.

Each is registered under the name its manifest entry gives in
``source.fetcher``. Adding a corpus with an ordinary source needs none of this.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import urllib.error
import urllib.request
import zipfile

from groundrails.dataset._deps import polars, require
from groundrails.dataset.fetch import register_fetcher
from groundrails.dataset.manifest import CorpusEntry

UA = {"User-Agent": "groundrails-dataset/1.0"}


@register_fetcher("tabfact")
def fetch_tabfact(entry: CorpusEntry, staging: Path) -> dict | None:
    """Rebuild the statement/table pairs from the upstream GitHub repo.

    Exactly what the retired loading script did: r1+r2 collected statements
    joined to their ``#``-delimited CSV tables, split by the official
    train/val/test id lists.
    """
    import io

    pl = polars()
    req = urllib.request.Request(entry.source.url, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    root = z.namelist()[0].split("/")[0]

    def _read(path: str) -> bytes:
        return z.read(f"{root}/{path}")

    tables = {
        n.split("/")[-1]: z.read(n).decode("utf-8")
        for n in z.namelist()
        if "/data/all_csv/" in n and n.endswith(".csv")
    }
    statements: dict[str, list] = {}
    for part in ("collected_data/r1_training_all.json", "collected_data/r2_training_all.json"):
        for tid, (stmts, labels, caption) in json.loads(_read(part)).items():
            statements.setdefault(tid, []).extend(
                (s, int(lb), caption) for s, lb in zip(stmts, labels, strict=True)
            )

    counts = {}
    for split, idfile in (
        ("train", "data/train_id.json"),
        ("validation", "data/val_id.json"),
        ("test", "data/test_id.json"),
    ):
        ids = json.loads(_read(idfile))
        rows = [
            {
                "table_id": tid,
                "table_caption": cap,
                "table_text": tables[tid],
                "statement": s,
                "label": lb,
            }
            for tid in ids
            if tid in statements and tid in tables
            for (s, lb, cap) in statements[tid]
        ]
        f = staging / f"tabfact__{split}.parquet"
        pl.DataFrame(rows).write_parquet(f)
        counts[split] = len(rows)
        print(f"    tabfact/{split}: {len(rows)} rows -> {f.name}", flush=True)
    return counts or None


@register_fetcher("pubhealth")
def fetch_pubhealth(entry: CorpusEntry, staging: Path) -> dict | None:
    """The authors' Drive zip holding PUBHEALTH/{train,dev,test}.tsv.

    The Hub repo ships only a loading script (datasets 5.x dropped script
    support); the url in the manifest is that script's own ``_DATA_URL``.
    """
    import csv
    import io

    gdown = require("gdown", "the Google Drive download PubHealth and FActScore need")
    pl = polars()
    out = gdown.download(entry.source.url, str(staging / "pubhealth.zip"), quiet=True)
    if not out:
        return None
    z = zipfile.ZipFile(out)
    counts = {}
    for split in ("train", "dev", "test"):
        text = z.read(f"PUBHEALTH/{split}.tsv").decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
        f = staging / f"pubhealth__{split}.parquet"
        pl.DataFrame(rows).write_parquet(f)
        counts[split] = len(rows)
        print(f"    pubhealth/{split}: {len(rows)} rows -> {f.name}", flush=True)
    (staging / "pubhealth.zip").unlink()
    return counts or None


def _drive_folder_file_ids(folder_id: str) -> dict[str, str]:
    """``file name -> id`` for a public Drive folder, via embeddedfolderview."""
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        html = r.read().decode("utf-8", "ignore")
    entries = re.findall(
        r'flip-entry" id="entry-([\w-]+)".*?flip-entry-title">([^<]+)<', html, re.DOTALL
    )
    return {name: fid for fid, name in entries}


def _get_json_retry(url: str, attempts: int = 4):
    """429/5xx-tolerant GET honouring the server's ``Retry-After``.

    A burst cadence puts a client in the Wikimedia penalty box (observed: 429
    with ``Retry-After: 20``), so back off as instructed rather than hammering.
    """
    import time

    for k in range(attempts):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503) or k == attempts - 1:
                raise
            wait = exc.headers.get("Retry-After")
            time.sleep(min(float(wait), 60.0) if wait else 5.0 * (k + 1))
    return None


@register_fetcher("factscore")
def fetch_factscore(entry: CorpusEntry, tree: Path) -> dict | None:
    """The authors' Drive ``data.zip`` plus per-topic Wikipedia evidence.

    Evidence is pinned to the last revision on or before 2023-04-01 - the
    corpus's own enwiki-20230401 knowledge source - so the article a fact was
    judged against is the article it is scored against. The authors' 28 GB
    database dump is not pulled; HTML flattening to text is ours.
    """
    import html as htmlmod
    import time
    import urllib.parse

    gdown = require("gdown", "the Google Drive download FActScore needs")
    folder_id = entry.source.url.rstrip("/").split("/")[-1]
    ids = _drive_folder_file_ids(folder_id)
    out = gdown.download(id=ids["data.zip"], output=str(tree / "data.zip"), quiet=True)
    if not out:
        return None
    z = zipfile.ZipFile(out)
    counts: dict = {}
    for member in z.namelist():
        if member.startswith("data/labeled/") and not member.endswith("/"):
            target = tree / member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(member))
    (tree / "data.zip").unlink()

    # the licence lives in the code repo, not the data zip - capture it into the
    # tree so the re-verification evidence travels with the data
    req = urllib.request.Request(
        "https://raw.githubusercontent.com/shmsw25/FActScore/main/LICENSE", headers=UA
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        (tree / "LICENSE").write_bytes(r.read())

    for model in ("InstructGPT", "ChatGPT", "PerplexityAI"):
        path = tree / "data" / "labeled" / f"{model}.jsonl"
        counts[model] = sum(1 for _ in path.read_text(encoding="utf-8").splitlines() if _.strip())
        print(f"    factscore/{model}: {counts[model]} biographies", flush=True)

    topics = (
        (tree / "data" / "labeled" / "prompt_entities.txt")
        .read_text(encoding="utf-8")
        .strip()
        .split("\n")
    )
    counts["entities"] = len(topics)

    api = "https://en.wikipedia.org/w/api.php"
    revid: dict[str, dict] = {}
    failed: list[str] = []

    def _lookup(title: str) -> None:
        """The oldid of the last revision <= 2023-04-01 for one title.

        Per-title only: the MediaWiki revisions endpoint rejects multi-title
        queries combined with rvstart/rvlimit, so there is no batch route.
        """
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "format": "json",
                "prop": "revisions",
                "titles": title,
                "rvstart": "2023-04-01T00:00:00Z",
                "rvdir": "older",
                "rvlimit": 1,
                "rvprop": "ids|timestamp",
                "redirects": 1,
            }
        )
        r = _get_json_retry(f"{api}?{params}")
        if not r or "query" not in r:
            raise RuntimeError(f"API error response: {str(r)[:160]}")
        for page in r["query"]["pages"].values():
            if "revisions" in page:
                revid[title] = {
                    "oldid": page["revisions"][0]["revid"],
                    "timestamp": page["revisions"][0]["timestamp"],
                    "title": page["title"],
                }

    for i, t in enumerate(topics):
        try:
            _lookup(t)
        except Exception as exc:  # noqa: BLE001 - recorded, not repaired
            failed.append(f"{t} (lookup: {type(exc).__name__}: {str(exc)[:80]})")
        if i % 25 == 0:
            print(f"    factscore revids: {i}/{len(topics)}", flush=True)
        time.sleep(1.0)

    evdir = tree / "evidence"
    evdir.mkdir(exist_ok=True)
    index: dict[str, dict] = {}
    for i, t in enumerate(topics):
        info = revid.get(t)
        if not info:
            failed.append(t)
            continue
        url = (
            "https://en.wikipedia.org/api/rest_v1/page/html/"
            f"{urllib.parse.quote(info['title'])}/{info['oldid']}"
        )
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                raw = r.read().decode("utf-8", "ignore")
            raw = re.sub(r"(?is)<(style|script)[^>]*>.*?</\1>", " ", raw)
            text = re.sub(r"\s+", " ", htmlmod.unescape(re.sub(r"<[^>]+>", " ", raw))).strip()
            fn = re.sub(r"[^A-Za-z0-9_.-]+", "_", t) + ".txt"
            (evdir / fn).write_text(text, encoding="utf-8")
            index[t] = {**info, "file": fn, "chars": len(text)}
        except Exception as exc:  # noqa: BLE001 - recorded, not repaired
            failed.append(f"{t} (evidence: {type(exc).__name__}: {str(exc)[:80]})")
        if i % 25 == 0:
            print(f"    factscore evidence: {i}/{len(topics)} topics", flush=True)
        time.sleep(1.0)

    (evdir / "_index.json").write_text(json.dumps(index, indent=2))
    counts["_evidence_topics"] = len(index)
    counts["_evidence_failed"] = failed
    print(
        f"    factscore evidence: {len(index)} topics pinned, {len(failed)} failed",
        flush=True,
    )
    return counts
