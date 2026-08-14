"""Pipeline - the six stages, composed.

    fetch -> contaminate -> format -> shape -> assemble -> census

Each stage consumes the previous stage's output, each is independently runnable,
and each is idempotent: a fetched corpus is not re-downloaded, a formatted lane
is rewritten from the same input to the same bytes, a census re-derives rather
than remembers. The first four are per corpus; the last two are per mix.

The stage that earns the sequence is `shape`. A corpus that passed licence and
contamination is legal and unseen - it is not yet admissible, because neither
says what its evidence costs to train on. Running shape here, on CPU, before a
mix is ever assembled, is the difference between a verdict and a trainer abort.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from groundrails.dataset import contaminate as contaminate_mod
from groundrails.dataset import format as format_mod
from groundrails.dataset import manifest as manifest_mod
from groundrails.dataset.assemble import AssembledMix, MixSpec, assemble
from groundrails.dataset.census import CensusReport, census
from groundrails.dataset.fetch import FetchResult, fetch
from groundrails.dataset.format import LaneResult
from groundrails.dataset.shape import ShapeReport, shape_lane

DEFAULT_DATA_DIR = Path("data/external/datasets")
DEFAULT_LANE_DIR = Path("data/interim/lanes")


@dataclass
class Pipeline:
    """The stages bound to one pair of directories.

    ``data_dir`` holds fetched archives and trees; ``lane_dir`` holds formatted
    lanes and their manifests. ``walled`` maps a walled corpus name to its
    document texts and is what the contamination stage measures against - a
    pipeline built without it can fetch and format but not gate.
    """

    data_dir: Path = DEFAULT_DATA_DIR
    lane_dir: Path = DEFAULT_LANE_DIR
    walled: Mapping[str, Sequence[str]] | None = None
    strict: bool = True

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.lane_dir = Path(self.lane_dir)

    # --- per corpus ------------------------------------------------------- #
    def fetch(self, name: str, *, dry_run: bool = False, force: bool = False) -> FetchResult:
        """Stage 1 - acquire the corpus into ``data_dir``."""
        return fetch(name, self.data_dir, dry_run=dry_run, force=force)

    def contaminate(self, name: str, **kwargs) -> dict:
        """Stage 2 - the n-gram gate over the corpus's deduplicated evidence."""
        if not self.walled:
            raise ValueError("no walled corpora given - Pipeline(walled=...) is required to gate")
        texts = format_mod.evidence_texts(name, self.data_dir)
        return contaminate_mod.check(texts, self.walled, label=name, **kwargs)

    def format(self, name: str, *, write: bool = True) -> LaneResult:
        """Stage 3 - normalise into the pair schema, and write the lane."""
        entry = manifest_mod.get(name)
        result = format_mod.format_corpus(entry, self.data_dir, strict=self.strict)
        if write:
            format_mod.write_lane(result, entry, self.lane_dir)
        return result

    def shape(self, name: str, projection=None) -> ShapeReport:
        """Stage 4 - the evidence-shape gate, at the corpus's declared presentation."""
        entry = manifest_mod.get(name)
        return shape_lane(
            self.lane_dir / f"{name}_lane.parquet",
            entry.presentation,
            name=name,
            projection=projection,
        )

    def run(self, name: str, *, projection=None, dry_run: bool = False) -> dict:
        """Stages 1-4 for one corpus, stopping at the first stage that refuses it."""
        out: dict = {"corpus": name}
        fetched = self.fetch(name, dry_run=dry_run)
        out["fetch"] = {"status": fetched.status, "detail": fetched.detail}
        if dry_run or not fetched.ok:
            return out
        if self.walled:
            gate = self.contaminate(name)
            out["contaminate"] = {"status": gate["status"], "gate": gate}
            if gate["status"] != "GREEN":
                return out
        lane = self.format(name)
        out["format"] = {"rows": lane.rows, "integrity": lane.integrity, "stats": lane.stats}
        report = self.shape(name, projection=projection)
        out["shape"] = report.to_dict()
        out["verdict"] = report.verdict
        return out

    # --- per mix ---------------------------------------------------------- #
    def assemble(self, spec: MixSpec) -> AssembledMix:
        """Stage 5 - build the mix its spec declares, or refuse."""
        return assemble(spec)

    def census(self, mix: AssembledMix) -> CensusReport:
        """Stage 6 - the pre-spend report a trainer wrapper reads before a GPU."""
        return census(mix)
