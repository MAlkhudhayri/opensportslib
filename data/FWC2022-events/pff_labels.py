"""PFF event-feed header labels, mapped onto FWC2022 games.

The FWC2022 header annotations shipped with the tracking data are the rule's
own output, so training and testing on them measures imitation of the rule
(paper section 3.3). The PFF event feed labels a header independently, as a
possession event played with the head, and crosswalk2.json maps PFF game ids
and clocks onto ours. This module turns that into UTC-stamped labels and into
OSL v2 annotation files the baseline evaluator can read.

PFF stamps the possession event, which lands systematically earlier than the
contact instant the rule and the human annotator mark: median +167 ms against
the final's human labels, with a 61 ms spread. SHIFT_MS corrects that bias so
PFF labels share the human convention.
"""

import json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

GAR = Path("/ibex/user/alkhudmk/sngar-frames")
ROOT = Path("/ibex/user/alkhudmk/fwc2022")
HERE = Path(__file__).resolve().parent
CROSSWALK = HERE / "crosswalk2.json"
ANNOT_DIR = ROOT / "pff_annotations"          # HEADER, kept for compatibility
def annot_dir(label="HEADER"):
    """HEADER keeps the original directory; other classes get their own."""
    return ANNOT_DIR if label == "HEADER" else ROOT / f"pff_annotations_{label.lower().replace(' ', '_')}"

SHIFT_MS = 167


def utc_ms(s):
    return int(round(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000))


def iso(ms):
    """Naive UTC, space separated, as the shipped annotation files use."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).replace(tzinfo=None)\
        .strftime("%Y-%m-%d %H:%M:%S.%f")


def _gar_events(label="HEADER"):
    """{sngar_game: {period: [position_ms]}} plus {sngar_game: split}."""
    ev = defaultdict(lambda: defaultdict(list))
    split = {}
    for sp in ("train", "valid", "test"):
        for e in json.load(open(GAR / f"annotations_{sp}.json"))["data"]:
            m = e["metadata"]
            split[m["game_id"]] = sp
            if e["labels"]["action"]["label"] == label:
                ev[m["game_id"]][int(str(m["game_time"]).split(" - ")[0])].append(m["position_ms"])
    return ev, split


def labels(label="HEADER", shift_ms=None):
    """The +167 ms correction was measured for headers only; other classes use 0."""
    if shift_ms is None:
        shift_ms = SHIFT_MS if label == "HEADER" else 0
    """{fwc_game_id: sorted UTC ms}, and {fwc_game_id: split}."""
    cw = json.load(open(CROSSWALK))
    ev, split = _gar_events(label)
    out, splits = {}, {}
    for sg, info in cw.items():
        fg = info["fwc"]
        ts = []
        for p, off in info["offsets_ms"].items():
            ts += [t + int(off) + shift_ms for t in ev[sg].get(int(p), [])]
        out[fg] = np.sort(np.array(ts, np.int64))
        splits[fg] = split[sg]
    return out, splits


def write_annotations(label="HEADER"):
    """One OSL v2 annotation file per game, readable by the baseline GroundTruth."""
    import h5py
    lab, _ = labels(label)
    out_dir = annot_dir(label)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for gid, ts in sorted(lab.items()):
        ball = ROOT / gid / "live_ball.h5"
        with h5py.File(ball, "r") as f:
            stamps = f["timestamp_utc"][:]
            half = f["half"][:]
        s = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in stamps])
        order = np.argsort(s, kind="stable")
        s, half = s[order], half[order]
        # kickoff of each played period: first in-play sample of that period
        kicks = []
        for p in sorted(set(int(h) for h in np.unique(half) if int(h) > 0)):
            m = half == p
            if m.any():
                kicks.append(utc_ms(s[m][0]))
        if not kicks:
            continue
        v0 = kicks[0]                      # arbitrary reference; positions are derived from UTC
        events = [{"label": "Header", "position_ms": int(t - v0), "timestamp_utc": iso(int(t)),
                   "team": "", "visibility": "visible", "metadata": {"note": ""}} for t in ts]
        events += [{"label": "Kickoff", "position_ms": int(k - v0), "timestamp_utc": iso(int(k)),
                    "team": "", "visibility": "visible", "metadata": {"note": ""}} for k in kicks]
        events.sort(key=lambda e: e["position_ms"])
        payload = {"version": "2.0", "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                   "task": "localization", "dataset_name": f"FWC2022-PFF-{label}",
                   "data": [{"id": gid,
                             "inputs": [
                                 {"type": "video", "path": f"{gid}.mp4", "UTC_time_start": iso(v0)},
                                 {"type": "player_joints_h5", "path": f"{gid}/live_joints.h5",
                                  "ball_path": f"{gid}/live_ball.h5"}],
                             "events": events}]}
        p = out_dir / f"{gid}.json"
        p.write_text(json.dumps(payload, indent=1))
        written.append((gid, len(ts), len(kicks)))
    return written


if __name__ == "__main__":
    import sys
    LABEL = sys.argv[1] if len(sys.argv) > 1 else "HEADER"
    lab, sp = labels(LABEL)
    import collections
    c = collections.Counter(sp.values())
    n = {k: sum(len(lab[g]) for g in lab if sp[g] == k) for k in c}
    print("games per split:", dict(c))
    print("header labels per split:", n, " total", sum(n.values()))
    w = write_annotations(LABEL)
    print(f"wrote {len(w)} annotation files to {annot_dir(LABEL)}")
    print("sample:", w[:3])
