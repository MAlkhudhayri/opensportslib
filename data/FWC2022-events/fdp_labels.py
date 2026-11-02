"""FIFA FDP-2022 event-feed labels, mapped onto the FWC2022 tracking clock.

Same role as pff_labels.py for a second, independent label source. FDP uses
our match ids; fdp_crosswalk.json holds one UTC offset per game and period
(fitted on head-body events against the rule's headers, 94% matched within
500 ms). FDP marks the contact instant itself: against the final's human
headers the median gap is +8 ms (sd 74 ms), so no shift is applied.

Classes are FDP events (CLASSES below). HEADER is any event played with the
head (body_type == "head"): 4,227 over 64 games against PFF's 5,723, because
FDP tags the body part only on passes, clearances, attempts and throw-ins.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/ibex/user/alkhudmk/fwc2022")
EVENTS = Path("/ibex/user/alkhudmk/fdp2022/events_all.pkl")
HERE = Path(__file__).resolve().parent
CROSSWALK = HERE / "fdp_crosswalk.json"
SHIFT_MS = 0

# label -> row filter on the FDP event table
CLASSES = {
    "HEADER":       lambda d: d.body_type == "head",
    "PASS":         lambda d: d.event == "pass",
    "CROSS":        lambda d: d.event == "cross",
    "THROW IN":     lambda d: d.event == "throwin",
    "FREE KICK":    lambda d: d.event == "freekick",
    "CORNER":       lambda d: d.event == "corner",
    "GOAL KICK":    lambda d: d.event == "goalkick",
    "SHOT":         lambda d: d.event == "attempt_at_goal",
    "CLEARANCE":    lambda d: d.event == "clearance",
    "TACKLE":       lambda d: d.event == "tackle",
    "INTERCEPTION": lambda d: d.event == "interception",
    "AERIAL DUEL":  lambda d: d.event == "aerial_duel",
    "BLOCK":        lambda d: d.event == "block",
    "FOUL":         lambda d: d.event == "foul_against",
    "GOAL":         lambda d: d.event == "goal",
}


def annot_dir(label):
    return ROOT / f"fdp_annotations_{label.lower().replace(' ', '_')}"


def utc_ms(s):
    return int(round(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000))


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).replace(tzinfo=None)\
        .strftime("%Y-%m-%d %H:%M:%S.%f")


_EV = None


def events():
    global _EV
    if _EV is None:
        _EV = pd.read_pickle(EVENTS)
    return _EV


def labels(label="HEADER", shift_ms=None):
    """{fwc_game_id: sorted UTC ms}, and {fwc_game_id: split} (the SNGAR split, shared with PFF)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("pff_labels", HERE / "pff_labels.py")
    pl = importlib.util.module_from_spec(spec); spec.loader.exec_module(pl)
    _, split = pl.labels("HEADER")
    shift = SHIFT_MS if shift_ms is None else shift_ms
    cw = json.load(open(CROSSWALK))
    d = events()
    d = d[CLASSES[label](d)]
    out = {g: np.array([], np.int64) for g in split}
    for gid, grp in d.groupby("match_id"):
        gid = str(int(gid))
        ts = [int(r.match_run_time_in_ms) + cw[gid][str(int(r.half_time))]["offset_ms"] + shift
              for r in grp.itertuples() if str(int(r.half_time)) in cw[gid]]
        out[gid] = np.sort(np.array(ts, np.int64))
    return out, split


def write_annotations(label="HEADER"):
    """One OSL v2 annotation file per game, readable by the baseline GroundTruth."""
    import h5py
    lab, _ = labels(label)
    out_dir = annot_dir(label)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for gid, ts in sorted(lab.items()):
        with h5py.File(ROOT / gid / "live_ball.h5", "r") as f:
            stamps, half = f["timestamp_utc"][:], f["half"][:]
        s = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in stamps])
        order = np.argsort(s, kind="stable"); s, half = s[order], half[order]
        kicks = [utc_ms(s[half == p][0]) for p in sorted(set(int(h) for h in np.unique(half) if int(h) > 0)) if (half == p).any()]
        if not kicks:
            continue
        v0 = kicks[0]
        events_ = [{"label": "Header", "position_ms": int(t - v0), "timestamp_utc": iso(int(t)),
                    "team": "", "visibility": "visible", "metadata": {"note": ""}} for t in ts]
        events_ += [{"label": "Kickoff", "position_ms": int(k - v0), "timestamp_utc": iso(int(k)),
                     "team": "", "visibility": "visible", "metadata": {"note": ""}} for k in kicks]
        events_.sort(key=lambda e: e["position_ms"])
        payload = {"version": "2.0", "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                   "task": "localization", "dataset_name": f"FWC2022-FDP-{label}",
                   "data": [{"id": gid,
                             "inputs": [{"type": "video", "path": f"{gid}.mp4", "UTC_time_start": iso(v0)},
                                        {"type": "player_joints_h5", "path": f"{gid}/live_joints.h5",
                                         "ball_path": f"{gid}/live_ball.h5"}],
                             "events": events_}]}
        (out_dir / f"{gid}.json").write_text(json.dumps(payload, indent=1))
        written.append((gid, len(ts), len(kicks)))
    return written


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or list(CLASSES)
    for label in which:
        lab, sp = labels(label)
        n = {k: sum(len(lab[g]) for g in lab if sp[g] == k) for k in ("train", "valid", "test")}
        w = write_annotations(label)
        print(f"{label:13s} train {n['train']:6d} valid {n['valid']:5d} test {n['test']:5d} | wrote {len(w)} files -> {annot_dir(label).name}", flush=True)
