"""Package the PFF and FDP labels as OSL v2 annotation files for release.

One file per source, in the layout of headers_annotations.json that ships with
FWC2022-skeleton-headers: 64 games, one event list per game, position_ms
relative to the first ball sample of the game's live_ball.h5, timestamp_utc
absolute, every class in one multi-class file. Also writes the crosswalks and
a README next to them.
"""

import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/ibex/user/alkhudmk/fwc2022")
OUT = ROOT / "release"
HERE = Path(__file__).resolve().parent
PFF_CLASSES = ["HEADER", "PASS", "CROSS", "THROW IN", "FREE KICK", "SHOT", "OUT",
               "PLAYER SUCCESSFUL TACKLE", "GOAL", "HIGH PASS"]


def load(n):
    s = importlib.util.spec_from_file_location(n, HERE / f"{n}.py")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def utc_ms(s):
    return int(round(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000))


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S.%f")


def ball_clock(gid):
    with h5py.File(ROOT / gid / "live_ball.h5", "r") as f:
        s = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["timestamp_utc"][:]]); half = f["half"][:]
    order = np.argsort(s, kind="stable"); s, half = s[order], half[order]
    kicks = {int(p): utc_ms(s[half == p][0]) for p in np.unique(half) if int(p) > 0}
    return utc_ms(s[0]), kicks


def package(source, classes, created_by, note):
    mod = load(f"{source}_labels")
    per = {c: mod.labels(c)[0] for c in classes}
    _, split = mod.labels(classes[0])
    games = []
    counts = {c: 0 for c in classes}
    for gid in sorted(split):
        v0, kicks = ball_clock(gid)
        events = []
        for c in classes:
            for t in per[c].get(gid, []):
                events.append({"head": "action", "label": c, "position_ms": int(t - v0), "timestamp_utc": iso(int(t))})
                counts[c] += 1
        for p, k in sorted(kicks.items()):
            events.append({"head": "action", "label": "kickOff", "position_ms": int(k - v0), "timestamp_utc": iso(k),
                           "metadata": {"period": p}})
        events.sort(key=lambda e: e["position_ms"])
        games.append({"id": gid, "split": split[gid],
                      "inputs": [{"type": "player_joints_h5", "path": f"{gid}/live_joints.h5", "ball_path": f"{gid}/live_ball.h5"},
                                 {"type": "player_centroids_h5", "path": f"{gid}/live_centroids.h5", "ball_path": f"{gid}/live_ball.h5"}],
                      "events": events})
    payload = {"version": "2.0", "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "task": "action_spotting",
               "dataset_name": f"FWC2022-{source.upper()}-events",
               "description": note,
               "metadata": {"type": "annotations", "created_by": created_by, "split": "SoccerNet-GAR game split, final (128083) held out",
                            "position_ms_reference": "first sample of live_ball.h5", "shift_ms": mod.SHIFT_MS},
               "labels": {"action": {"type": "multi_label", "labels": classes + ["kickOff"]}},
               "modalities": ["player_joints_h5", "player_centroids_h5"],
               "data": games}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"FWC2022-{source.upper()}-events.json"
    p.write_text(json.dumps(payload, indent=1))
    return p, counts


if __name__ == "__main__":
    fdp = load("fdp_labels")
    out = {}
    out["pff"] = package("pff", PFF_CLASSES, "pff_labels.py + crosswalk2.json",
                         "PFF event feed (SoccerNet-GAR release) joined to the FWC2022 skeleton tracking: per-game, per-period UTC offsets fitted on headers; HEADER shifted +167 ms to the contact instant.")
    out["fdp"] = package("fdp", list(fdp.CLASSES), "fdp_labels.py + fdp_crosswalk.json",
                         "FIFA FDP-2022 event data joined to the FWC2022 skeleton tracking: per-game, per-period UTC offsets fitted on head-body events; no shift.")
    for n in ("crosswalk2.json", "fdp_crosswalk.json", "pff_labels.py", "fdp_labels.py", "crosswalk2.py", "fdp_crosswalk.py"):
        shutil.copy(HERE / n, OUT / n)
    for k, (p, counts) in out.items():
        print(p.name, f"{p.stat().st_size/1e6:.1f} MB", {c: v for c, v in counts.items()})


# ----------------------------------------------------------------- h5 export
def to_h5(json_path, h5_path):
    """One HDF5 per source: a group per game, column datasets as in live_ball.h5.

    Columns: label (str), timestamp_utc (str, same format as the tracking),
    position_ms (int64, from the first ball sample), ball_half (int8, the in-play flag of the matched ball frame), frame
    (int64, the nearest 50 Hz ball frame of live_ball.h5), split (str).
    """
    d = json.loads(Path(json_path).read_text())
    with h5py.File(h5_path, "w") as h:
        h.attrs["dataset_name"] = d["dataset_name"]; h.attrs["description"] = d["description"]
        h.attrs["labels"] = json.dumps(d["labels"]["action"]["labels"]); h.attrs["created"] = d["date"]
        for k, v in d["metadata"].items():
            h.attrs[k] = v if isinstance(v, (int, float, str)) else json.dumps(v)
        ids, splits = [], []
        for g in d["data"]:
            gid = g["id"]
            with h5py.File(ROOT / gid / "live_ball.h5", "r") as b:
                bs = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in b["timestamp_utc"][:]])
                bf, bh = b["frame"][:], b["half"][:]
            order = np.argsort(bs, kind="stable"); bs, bf, bh = bs[order], bf[order], bh[order]
            bt = np.array([utc_ms(x) for x in bs], np.int64)
            ev = [e for e in g["events"] if e["label"] != "kickOff"]
            t = np.array([utc_ms(e["timestamp_utc"]) for e in ev], np.int64)
            k = np.clip(np.searchsorted(bt, t), 1, len(bt) - 1)
            k = np.where(np.abs(bt[k - 1] - t) < np.abs(bt[k] - t), k - 1, k)
            grp = h.create_group(gid)
            grp.attrs["split"] = g["split"]
            S = h5py.string_dtype()
            grp.create_dataset("label", data=np.array([e["label"] for e in ev], dtype=object), dtype=S)
            grp.create_dataset("timestamp_utc", data=np.array([e["timestamp_utc"] for e in ev], dtype=object), dtype=S)
            grp.create_dataset("position_ms", data=np.array([e["position_ms"] for e in ev], np.int64))
            grp.create_dataset("frame", data=bf[k].astype(np.int64))
            grp.create_dataset("ball_half", data=bh[k].astype(np.int8))   # in-play flag of the matched ball frame (0 = flagged out of play)
            kick = [e for e in g["events"] if e["label"] == "kickOff"]
            grp.create_dataset("kickoff_utc", data=np.array([e["timestamp_utc"] for e in kick], dtype=object), dtype=S)
            grp.create_dataset("kickoff_period", data=np.array([e["metadata"]["period"] for e in kick], np.int8))
            ids.append(gid); splits.append(g["split"])
        h.create_dataset("games", data=np.array(ids, dtype=object), dtype=h5py.string_dtype())
        h.create_dataset("splits", data=np.array(splits, dtype=object), dtype=h5py.string_dtype())
    return h5_path


if __name__ == "__main__":
    for src in ("PFF", "FDP"):
        p = to_h5(OUT / f"FWC2022-{src}-events.json", OUT / f"FWC2022-{src}-events.h5")
        with h5py.File(p) as h:
            g = h["128083"]
            print(p.name, f"{p.stat().st_size/1e6:.1f} MB", len(h["games"]), "games; final:", len(g["label"]), "events; columns", list(g.keys()),
                  "| first:", g["label"][0], g["timestamp_utc"][0], int(g["frame"][0]), int(g["ball_half"][0]))
