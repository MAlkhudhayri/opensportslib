"""Merge the skeleton header annotations with the FIFA FDP-2022 events.

FWC2022-skeleton-headers ships the rule's header detections with the FIFA
player id of the heading player. FDP carries the same player ids, so the two
sources can be matched by player and time, not time alone. Output: one OSL v2
file and one HDF5 with, per game, every rule header and every FDP event of the
15 released classes, each FDP header carrying the index of its rule header
(same player within 500 ms, else nearest within 500 ms, else -1) and each rule
header the index of its FDP header.
"""

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path("/ibex/user/alkhudmk/fwc2022")
OUT = ROOT / "release"
HERE = Path(__file__).resolve().parent
TOL_MS = 500


def load(n):
    s = importlib.util.spec_from_file_location(n, HERE / f"{n}.py")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def utc_ms(s):
    return int(round(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000))


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S.%f")


def ball_clock(gid):
    with h5py.File(ROOT / gid / "live_ball.h5", "r") as f:
        s = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["timestamp_utc"][:]])
        bf, bh = f["frame"][:], f["half"][:]
    order = np.argsort(s, kind="stable")
    s, bf, bh = s[order], bf[order], bh[order]
    bt = np.array([utc_ms(x) for x in s], np.int64)
    return bt, bf, bh


def nearest_frame(bt, bf, bh, t):
    k = np.clip(np.searchsorted(bt, t), 1, len(bt) - 1)
    k = np.where(np.abs(bt[k - 1] - t) < np.abs(bt[k] - t), k - 1, k)
    return bf[k].astype(np.int64), bh[k].astype(np.int8)


def match(a_t, a_p, b_t, b_p):
    """Index in b for each a: same player within TOL first, else nearest within TOL, else -1."""
    out = np.full(len(a_t), -1, np.int64)
    if len(b_t) == 0:
        return out
    for i, (t, p) in enumerate(zip(a_t, a_p)):
        d = np.abs(b_t - t)
        ok = d <= TOL_MS
        if not ok.any():
            continue
        same = ok & (b_p == p) & (p != "")
        cand = np.nonzero(same)[0] if same.any() else np.nonzero(ok)[0]
        out[i] = cand[np.argmin(d[cand])]
    return out


def main():
    fl = load("fdp_labels")
    cw = json.load(open(HERE / "fdp_crosswalk.json"))
    _, split = fl.labels("HEADER")
    ev = pd.read_pickle(fl.EVENTS)
    rule = {g["id"]: g for g in json.load(open(ROOT / "headers_annotations.json"))["data"]}
    class_of = {}
    for c, f in fl.CLASSES.items():
        if c == "HEADER":
            continue
        class_of[c] = f
    games, stats = [], {"rule": 0, "fdp": 0, "fdp_head": 0, "fdp_head_matched": 0, "same_player": 0, "rule_matched": 0}
    for gid in sorted(split):
        bt, bf, bh = ball_clock(gid)
        v0 = int(bt[0])
        # rule headers
        rh = [e for e in rule[gid]["events"] if e["label"] == "header"]
        r_t = np.array([utc_ms(e["timestamp_utc"]) for e in rh], np.int64)
        r_p = np.array([str(e["metadata"].get("player_id") or "") for e in rh])
        # FDP rows of the released classes, with their UTC
        g = ev[ev.match_id == int(gid)].copy()
        g = g[g.half_time.astype(str).isin(cw[gid].keys())]
        g["utc_ms"] = [int(r.match_run_time_in_ms) + cw[gid][str(int(r.half_time))]["offset_ms"] for r in g.itertuples()]
        g["cls"] = ""
        for c, f in class_of.items():
            g.loc[f(g) & (g.cls == ""), "cls"] = c
        g["is_header"] = fl.CLASSES["HEADER"](g)
        g = g[(g.cls != "") | g.is_header].sort_values("utc_ms")
        f_t = g.utc_ms.to_numpy(np.int64)
        f_p = np.array([str(int(x)) if pd.notna(x) else "" for x in g.from_player_id])
        head = g.is_header.to_numpy()
        # matches: FDP headers -> rule headers, rule headers -> FDP headers
        fh_idx = np.nonzero(head)[0]
        m_f = np.full(len(g), -1, np.int64)
        m_f[fh_idx] = match(f_t[fh_idx], f_p[fh_idx], r_t, r_p)
        m_r = match(r_t, r_p, f_t[fh_idx], f_p[fh_idx])
        m_r = np.where(m_r >= 0, fh_idx[np.maximum(m_r, 0)], -1)       # index into the FDP list
        same = (m_f[fh_idx] >= 0) & (r_p[np.maximum(m_f[fh_idx], 0)] == f_p[fh_idx])
        events = []
        rf, rb = nearest_frame(bt, bf, bh, r_t) if len(r_t) else (np.array([], np.int64), np.array([], np.int8))
        for i, e in enumerate(rh):
            events.append({"head": "action", "label": "HEADER", "source": "skeleton_rule", "event": "header",
                           "class": "HEADER",
                           "position_ms": int(r_t[i] - v0), "timestamp_utc": iso(int(r_t[i])), "frame": int(rf[i]), "ball_half": int(rb[i]),
                           "player_id": r_p[i], "to_player_id": "", "team": e["metadata"].get("team", ""),
                           "body_type": "head", "outcome": "", "confidence_score": float(e.get("confidence_score", float("nan"))),
                           "match_index": int(m_r[i]) if m_r[i] < 0 else int(len(rh) + m_r[i])})
        ff, fb = nearest_frame(bt, bf, bh, f_t) if len(f_t) else (np.array([], np.int64), np.array([], np.int8))
        for j, r in enumerate(g.itertuples()):
            events.append({"head": "action", "label": "HEADER" if head[j] else r.cls, "source": "fdp",
                           "event": str(r.event), "class": r.cls,
                           "position_ms": int(f_t[j] - v0), "timestamp_utc": iso(int(f_t[j])), "frame": int(ff[j]), "ball_half": int(fb[j]),
                           "player_id": f_p[j], "to_player_id": str(int(r.to_player_id)) if pd.notna(r.to_player_id) else "",
                           "team": str(r.team_name) if pd.notna(r.team_name) else "", "body_type": str(r.body_type) if pd.notna(r.body_type) else "",
                           "outcome": str(r.outcome) if pd.notna(r.outcome) else "", "confidence_score": float("nan"),
                           "match_index": int(m_f[j])})
        games.append({"id": gid, "split": split[gid],
                      "inputs": [{"type": "player_joints_h5", "path": f"{gid}/live_joints.h5", "ball_path": f"{gid}/live_ball.h5"}],
                      "events": events})
        stats["rule"] += len(rh); stats["fdp"] += len(g); stats["fdp_head"] += int(head.sum())
        stats["fdp_head_matched"] += int((m_f[fh_idx] >= 0).sum()); stats["same_player"] += int(same.sum()); stats["rule_matched"] += int((m_r >= 0).sum())
    payload = {"version": "2.0", "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "task": "action_spotting",
               "dataset_name": "FWC2022-skeleton-headers-x-FDP",
               "description": "The skeleton rule's header detections (FWC2022-skeleton-headers, with FIFA player ids) merged with the FIFA FDP-2022 events of 15 classes, matched by player and time (500 ms). match_index points at the matched event within the same game's list, -1 if none.",
               "metadata": {"type": "annotations", "created_by": "merge_skeleton_fdp.py", "tolerance_ms": TOL_MS,
                            "position_ms_reference": "first sample of live_ball.h5", "player_id": "FIFA player id, shared by both sources", "stats": stats},
               "labels": {"action": {"type": "multi_label", "labels": list(fl.CLASSES)}},
               "modalities": ["player_joints_h5"], "data": games}
    OUT.mkdir(parents=True, exist_ok=True)
    jp = OUT / "FWC2022-skeleton-headers-x-FDP.json"
    jp.write_text(json.dumps(payload, indent=1))
    # HDF5
    hp = OUT / "FWC2022-skeleton-headers-x-FDP.h5"
    S = h5py.string_dtype()
    with h5py.File(hp, "w") as h:
        for k in ("dataset_name", "description", "date"):
            h.attrs[k] = payload[k]
        h.attrs["labels"] = json.dumps(payload["labels"]["action"]["labels"]); h.attrs["stats"] = json.dumps(stats)
        h.attrs["tolerance_ms"] = TOL_MS; h.attrs["position_ms_reference"] = "first sample of live_ball.h5"
        for g in games:
            grp = h.create_group(g["id"]); grp.attrs["split"] = g["split"]
            E = g["events"]
            for col in ("source", "label", "class", "event", "timestamp_utc", "player_id", "to_player_id", "team", "body_type", "outcome"):
                grp.create_dataset(col, data=np.array([e[col] for e in E], dtype=object), dtype=S)
            for col, dt in (("position_ms", np.int64), ("frame", np.int64), ("ball_half", np.int8), ("match_index", np.int64)):
                grp.create_dataset(col, data=np.array([e[col] for e in E], dt))
            grp.create_dataset("confidence_score", data=np.array([e["confidence_score"] for e in E], np.float32))
        h.create_dataset("games", data=np.array([g["id"] for g in games], dtype=object), dtype=S)
        h.create_dataset("splits", data=np.array([g["split"] for g in games], dtype=object), dtype=S)
    print(json.dumps(stats))
    print(jp.name, f"{jp.stat().st_size/1e6:.1f} MB;", hp.name, f"{hp.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
