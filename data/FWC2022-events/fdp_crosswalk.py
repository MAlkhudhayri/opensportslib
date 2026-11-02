"""Align the FIFA (FDP-2022) event feed with our tracking clock, per period.

FDP uses our match ids, so no id crosswalk is needed. Its events carry a
match run time in ms (a clock that keeps running through the break), and the
tracking carries UTC. We fit one UTC offset per period in two independent
ways and keep both:

  modal   the modal value of (rule header UTC - head event run time) over the
          head-body events of that period, as crosswalk2.py did for PFF;
  phase   tracking UTC of the first in-play frame of the period minus the
          period's phase_start in the metadata; label free, used as the
          fallback where a period has too few headers (extra time, shoot-outs).

Writes fdp_crosswalk.json {game: {period: {"offset_ms", "method", "n", "matched"}}}.
"""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/ibex/user/alkhudmk/fwc2022")
FDP = Path("/ibex/user/alkhudmk/fdp2022/FDP-2022-original-event-data/games")
EVENTS = Path("/ibex/user/alkhudmk/fdp2022/events_all.pkl")
PSEUDO = ROOT / "headers_annotations.json"
HERE = Path(__file__).resolve().parent
BIN_MS, TOL_MS, MIN_HEADERS = 200, 500, 8


def utc_ms(s):
    return int(round(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000))


def hms_ms(s):
    h, m, sec = s.split(":")
    return (int(h) * 3600 + int(m) * 60 + int(sec)) * 1000


def metadata(gid):
    d = next(FDP.glob(f"M*/*_Metadata.json"), None)
    for p in FDP.glob("M*/*_Metadata.json"):
        m = json.load(open(p))
        if int(m["match_id"]) == int(gid):
            return m
    raise KeyError(gid)


def rule_headers():
    out = {}
    for g in json.load(open(PSEUDO))["data"]:
        ts = [utc_ms(e["timestamp_utc"]) for e in g["events"] if e["label"] == "header"]
        out[g["id"]] = np.sort(np.array(ts, np.int64))
    return out


def modal_fit(run_ms, rule_utc):
    """Offset = mode of (rule - run) in BIN_MS bins; match rate of run events within TOL."""
    if len(run_ms) == 0 or len(rule_utc) == 0:
        return None, 0, 0
    d = (rule_utc[None, :] - run_ms[:, None]).ravel() // BIN_MS
    off = Counter(d.tolist()).most_common(1)[0][0] * BIN_MS
    # refine: median of the matched residuals
    k = np.clip(np.searchsorted(rule_utc, run_ms + off), 1, len(rule_utc) - 1)
    res = np.where(np.abs(rule_utc[k] - (run_ms + off)) < np.abs(rule_utc[k - 1] - (run_ms + off)),
                   rule_utc[k] - (run_ms + off), rule_utc[k - 1] - (run_ms + off))
    hit = np.abs(res) <= TOL_MS
    if hit.sum():
        off += int(np.median(res[hit]))
    return int(off), int(hit.sum()), len(run_ms)


def main():
    ev = pd.read_pickle(EVENTS)
    rule = rule_headers()
    out, rows = {}, []
    for gid in sorted(ev.match_id.unique()):
        gid = str(int(gid))
        meta = metadata(gid)
        phase_start = {int(p["phase_id"]): hms_ms(p["phase_start"]) for p in meta["phases"]}
        d = np.load(ROOT / "features" / f"{gid}.npz")
        t_ms, half = d["t_ms"], d["half"]
        g = ev[ev.match_id == int(gid)]
        out[gid] = {}
        for per in sorted(int(x) for x in g.half_time.dropna().unique()):
            frames = np.nonzero(half == per)[0]
            off_phase = int(t_ms[frames[0]] - phase_start[per]) if len(frames) and per in phase_start else None
            heads = np.sort(g[(g.half_time == per) & (g.body_type == "head")].match_run_time_in_ms.to_numpy(np.int64))
            off_modal, matched, n = modal_fit(heads, rule.get(gid, np.array([], np.int64)))
            use_modal = off_modal is not None and n >= MIN_HEADERS and matched >= 0.6 * n
            off = off_modal if use_modal else off_phase
            out[gid][str(per)] = {"offset_ms": off, "method": "modal" if use_modal else "phase",
                                  "n_head_events": int(n), "matched": int(matched),
                                  "offset_modal_ms": off_modal, "offset_phase_ms": off_phase,
                                  "n_frames": int(len(frames))}
            rows.append((gid, per, n, matched, off_modal, off_phase,
                         None if (off_modal is None or off_phase is None) else off_modal - off_phase))
    # fallback periods: the phase offset carries a systematic bias (the in-play
    # flag starts a few seconds before the kickoff event), so correct it with
    # the game's own modal-minus-phase difference, or the global median
    all_diff = [v["offset_modal_ms"] - v["offset_phase_ms"] for g in out.values() for v in g.values()
                if v["method"] == "modal" and v["offset_phase_ms"] is not None]
    glob_med = int(np.median(all_diff))
    for gid, pers in out.items():
        own = [v["offset_modal_ms"] - v["offset_phase_ms"] for v in pers.values()
               if v["method"] == "modal" and v["offset_phase_ms"] is not None]
        corr = int(np.median(own)) if own else glob_med
        for v in pers.values():
            if v["method"] == "phase" and v["offset_phase_ms"] is not None:
                v["offset_ms"] = v["offset_phase_ms"] + corr
                v["method"] = "phase+corr"
    (HERE / "fdp_crosswalk.json").write_text(json.dumps(out, indent=1))
    print(f"{'game':>7} {'per':>3} {'nHead':>5} {'match':>5} {'modal-phase ms':>14}")
    diffs = []
    for gid, per, n, m, om, op, diff in rows:
        print(f"{gid:>7} {per:>3} {n:>5} {m:>5} {'' if diff is None else diff:>14}")
        if diff is not None and n >= MIN_HEADERS:
            diffs.append(diff)
    diffs = np.array(diffs)
    tot_n = sum(r[2] for r in rows); tot_m = sum(r[3] for r in rows)
    print(f"\nperiods: {len(rows)}; head events {tot_n}, matched to a rule header within {TOL_MS} ms: {tot_m} ({100*tot_m/tot_n:.1f}%)")
    print(f"modal vs phase offset over {len(diffs)} periods with >= {MIN_HEADERS} headers: "
          f"median {np.median(diffs):.0f} ms, IQR {np.percentile(diffs,25):.0f}..{np.percentile(diffs,75):.0f}, "
          f"|diff| > 1 s in {(np.abs(diffs) > 1000).sum()} periods")
    print("method used:", Counter(v["method"] for g in out.values() for v in g.values()))


if __name__ == "__main__":
    main()
