"""Refine the crosswalk with a per-period offset and validate on the final.

crosswalk.py fits one offset per game, which cannot hold across a halftime
break: the video clock keeps running while the match clock stops. Fitting one
offset per period removes that error, and the residual match rate then measures
something real, namely how many PFF headers the rule never proposed.
"""

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

GAR = "/ibex/user/alkhudmk/sngar-frames"
PSEUDO = "/ibex/user/alkhudmk/fwc2022/headers_annotations.json"
FEAT = "/ibex/user/alkhudmk/fwc2022/features"
HUMAN = "/home/alkhudmk/ai4sports_semester2/pixels_vs_positions/headers_data/FWC2022-final-224p-Milli-OpenSportsLib.json"
BIN_MS, TOL_MS = 200, 500


def utc_ms(s):
    return int(round(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000))


def sngar_by_period():
    """{game: {period: sorted header times ms}}"""
    out = defaultdict(lambda: defaultdict(list))
    for sp in ("train", "valid", "test"):
        for e in json.load(open(f"{GAR}/annotations_{sp}.json"))["data"]:
            if e["labels"]["action"]["label"] != "HEADER":
                continue
            m = e["metadata"]
            per = int(str(m["game_time"]).split(" - ")[0])
            out[m["game_id"]][per].append(m["position_ms"])
    return {g: {p: np.sort(np.array(v, np.int64)) for p, v in d.items()} for g, d in out.items()}


def fwc_rule():
    out = {}
    for g in json.load(open(PSEUDO))["data"]:
        ts = [utc_ms(e["timestamp_utc"]) for e in g["events"] if e["label"] == "header"]
        if ts:
            out[g["id"]] = np.sort(np.array(ts, np.int64))
    return out


def fit(s, f):
    """Modal offset of f - s, then the fraction of s matched within TOL."""
    if len(s) == 0 or len(f) == 0:
        return 0.0, 0, 0
    d = (f[None, :] - s[:, None]).ravel() // BIN_MS
    off = Counter(d.tolist()).most_common(1)[0][0] * BIN_MS
    k = np.clip(np.searchsorted(f, s + off), 1, len(f) - 1)
    near = np.minimum(np.abs(f[k] - (s + off)), np.abs(f[k - 1] - (s + off)))
    hit = near <= TOL_MS
    return float(hit.mean()), int(off), int(hit.sum())


def main():
    S, F = sngar_by_period(), fwc_rule()
    pairs = {k: v["fwc"] for k, v in json.load(open("crosswalk.json")).items()} \
        if False else None
    # recompute the bijection using per-period fitting, period 1 only (cheap, unambiguous)
    rows = []
    for sg, per in S.items():
        s1 = per.get(1, np.array([], np.int64))
        cand = sorted(((fit(s1, f)[0], fg) for fg, f in F.items()), reverse=True)
        rows.append((sg, cand[0][1], cand[0][0], cand[1][0]))
    assign = {sg: fg for sg, fg, _, _ in rows}
    print("bijection:", len(set(assign.values())), "distinct targets for", len(assign), "games")

    tot_s = tot_hit = 0
    print(f"\n{'sngar':>6} {'fwc':>7} {'per':>4} {'nPFF':>5} {'matched':>8} {'rate':>6} {'offset_s':>9}")
    detail = {}
    for sg, fg in sorted(assign.items()):
        f = F[fg]
        g_hit = g_n = 0
        offs = {}
        for p, s in sorted(S[sg].items()):
            rate, off, hit = fit(s, f)
            g_hit += hit; g_n += len(s); offs[p] = off
            if sg == "10517":
                print(f"{sg:>6} {fg:>7} {p:>4} {len(s):>5} {hit:>8} {rate:6.2f} {off/1000:9.1f}")
        detail[sg] = {"fwc": fg, "n": g_n, "matched": g_hit, "offsets_ms": offs}
        tot_s += g_n; tot_hit += g_hit
    print(f"\nall games: {tot_hit} of {tot_s} PFF headers within {TOL_MS} ms of a rule detection "
          f"({100*tot_hit/tot_s:.1f}%)")
    rates = np.array([d["matched"]/d["n"] for d in detail.values()])
    print(f"per-game rate: mean {rates.mean():.3f}  min {rates.min():.3f}  max {rates.max():.3f}")

    # ---- validate on the final against the 139 human labels -------------
    fg = assign["10517"]
    print(f"\n--- validation on the final (sngar 10517 -> fwc {fg}) ---")
    g = json.load(open(HUMAN))["data"][0]
    v0 = utc_ms(next(i for i in g["inputs"] if i["type"] == "video")["UTC_time_start"])
    hum = np.sort(np.array([v0 + e["position_ms"] for e in g["events"] if e["label"] == "Header"]))
    for p, s in sorted(S["10517"].items()):
        rate, off, hit = fit(s, hum)
        print(f"  period {p}: {len(s):>3} PFF headers, {hit:>3} within {TOL_MS} ms of a human label "
              f"({rate:.2f}), offset {off/1000:.1f} s")
    json.dump(detail, open("crosswalk2.json", "w"), indent=1)
    print("\nwrote crosswalk2.json")


if __name__ == "__main__":
    main()
