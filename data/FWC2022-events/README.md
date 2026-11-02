# FWC2022 skeleton tracking x event labels (PFF and FIFA FDP-2022)

Two independent event-label sets aligned to the 3D skeletal tracking of the 64
matches of the 2022 FIFA World Cup (OpenSportsLab/FWC2022-skeleton-headers).
Each file is one OSL v2 annotation file with every class, one game per entry,
`position_ms` relative to the first sample of the game's `live_ball.h5`, and
`timestamp_utc` absolute. Game ids are the tracking ids (128068..133041).

| file | source | classes | events |
|---|---|---|---|
| FWC2022-PFF-events.json | PFF event feed (SoccerNet-GAR release) | HEADER, PASS, CROSS, THROW IN, FREE KICK, SHOT, OUT, PLAYER SUCCESSFUL TACKLE, GOAL, HIGH PASS | 87,939 |
| FWC2022-FDP-events.json | FIFA FDP-2022 original event data | HEADER, PASS, CROSS, THROW IN, FREE KICK, CORNER, GOAL KICK, SHOT, CLEARANCE, TACKLE, INTERCEPTION, AERIAL DUEL, BLOCK, FOUL, GOAL | 94,617 |

## How the clocks were aligned

PFF: game ids and clocks differ from the tracking. `crosswalk2.json` maps each
PFF game to a tracking game (64 to 64, no collisions) with one UTC offset per
period, fitted as the modal offset between PFF headers and the rule detected
headers of the tracking. PFF stamps the possession event, which precedes the
contact by 167 ms (median against the final's human labels, sd 61 ms); HEADER
timestamps carry that shift, other classes do not.

FDP: match ids equal the tracking ids. `fdp_crosswalk.json` holds one UTC
offset per period, fitted the same way on head-body events (94.2% of 4,227
matched within 500 ms); seven short periods use the metadata phase start with
the game's own correction. FDP marks the contact itself (median +8 ms, sd 74
ms against the human labels), so no shift.

HEADER: FDP tags the body part only on passes, clearances, attempts and
throw-ins, so it has 4,227 headers against PFF's 5,723; 92% of FDP headers have
a PFF header within 500 ms. PFF is the fuller header label, FDP the cleaner clock.

## Split

`split` per game follows the SoccerNet-GAR split (45 train, 9 valid, 10 test);
the final (128083), the only game with human header labels, is marked train
in GAR but was held out of training in all reported experiments.

## Files

`pff_labels.py`, `fdp_labels.py` rebuild the labels from the raw feeds and the
crosswalks; `crosswalk2.py`, `fdp_crosswalk.py` refit the offsets.
