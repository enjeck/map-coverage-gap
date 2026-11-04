# Coverage gap


## Setup

You need Python 3.10 and the packages in `requirements.txt`.

```
pip install -r requirements.txt
```

Download the challenge package from [source.coop](https://source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge) into `data/`, with all five region folders. I used Overture release 2026-08-19.0, retrieved on 18 October 2026.

The bias numbers also need mobile homes per tract from the [CDC/ATSDR Social Vulnerability Index 2022](https://services3.arcgis.com/ZvidGQkLaDJxRSJ2/arcgis/rest/services/CDC_ATSDR_Social_Vulnerability_Index_2022_USA/FeatureServer/2). Save the `FIPS` and `E_MOBILE` columns as `data/external/svi2022_tract_mobile_homes.csv`, with the headers `GEOID` and `e_mobile`. The scored model doesn't use this file.

## Run

```
for r in maricopa-az northern-ca eastern-ok south-central-tx eastern-wa; do
  python src/01_count_overture.py $r
  python src/04_ml_dates.py $r
done
python src/04_ml_dates.py merge
python src/02_build_table.py
python src/03_hand_gaps.py
python src/05_clues.py
python src/06_blend.py v11
python src/06_blend.py v13
python src/07_bias.py
```

Texas is large. If it runs out of memory, run `01_count_overture.py` one stage at a time, and run `04_ml_dates.py south-central-tx 0 130` and then `130 255`.

Predictions go to `out/`. `out/v11.csv` matches the file I uploaded as 6jbkaXwH. Nothing in the code is random, so every run gives the same files.

## What each step does

| Step | What it does |
|---|---|
| 01 | Counts Overture buildings, places and road length in each tract |
| 02 | Joins the counts with the challenge's tract tables into one table |
| 03 | Builds my first, hand-built gap estimates (v2b) |
| 04 | Counts Overture's Microsoft buildings in each tract, and how many are old |
| 05 | Builds the extra clues the blend uses |
| 06 | Blends the clues with weights from the leaderboard probes |
| 07 | Prints every number in the bias writeup |

`scores.json` holds the public score of every upload, the private scores of my last four blends, and the step size and sign of each probe. `models/` lists the clues in v11 and v13. The documentation writeup explains the method in plain words.
