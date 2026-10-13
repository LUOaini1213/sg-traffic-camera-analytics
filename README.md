# sg-traffic-camera-analytics

Vehicle detection and congestion flags on Singapore's public LTA traffic cameras: a labelled
set of 330 camera frames, a YOLO11n detector fine-tuned on it and scored on cameras it never
saw, a streaming service (Redpanda/Kafka API -> detector -> per-camera state -> HTTP), local
MLflow tracking with a model registry and a drift check, and a comparison of the camera flags
with LTA speed bands and incident reports.

Built on a laptop (GTX 1650 4 GB, Windows 11, WSL2 + Docker) between 2026-10-13 and
2026-10-14. Everything below is measured on that machine; the numbers come from the files in
`results/`, and CI fails if a number quoted here stops matching them
(`scripts/check_readme.py`).

## Results at a glance

| What | Result | File |
|---|---|---|
| Detection, 65 frames from 14 held-out cameras, mAP50 | COCO YOLO11n 0.413 -> fine-tuned 0.539 (95% CI 0.473-0.653) | `results/detection_eval.json` |
| Same, class-agnostic AP50 (is there a vehicle?) | 0.707 -> 0.824 | same |
| Vehicle-count error per frame (MAE) | 5.78 -> 2.37 | same |
| Streaming replay, 1,024 frames through Redpanda | 14.8 frames/s, 56 ms model time per frame | `results/stream_week_replay.json` |
| Live run on the DataMall feed | see "Streaming service" | `results/stream_live.json` |
| Counts vs LTA speed bands (archive week) | Spearman -0.78 on one camera, near zero on the others; flag/slow kappa 0.11 | `results/validation_week.json` |
| Small local VLM as scene tagger (65 frames) | day/night 93.8% (majority 56.9%); rain and traffic level no better than guessing | `results/vlm_eval.json` |
| Tests / planted bugs caught | 39 tests; 23/23 mutations caught | CI |

## Data and licence

| Source | What | Used for |
|---|---|---|
| LTA DataMall `Traffic-Imagesv2` | current frame of each camera (signed link, valid ~5 min) | live collector and live streaming |
| data.gov.sg `transport/traffic-images` | the same LTA feed, queryable by `date_time`, persistent image URLs with MD5 | labelling set and the archive week |
| LTA DataMall `TrafficIncidents`, `v4/TrafficSpeedBands` | incident list; speed band per road link | checking congestion flags |
| data.gov.sg `environment/rainfall` | 5-minute rain-gauge readings | choosing rainy frames; rain reference for the VLM check |

**Only 8 cameras are published now.** Until 2026-06-29 both image endpoints listed 90 cameras; from 2026-06-30 they list 8 (found on 2026-10-13 by bisecting `date_time`). So the labelling set comes from the archive (2025-07 to 2026-06, 90 cameras) and everything live runs on the cameras still published.

**Scope.** Cameras west of 103.66 E are left out everywhere (`src/tcam/cameras.py`). That leaves 86 of the 90 archive cameras and 5 of the 8 live ones (2701, 2702, 2704, 4798, 4799). Eleven archive cameras publish only 320x240 frames; those frames are not labelled.

**Licence and what is committed.** The datasets are under the Singapore Open Data Licence. LTA's API terms grant no rights over personal data or third-party content that may appear in what the APIs return, and camera frames can show number plates and people, so **no camera image is committed**. The repository holds derived data only: frame manifests (camera, time, public archive URL, MD5), model pre-annotations, review corrections, final YOLO labels, per-frame counts and metrics. `scripts/fetch_frames.py` and `scripts/fetch_week_frames.py` re-download the exact archive frames and check each MD5. Live frames cannot be re-fetched (DataMall serves only the current image), so for those only counts and timings are committed. Contains information from LTA DataMall and data.gov.sg accessed under the Singapore Open Data Licence.

The DataMall AccountKey is read from the `LTA_KEY` environment variable at run time; it is never written to disk or logged.

## Labels: how they were made

**These are not human hand-labels.** Every box started as a model prediction and was then checked by an AI assistant (Claude) looking at each rendered frame. No person drew or checked the boxes. The steps:

1. **Frame choice** (`scripts/sample_archive.py`, seed 13). 76 random times between 2025-07-01 and 2026-06-29, with rainy times over-sampled using the rain-gauge archive, and 5 cameras per time, rotating so that every in-scope camera appears 4-5 times. That gives 380 frames; 47 are 320x240 and were dropped, leaving 333.
2. **Pre-annotation** (`scripts/preannotate.py`). COCO-pretrained YOLO11x on the full frame (imgsz 1920) and on four overlapping tiles (imgsz 1280), merged with class-agnostic NMS, confidence >= 0.20. COCO car/bus/truck/motorcycle map to the four classes. Output: `data/pre/*.json`, one numbered box list per frame.
3. **Visual review** (`scripts/review_render.py`; corrections in `data/review/*.json`). Each frame was rendered with numbered, colour-coded boxes on a 100-px grid, zoomed where needed, and corrected by removing false or duplicate boxes, changing the class, refitting a box, or adding a missed vehicle with coordinates read off the grid. Each correction file names the reviewer and carries a one-line note of what was changed and why.
4. **Final labels** (`scripts/build_dataset.py`) = pre-annotation + corrections, in `data/labels/*.txt` (YOLO format).

Class rules used in review: vans, MPVs, pickups and minibuses are `car`; lorries, tippers, mixers, tankers and prime movers are `truck`; public buses and coaches are `bus`; a rider and motorcycle together are one `motorcycle`. Vehicles whose shorter side is under 10 px on a 1920-px frame are neither labelled nor scored (`MIN_SIDE_FRAC`, applied identically to labels and predictions).

What the review changed (`results/label_stats.json`), over the 330 frames kept:

| | boxes |
|---|---|
| model pre-annotations | 4,888 |
| removed (false positives, duplicates, tile-edge fragments, signs, road markings) | 442 |
| class changed | 302 |
| refitted | 22 |
| added (missed vehicles) | 519 |
| **final labels** | **4,965** |

304 of the 330 frames got at least one correction; in 29 frames the pre-annotation found nothing and the review added vehicles (mostly night, rain or motion blur). Three checkpoint frames with several hundred vehicles were excluded because their labels could not be made close to complete.

**Known weaknesses of these labels.** Added boxes are placed by eye on a grid, so they are looser than model boxes; night boxes drawn from headlight pairs are approximate. Dense far-field queues were reviewed coarsely and still miss small vehicles, so dense frames under-count. The van/car versus truck boundary is a convention set during review, and the COCO model does not share it. Because the labels start from a YOLO11x pre-annotation, YOLO-style boxes are favoured when scored against them; the pre-annotation's own score is reported below so the size of that effect is visible (its mAP50-95 of 0.499 is high for the same reason: most label boxes *are* its boxes).

**Split.** Cameras closer than 600 m are grouped, then whole groups are assigned (`src/tcam/split.py`, seed 13): 235 train frames (54 cameras), 30 validation frames (7 cameras), 65 test frames (14 cameras). The five cameras still live are kept in train because they are the deployment cameras; test cameras appear nowhere in training or model selection. Test labels: 777 car, 18 bus, 131 truck, 42 motorcycle.

## Detector: fine-tuning and held-out evaluation

`scripts/train.py` fine-tunes COCO YOLO11n at imgsz 1280 for 40 epochs (AdamW, cosine LR, mosaic off for the last 10 epochs). Batch size is 1 because the GPU was shared with another process's LLM server; at batch 2 and 4 the runs ran out of memory or slowed 10-50x (those attempts are kept in MLflow as FAILED/KILLED, see `results/mlflow_runs.json`). fp16 is off: on this GTX 1650, half-precision inference returned no boxes at all.

Three fine-tuned runs were trained and the one to deploy was chosen **on the validation cameras** (`results/detection_eval_val.json`): seed 0 0.585, seed 1 0.605, and a variant that starts the class head from the COCO head's car/bus/truck/motorcycle rows 0.561. Seed 1 was registered as the champion. Test results (`scripts/evaluate.py`; COCO-style 101-point AP at IoU 0.5; 95% intervals from 2,000 bootstrap resamples **of cameras**, since frames from one camera are not independent):

| Model (imgsz) | mAP50 | 95% CI | car | bus | truck | motorcycle | class-agnostic AP50 | count MAE / frame |
|---|---|---|---|---|---|---|---|---|
| COCO YOLO11n (640) | 0.234 | 0.184-0.289 | 0.486 | 0.158 | 0.264 | 0.030 | 0.499 | 9.25 |
| COCO YOLO11n (1280) | 0.413 | 0.324-0.472 | 0.707 | 0.389 | 0.424 | 0.134 | 0.707 | 5.78 |
| fine-tuned, seed 0 | 0.477 | 0.442-0.562 | 0.824 | 0.037 | 0.519 | 0.529 | 0.823 | 2.40 |
| **fine-tuned, seed 1 (champion)** | **0.539** | **0.473-0.653** | 0.818 | 0.279 | 0.520 | 0.541 | 0.824 | 2.37 |
| fine-tuned, COCO-initialised class head | 0.461 | 0.426-0.539 | 0.814 | 0.021 | 0.513 | 0.497 | 0.813 | 2.74 |
| pre-annotation (YOLO11x, 5 passes) | 0.510 | 0.402-0.610 | 0.792 | 0.470 | 0.472 | 0.306 | 0.825 | 2.54 |

What this does and does not show:

- **Finding vehicles and counting them clearly improved**, in all three runs. Class-agnostic AP50 rose from 0.707 to 0.824 for the champion (paired camera-bootstrap difference +0.094 to +0.135), and the per-frame count error fell from 5.78 to 2.37 vehicles (difference -5.24 to -1.85). The COCO model mostly misses small and dark vehicles: its recall at confidence 0.25 is 0.494 against 0.721 after fine-tuning. Day and night frames gained by similar amounts (night mAP50 0.267 -> 0.408, day 0.451 -> 0.598).
- **The headline mAP50 gain is less certain than it looks.** For the champion the paired interval is +0.077 to +0.245, but seed 0, trained identically, sits at 0.477 and its interval (-0.005 to +0.213) includes zero. The gap between seeds is almost all the **bus** class: there are only 55 bus boxes in training and 18 in test, and the fine-tuned models often call buses trucks. Starting the class head from COCO's bus weights did not fix this (0.021).
- **The fine-tuned YOLO11n (one pass, 2.6M parameters) matches the YOLO11x tiled pre-annotation on vehicle presence** (0.824 vs 0.825 class-agnostic) and has a smaller count error, but this comparison is tilted towards the pre-annotation, whose boxes the labels were built from.
- 65 test frames from 14 cameras is a small test set; the intervals above are the honest size of the uncertainty.

## Streaming service

```
DataMall poller ──> tcam.frames (Redpanda, Kafka API) ──> detector (YOLO11n, GPU) ──> tcam.camera-state ──> FastAPI /cameras
                                                              │
                                                              └─> JSONL log (counts, flag, timings)
```

- `src/tcam/stream/producer.py`: `live` polls `Traffic-Imagesv2` every 30 s and publishes only frames whose MD5 changed (cameras refresh every 1-5 minutes); `replay` publishes a frame manifest as fast as the broker accepts.
- `src/tcam/stream/detector.py`: consumes in batches of up to 4, counts vehicles per class (`src/tcam/counting.py`), updates the per-camera congestion flag (`src/tcam/flags.py`) and publishes the camera state with timings.
- `src/tcam/stream/api.py`: FastAPI service that keeps the latest state per camera (`GET /cameras`, `GET /cameras/{id}`); sample responses from the live run are in `results/api_*_sample.json`.
- `docker-compose.yml`: single-node Redpanda. On this machine it runs in WSL2 Docker and the Windows-side clients connect to the WSL VM address (the localhost relay dropped Kafka connections).

**Congestion flag.** A camera is flagged when the median of its last 3 vehicle counts reaches a threshold, and released when that median falls below 0.8 x threshold. The threshold is the camera's own 85th-percentile count on a calibration period, never below 8 vehicles (`data/thresholds.json`, calibrated on 2026-09-26 to 09-29 archive frames only).

**Throughput (replay).** 1,024 archive frames through Redpanda in 69.1 s: **14.8 frames/s**, median model time 56 ms per 1280-px frame at batch 4. Another process's LLM server was using the same GPU during this run, so this is a lower bound for this laptop rather than a clean benchmark. The five live cameras together produce about one new frame per minute (each refreshes every few minutes), so the service runs far below capacity.

LIVE_STREAM_SECTION

## Model operations

- **Tracking** (`scripts/train.py`): every training attempt is an MLflow run in a local SQLite store with its parameters, per-epoch metrics, training time and weights; failed and abandoned attempts stay in the store with the reason as a tag. `results/mlflow_runs.json` is a plain copy of the run table.
- **Registry** (`scripts/register_model.py`): the three finished runs are versions 1-3 of `sg-traffic-vehicle-detector`, each with its held-out test metrics logged back to the run; version 2 carries the alias `champion` (`results/mlflow_registry.json`). The streaming detector loads the champion weights.
- **Drift check** (`scripts/drift_check.py`): compares each live camera's count distribution in the live window with the archive week, using the population stability index (PSI, alert above 0.25) and the KS statistic, on all hours and on the hours both windows cover. It is logged as an MLflow monitoring run.

DRIFT_SECTION

## Checking the flags against LTA speed bands and incidents

Rules fixed before any agreement number was computed (`src/tcam/validate.py`): a camera's speed context is the mean `SpeedBand` of category 1-4 road links within 200 m; "slow" means that mean is at most 3 (band 3 = 20-29 km/h); an incident counts if it is within 1 km and in the incident snapshot closest in time.

**Archive week** (1,024 frames, 2026-09-26 to 10-03, at the times of an existing speed-band log; `results/validation_week.json`). Thresholds calibrated on the first 3.2 days, flags scored on the rest:

| Camera | frames scored | Spearman(count, speed band), all frames | slow share | flag share | flag vs slow |
|---|---|---|---|---|---|
| 2701 | 78 | -0.19 | 18% | 45% | 6 TP, 29 FP, 8 FN, 35 TN |
| 2702 | 81 | -0.25 | 100% | 64% | links always slow, uninformative |
| 2704 | 80 | -0.78 | 0% | 72% | never slow, uninformative |
| 4798 | 77 | -0.06 | 0% | 35% | never slow |
| 4799 | 80 | -0.07 | 0% | 32% | never slow |

Pooled flag-versus-slow agreement is weak (kappa 0.11, precision 0.29, recall 0.61). The camera counts track the speed bands clearly on only one camera (2704, Spearman -0.78: more vehicles in view when the nearby expressway links are slower). The reasons are visible in the data rather than guessed: the links near 2702 are always band 1-2 (checkpoint service roads), the links near 2704/4798/4799 never drop to band 3 in this week, and the frames are 20 minutes apart, so a 3-frame median spans an hour. The flag thresholds were also calibrated on a period that included a weekend and scored on weekdays, which pushes the flag share up. Consecutive frames are autocorrelated, so these correlations should not be read with i.i.d. significance tests.

LIVE_VALIDATION_SECTION

## Vision-language model as a scene tagger

`scripts/vlm_eval.py` asks gemma3:4b (local Ollama, run on CPU because the GPU was busy, about 50 s per frame) for `time_of_day`, `rain` and `traffic` on the 65 held-out test frames, and scores the answers against references that do not come from the model: capture hour, the nearest NEA rain gauge, and the reviewed label count (<=10 light, 11-30 moderate, >30 heavy).

| Field | accuracy | majority-class baseline |
|---|---|---|
| time of day | 93.8% | 56.9% |
| rain | 93.8% | 95.4% |
| traffic level | 50.8% | 44.6% |

Only day/night is useful. The rain answer is worse than always saying "no rain" (only 3 test frames had gauge rain, and the model raised 2 false alarms), and the traffic level is barely above the baseline: the model calls most light-traffic frames "moderate" (24 of 29). A model of this size is not a reliable incident or congestion classifier on these frames; this was not pursued further.

## Engineering checks

- `tests/`: 39 unit tests on counting (ROI, thresholds, size floor), the congestion flag (median window, hysteresis, calibration), AP and matching (hand-computed cases), the camera bootstrap, speed-band and incident matching, PSI/KS, label application and the camera split.
- `scripts/mutation_check.py`: plants 23 known bugs (exclusive thresholds, mean instead of median, no hysteresis, no precision envelope, labels matched twice, wrong kappa, flipped PSI, ignored review edits, and more) one at a time and requires the test suite to fail on each. All 23 are caught. The first run let 3 through: two exposed weak tests, which were strengthened, and one was a mutation that cannot change behaviour (flipping the ray-casting direction in point-in-polygon), which was replaced.
- `scripts/check_readme.py`: every number listed in `results/readme_claims.json` must appear in this README exactly as the result file gives it.
- GitHub Actions runs all three on every push. No training or GPU step runs in CI.

## Reproduce

```bash
pip install -r requirements.txt                    # torch with CUDA installed separately
python scripts/fetch_frames.py                     # 380 archive frames, MD5-checked (TCAM_DATA sets the folder)
python scripts/build_dataset.py                    # labels + split + YOLO folders
python scripts/train.py --model yolo11n.pt --imgsz 1280 --epochs 40 --batch 1 --seed 1
python scripts/evaluate.py --ft "ft-yolo11n_1280_e40_s1=<runs>/yolo11n_1280_e40_s1/weights/best.pt:1280"
TCAM_ADVERTISE_HOST=<host ip> docker compose up -d
TCAM_BOOTSTRAP=<host ip>:39092 bash scripts/run_stream.sh replay data/week_frames_manifest.csv <data>/week/images out.jsonl best.pt
LTA_KEY=... python -m tcam.collect <data>/live --until YYYYmmddHHMM
python -m pytest -q && python scripts/mutation_check.py && python scripts/check_readme.py
```

Re-labelling from scratch is `scripts/preannotate.py` followed by a new review; the committed `data/review/*.json` files reproduce the current labels exactly.

## Limitations

- The labels are model pre-annotations corrected by an AI reviewer, not by a person, and they are known to miss small vehicles in dense far-field queues.
- The test set is 65 frames from 14 cameras; bus has 18 test boxes, so per-class bus numbers are noise-dominated.
- Training and the streaming benchmark ran on a GPU shared with other processes; times and throughput are what this machine delivered under that load.
- Only 5 cameras are live, and the speed-band links near them are poor proxies for what each camera sees (camera direction is not published).
- Congestion here means "many vehicles in view, persistently"; the system does not track vehicles, measure speed or detect stopped vehicles.
