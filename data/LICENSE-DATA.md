# Data terms

The MIT licence in `LICENSE` covers the code in this repository. It does not cover the data
the code was built from:

- **Source data.** Frame manifests (`data/*_manifest.csv`), the per-frame counts, timings and
  speed-band/incident series in `results/` are derived from LTA DataMall (`Traffic-Imagesv2`,
  `TrafficIncidents`, `v4/TrafficSpeedBands`) and data.gov.sg (`transport/traffic-images`,
  `environment/rainfall`). Contains information from LTA DataMall and data.gov.sg accessed
  under the Singapore Open Data Licence.
- **No camera images are committed.** LTA's API terms grant no rights over personal data or
  third-party content that may appear in what the APIs return, and camera frames can show
  number plates and people. `scripts/fetch_frames.py` and `scripts/fetch_week_frames.py`
  download the frames from their public URLs instead; anyone doing so is bound by the same terms.
- **Labels.** `data/pre`, `data/review` and `data/labels` are this project's annotations of
  those frames (model pre-annotations plus review corrections; see the README).
