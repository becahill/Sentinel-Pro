# Demo Capture (macOS)

Goal: a 20–40 second clip showing the end-to-end workflow: audit -> dashboard -> filter -> export.

## Recommended flow

1) Run the demo data and open the dashboard:
```bash
python3 auditor.py --input-jsonl data/golden_path.jsonl --project golden-path --tags demo,golden
streamlit run dashboard.py
```

2) In the UI:
- Show the metrics row.
- Filter to "Flagged" only.
- Scroll to "Signal Breakdown".
- Open the record details panel.
- Download CSV.

## Recording steps

1) Press `Cmd + Shift + 5` and choose **Record Selected Portion**.
2) Record a 20–40 second clip, then click **Stop**.
3) Save as `demo.mov`.

## Convert to GIF

```bash
brew install ffmpeg
ffmpeg -i demo.mov -vf "fps=12,scale=1200:-1:flags=lanczos" -loop 0 assets/demo.gif
```

## PNG fallback

```bash
ffmpeg -i demo.mov -vframes 1 assets/demo.png
```

Commit `assets/demo.gif` and `assets/demo.png` after capture.
