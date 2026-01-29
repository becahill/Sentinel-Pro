# Demo Capture (macOS)

1) Run the demo:
```bash
python3 auditor.py --demo
streamlit run dashboard.py
```

2) Open the Streamlit app in your browser.

3) Record a short GIF using built-in tooling:
- Press `Cmd + Shift + 5` and choose **Record Selected Portion**.
- Record a 10–15 second clip (filters, drill-down, and a chart).
- Click **Stop** in the menu bar.

4) Convert the `.mov` to a GIF (optional but recommended):
```bash
brew install ffmpeg
ffmpeg -i demo.mov -vf "fps=12,scale=1200:-1:flags=lanczos" -loop 0 assets/demo.gif
```

5) Replace the placeholder `assets/demo.gif` with your new capture.
