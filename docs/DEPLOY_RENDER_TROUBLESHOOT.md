# Render deployment notes / troubleshooting

## Fixing occasional 502 after deploy
If Render shows **502** or "connection failed", but logs say Streamlit started, the most common cause is a slow or failing write to the Matplotlib font cache (especially on fresh containers).

This repo sets:

- `MPLCONFIGDIR=/tmp/matplotlib`
- `XDG_CACHE_HOME=/tmp`
- `browser.gatherUsageStats=false`

via `render.yaml` (and `.streamlit/config.toml` as a backup).

If you still see 502, try:
1. Trigger a manual redeploy (Render dashboard).
2. Increase plan (Starter/Standard) for more CPU/RAM and faster cold start.
3. Reduce heavy imports at top-level (optional): move `matplotlib.pyplot` import inside the plotting section.

## Port binding
Render expects the process to bind to `$PORT`. The start command uses:

`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`


## Non-ASCII filenames (日本語ファイル名) に注意

Render/Linux環境では、**リポジトリ内の日本語ファイル名**が原因でビルド/起動時に予期せぬ問題が出ることがあります（特に assets 配下など）。

- 対策1: リポジトリ内のファイル名は **英数字/ASCIIのみ** にする
- 対策2: 付属スクリプトで一括変換: `python tools/rename_non_ascii.py`

変換後は README 内の参照（画像パスなど）も合わせて修正してください。
