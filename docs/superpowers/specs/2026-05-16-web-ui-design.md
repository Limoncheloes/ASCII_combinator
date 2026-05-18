# ASCII Combinator — Web UI Design Spec

**Date:** 2026-05-16
**Status:** Approved

## Goal

A local web application that wraps the existing ASCII Combinator CLI in a browser-based interface. The user runs `python3 web_ui.py`, the browser opens automatically, and they can drag-and-drop an image or video, adjust all available parameters via sliders/toggles, convert, and save results to the `results/` directory — all without touching the terminal.

---

## Architecture

### File Structure

```
ASCII_combinator/
├── web_ui.py               # Flask entry point
├── templates/
│   └── index.html          # Single-page app (SPA)
├── static/
│   ├── app.js              # Drag-drop, sliders, fetch, SSE
│   └── style.css           # Dark theme, green/yellow accent
├── results/                # Existing directory, auto-created per run
│   └── <stem>/             # Named after input file stem (e.g. horse/)
│       ├── <original>      # Copy of source file
│       └── result_w<N>_f<M>[_bg-<mode>].<ext>   # ASCII output
└── ascii_combinator/       # Unchanged
```

### Entry Point

```python
# web_ui.py bottom
if __name__ == "__main__":
    import webbrowser, threading
    threading.Timer(1.0, lambda: webbrowser.open("http://localhost:5000")).start()
    app.run(debug=False)
```

`python3 web_ui.py` — starts Flask, auto-opens browser at `http://localhost:5000`.

---

## Backend (web_ui.py)

### Flask Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Serves `index.html` |
| `/api/convert` | POST | Multipart: file + params JSON → runs conversion → returns `{result_url, result_path, task_id}` |
| `/api/progress/<task_id>` | GET | SSE stream: `data: {"frame": 47, "total": 150}` (video only) |
| `/api/open-folder` | POST | Body `{path}` → opens folder in OS file manager (cross-platform) |
| `/results/<path:filename>` | GET | Serves files from `results/` directory |

### Conversion Logic

**Image:**
- Call Python functions directly (no subprocess): open image → run layers → `Compositor` → `Renderer` → save
- Runs synchronously in the Flask request handler (fast enough for images)

**Video:**
- `task_id` = `uuid.uuid4().hex` generated at conversion start, returned immediately
- Create a background thread, run `VideoProcessor.process()`
- Progress written to a shared `dict[task_id → {frame, total}]`
- SSE endpoint reads from that dict and streams updates
- On completion, `dict` entry updated with `{done: true, result_url}`

### Results Naming

```
results/<stem>/result_w<width>_f<fontsize>[_bgs<opacity>][_bgr].{png,mp4,gif}
```

Examples:
- `results/horse/result_w80_f12.png`
- `results/demo/result_w80_f12_fps10.mp4`
- `results/demo/result_w80_f12_fps10.gif`

Source file is copied to `results/<stem>/<original_filename>` on first conversion.

---

## Frontend

### Layout: Split Pane

```
┌──────────────────────────────────────────────────────┐
│  Left panel (280px fixed)  │  Right panel (flex 1)   │
│                            │                          │
│  [drop zone]               │  [result image/video]    │
│  [sliders]                 │                          │
│  [toggles]                 │  [progress bar — video]  │
│  [convert button]          │  [save / open folder]    │
└──────────────────────────────────────────────────────┘
```

- **Green accent** (`#4ade80`) — image mode
- **Yellow accent** (`#f59e0b`) — video mode
- Accent switches automatically on file drop

### Drag-and-drop Zone

- Accepts: `image/*` and `video/*`
- On drop/select: reads MIME type, switches panel mode, shows filename + dimensions
- Click on zone re-opens file picker

### Controls — Shared (image + video)

| Control | Type | Range / Options | Default |
|---------|------|-----------------|---------|
| `width` | slider | 10–300, step 10 | 80 |
| `font-size` | slider | 6–24, step 1 | 12 |
| `jitter` | slider | 0–5, step 1 | 1 |
| `threshold` | slider | 0.00–0.50, step 0.01 | 0.15 |
| `layers` | 4 toggles | brightness / sobel_x / sobel_y / diagonal | all on |
| `bg-mode` | 3-tab selector | keep / soft / remove* | keep |
| `bg-opacity` | slider (conditional) | 0.0–1.0, step 0.05 | 0.25 |
| `bg-chars` | text input (conditional) | any non-empty string | `.,` |
| `profile` | dropdown | monochrome (expandable) | monochrome |

\* `remove` tab hidden in video mode

`bg-opacity` and `bg-chars` appear only when `bg-mode = soft`.

### Controls — Video Only

| Control | Type | Range / Options | Default |
|---------|------|-----------------|---------|
| `fps` | slider | 1–30, step 1 | 10 |
| `workers` | slider | 1–`cpu_count()`, step 1 | `cpu_count()` |
| `frame-step` | number input (optional) | 1–N | empty (disabled) |
| `preview` | checkbox | — | unchecked |
| `+gif` | checkbox | — | unchecked |
| `gif-fps` | slider (conditional) | 1–30, step 1 | 10 |

`gif-fps` appears only when `+gif` is checked.

### Convert Button

Single "▶ Конвертировать" button for both modes. No live preview — conversion only on explicit click.

### Right Panel — Image Result

- Full-size PNG displayed via `<img>` tag
- Below: file path label + `📁 Открыть папку` button (calls `/api/open-folder`; backend uses `subprocess` with `xdg-open` / `open` / `explorer` depending on `platform.system()`)
- `📋 Копировать путь` button — `navigator.clipboard.writeText(path)` (pure JS, no server call)

### Right Panel — Video Result

- Progress bar: `frame X / total` updated via SSE
- On completion: preview frame image shown
- Buttons: `💾 MP4`, `🎞 GIF` (if gif was requested)
- `📁 Открыть папку` button

---

## Error Handling

- File type not image or video → red banner: "Поддерживаются только изображения и видео"
- No layers selected → convert button disabled, tooltip "Выбери хотя бы один слой"
- `bg-mode = soft` with empty `bg-chars` → convert button disabled
- ffmpeg not found (video) → error message in right panel with install hint
- Conversion Python exception → JSON `{error: "..."}` shown in right panel

---

## Dependencies

New dependency added to `requirements.txt`:
```
flask>=3.0.0
```

No other new dependencies. `webbrowser` and `threading` are stdlib.

---

## Out of Scope

- Authentication / multi-user
- Cloud storage
- Batch processing multiple files at once
- History / undo
- Real-time live preview (sliders update result automatically)
