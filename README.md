# OCELOT-
Opitcal Character recognition Editor by Location Of Text

## Interfaces

- `ocelotkivy.py` — the actively developed UI built with Kivy. Includes zoomable canvas, undo/redo, handwriting toggle, and Windows pen/stylus fixes.
- `ocelot.py` — the original Tkinter proof-of-concept. Still usable, but missing most of the new Kivy tooling.

## Quick start

```powershell
poetry install
poetry run python ocelotkivy.py  # run the new UI
```

If you prefer the classic Tk build:

```powershell
poetry run python ocelot.py
```

- The Kivy build now disables the legacy `wm_pen` provider to avoid crashes on newer Windows drivers. If you need stylus events from that provider, remove the override at the top of `ocelotkivy.py`.
- Both front-ends can load/save OCR regions from the `input/` directory; drop images there to keep dialogs pointing to the right spot.

should be able to handle audio files but is not tested 

## Poetry notes

- Use `poetry shell` to drop into the virtual environment if you prefer an interactive session.
- `poetry lock --no-update` refreshes the lock file after editing dependencies by hand.
- To add packages, run `poetry add <name>`; they'll be tracked in `pyproject.toml` and `poetry.lock`.


