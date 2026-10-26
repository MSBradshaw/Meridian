# Applied Data Science Take-Home

Start with [`client-brief.md`](client-brief.md) — the client, their problem, and what
they asked for. Then read [`assignment.md`](assignment.md) — what to produce, the
submission contract, and how it is evaluated.

## Setup

```bash
uv sync
```

## Run

```bash
uv run main.py train
uv run main.py score
```

## Test

```bash
uv run pytest
```

## Layout

- `main.py` — entrypoint script.
- `assignment/` — the library where the assignment is implemented.
- `tests/` — tests.
- `data/` — the extract.

---

# Methodology

*Candidate: replace this section with your write-up. Per `assignment.md` it is a
brief summary of your methodology.*

