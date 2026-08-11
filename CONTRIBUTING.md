# Contributing

Small, focused pull requests are preferred. Add or update tests for behavior changes and include a short benchmark note when performance or scoring logic changes.

## Development

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
pytest -q
```
