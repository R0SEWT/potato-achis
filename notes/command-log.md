# Command Log

Commands executed to review the validation notebook:

- `ls`
- `find .. -name AGENTS.md -print`
- `ls notebooks`
- `python - <<'PY' ...` (parsed `notebooks/01_exploration.ipynb` to locate validation-related cells)
- `rg -n "class_filter|PotatoDiseaseDataset|DataModule|split|val" -n src/data`
- `sed -n '1,220p' src/data/datamodule.py`
- `nl -ba notebooks/01_exploration.ipynb | sed -n '260,520p'`
- `nl -ba notebooks/01_exploration.ipynb | sed -n '520,760p'`
- `nl -ba src/data/datamodule.py | sed -n '100,190p'`
