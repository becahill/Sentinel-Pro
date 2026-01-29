PY?=python3

install:
	$(PY) -m pip install -r requirements.txt

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m black .

format-check:
	$(PY) -m black --check .

mypy:
	$(PY) -m mypy .

test:
	SENTINEL_DISABLE_TOXICITY=1 $(PY) -m pytest

demo:
	$(PY) auditor.py --demo

dashboard:
	streamlit run dashboard.py

api:
	uvicorn api:app --reload
