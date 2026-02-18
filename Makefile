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
	$(PY) auditor.py --input-jsonl data/golden_path.jsonl --project golden-path --tags demo,golden

dashboard:
	streamlit run dashboard.py

api:
	uvicorn api:app --reload

eval:
	$(PY) scripts/evaluate.py --dataset eval/labeled.jsonl --output-json eval/current_metrics.json

eval-gate:
	$(PY) scripts/check_eval_regression.py --baseline eval/baseline_metrics.json --current eval/current_metrics.json

migrate:
	$(PY) -m alembic upgrade head

up:
	docker compose up --build

web-install:
	cd web && npm install

web-dev:
	cd web && npm run dev

web-build:
	cd web && npm run build
