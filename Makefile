.PHONY: install train infer compare test clean

install:
	pip install -e .
	pip install -r requirements.txt

train:
	python scripts/train_lora.py --config config.yaml

infer:
	python scripts/infer.py --prompt "$(Q)"

compare:
	python scripts/compare.py

test:
	pytest -q

clean:
	rm -rf outputs __pycache__ src/finetune/__pycache__ .pytest_cache
