.PHONY: audit behavior-audit init corpus-check start study test

PYTHON := python3
DB ?= tsq.db
LEARNER ?= local
TOPIC ?= Large Language Models

audit:
	PYTHONPATH=src $(PYTHON) -m tsq audit corpus/ai_curriculum.json --strict
	$(MAKE) behavior-audit

behavior-audit:
	PYTHONPATH=src $(PYTHON) scripts/run_behavioral_audit.py --root t_machine_learning --profile strong --trials 5 --steps 24 --summary-only
	PYTHONPATH=src $(PYTHON) scripts/run_behavioral_audit.py --root t_machine_learning --profile always-wrong --trials 5 --steps 24 --summary-only

init:
	PYTHONPATH=src $(PYTHON) -m tsq --db $(DB) init

start:
	./start

corpus-check:
	$(PYTHON) scripts/sync_bundled_corpus.py

test:
	PYTHONPATH=src $(PYTHON) -W error::ResourceWarning -m unittest discover -s tests -v

study:
	PYTHONPATH=src $(PYTHON) -m tsq --db $(DB) study --learner $(LEARNER) --topic "$(TOPIC)" --explain-policy
