.PHONY: audit behavior-audit init corpus-check test study

PYTHON := python3
DB ?= tsq.db
LEARNER ?= local
TOPIC ?= c_ai_learning_systems

audit:
	PYTHONPATH=src $(PYTHON) -m tsq audit corpus/ai_curriculum.json
	$(MAKE) behavior-audit

behavior-audit:
	PYTHONPATH=src $(PYTHON) scripts/run_behavioral_audit.py --root c_ai_learning_systems --profile strong --trials 5 --steps 24 --summary-only
	PYTHONPATH=src $(PYTHON) scripts/run_behavioral_audit.py --root c_ai_learning_systems --profile always-wrong --trials 5 --steps 24 --summary-only

init:
	PYTHONPATH=src $(PYTHON) -m tsq --db $(DB) init

corpus-check:
	$(PYTHON) scripts/sync_bundled_corpus.py

test:
	PYTHONPATH=src $(PYTHON) -W error::ResourceWarning -m unittest discover -s tests -v

study:
	PYTHONPATH=src $(PYTHON) -m tsq --db $(DB) study --learner $(LEARNER) --topic $(TOPIC) --explain-policy
