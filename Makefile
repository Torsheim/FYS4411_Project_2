.PHONY: install test part-b part-c part-d part-e sweep figures tables reproduce clean

install:
	pip install -e .

test:
	pytest -q

part-b:
	python scripts/run_experiment.py --config configs/part_b_1p1d_2h_bruteforce.yaml

part-c:
	python scripts/run_experiment.py --config configs/part_c_1p1d_2h_importance.yaml

part-d:
	python scripts/run_blocking.py --config configs/part_d_blocking_importance.yaml

part-e:
	python scripts/run_experiment.py --config configs/part_e_2p2d_interacting.yaml

sweep:
	python scripts/run_sweep.py --config configs/sweep_learning_rate.yaml
	python scripts/run_sweep.py --config configs/sweep_hidden_units.yaml

figures:
	python scripts/make_figures.py

tables:
	python scripts/make_tables.py

reproduce:
	python scripts/reproduce_selected_results.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
