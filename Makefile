init:
	poetry install

test:
	poetry run pytest ./processing/test_main.py
