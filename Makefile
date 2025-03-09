docker-build:
	docker build -t mlops-seminary .

docker-run:
	docker run -it --rm --name mlops-seminary-container -v "$(PWD):/app" -p 5000:5000 mlops-seminary

docker-enter-container:
	docker exec -it mlops-seminary-container bash

run-mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000


# Quality Checks
flake8:
	flake8 .
isort-check:
	isort --check-only .
isort:
	isort .
black-check:
	black --check .
black:
	black .
quality-check: isort-check black-check flake8
quality: isort black flake8