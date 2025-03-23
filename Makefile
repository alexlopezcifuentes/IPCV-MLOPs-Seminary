docker-build:
	# Build a Docker image
	# -t mlops-seminary:latest    # Name and tag the image as 'mlops-seminary:latest'
	# .                           # Use Dockerfile in current directory
	docker build -t alexlopezcifuentes/ipcv-mlops:latest .

docker-pull-image:
	docker pull alexlopezcifuentes/ipcv-mlops:latest

docker-run:
	# Run a Docker container from the image
	# -it                         # Interactive mode with terminal
	# --rm                        # Remove container when it exits
	# --name mlops-seminary-container  # Name the running container
	# -v "$(PWD):/app"           # Mount current directory to /app in container
	# -p 5000:5000               # Map port 5000 of host to port 5000 of container
	# alexlopezcifuentes/ipcv-mlops:latest      # Use the latest version of our image
	docker run -it --rm --name mlops-seminary-container -v "$(PWD):/app" -p 5000:5000 alexlopezcifuentes/ipcv-mlops:latest

docker-enter-container:
	# Execute an interactive bash shell in a running container
	# -it                         # Interactive mode with terminal
	# mlops-seminary-container    # Target container name
	# bash                        # Command to run (bash shell)
	docker exec -it mlops-seminary-container bash

run-mlflow-ui:
	# Run MLflow's tracking UI server
	# --host 0.0.0.0              # Make server accessible on all network interfaces
	# --port 5000                 # Set the port to 5000
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