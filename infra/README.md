# Infrastructure (`infra/`)

This folder contains AWS infrastructure code (Terraform) and helper utilities for the seminar.

## Files

- `main.tf`: defines the infrastructure (EC2 instances) and exposes the `ips_alumnos` output.
- `ips_config.py`: reads `ips_alumnos` from Terraform and generates SSH config blocks for students.
- `mlflow_daemon.txt`: example `systemd` unit to run the MLflow Tracking Server as a service.
- `.gitignore`: excludes Terraform state and sensitive/generated files.
- `.terraform.lock.hcl`: Terraform provider lock file (reproducible versions).


## Quick run

From the project root:

```bash
cd infra
terraform init
terraform plan
terraform apply
```

To generate SSH config from the created IPs:

```bash
cd infra
python3 ips_config.py
```

## MLflow as a service (optional)

1. Copy `mlflow_daemon.txt` to `/etc/systemd/system/mlflow.service`.
2. Reload and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl start mlflow
```
