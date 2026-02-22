provider "aws" {
  region = "eu-west-3" # París
}

# 1. Definimos las instancias para los alumnos
resource "aws_instance" "alumnos" {
  count         = 1
  ami           = "ami-0f2c8f24be941f7e1" # PyTorch AMI
  instance_type = "t3.xlarge"

  key_name = "mlops_seminary_ssh"

  # Usamos el Rol que creaste para el acceso a S3/DVC
  iam_instance_profile = "RolSeminarioMLOPs"

  # Aquí conectamos con tu Security Group ya existente
  vpc_security_group_ids = ["sg-0f28560d53abf5f22"]

  # Disco duro de 50GB (Root Volume)
  root_block_device {
    volume_size           = 50
    volume_type           = "gp3"
    delete_on_termination = true
  }

  # Bootstrap: clone workshop repository on first boot
  user_data = <<-EOF
    #!/bin/bash
    set -euxo pipefail

    REPO_URL="https://github.com/alexlopezcifuentes/IPCV-MLOPs-Seminary.git"
    TARGET_DIR="/home/ubuntu/IPCV-MLOPs-Seminary"

    if [ ! -d "$${TARGET_DIR}/.git" ]; then
      git clone "$${REPO_URL}" "$${TARGET_DIR}"
      chown -R ubuntu:ubuntu "$${TARGET_DIR}"
    fi
  EOF

  tags = {
    Name = "MLOps-Alumno-${count.index + 1}"
  }
}

# 2. Resultado: Lista de IPs para repartir en clase
output "ips_alumnos" {
  value = aws_instance.alumnos[*].public_ip
}
