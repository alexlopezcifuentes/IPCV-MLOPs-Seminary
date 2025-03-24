#!/bin/bash

# Script para cambiar el propietario de todos los archivos en el directorio actual
# Creado: $(date)

# Verificar si se está ejecutando como root o con sudo
if [ "$EUID" -ne 0 ]; then
  echo "Este script debe ejecutarse con privilegios de administrador (sudo)"
  echo "Uso: sudo ./change_owner.sh [usuario]"
  exit 1
fi

# Establecer el usuario al que cambiar la propiedad
USER=${1:-alex}

echo "Cambiando el propietario de todos los archivos en $(pwd) al usuario $USER..."

# Cambiar propietario de todos los archivos y directorios
chown -R $USER .

echo "¡Cambio de propietario completado!"
echo "Se ha cambiado el propietario de todos los archivos a $USER"