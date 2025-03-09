# Instrucciones para Ejecutar el Código

## Requisitos Previos
- Docker Instalado

## Verificación de Docker
Comprobación de que Docker está instalado:
  ```bash
  docker run hello-world
  ```

Deberíamos de ver la siguiente salida:
```text
Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/
```

## Ejecución del Proyecto

Una vez tenemos Docker funcionando deberemos de ejecutar una serie de comandos. Para facilitar el uso de los mismos, todos ellos se encuentran creados en el fichero Makefile. Esto nos permite no tener que conocer la sintaxis completa del comando sino solo su shortcut.

### 1. Crear la imagen de Docker:
Debemos de crear la imagen de Docker del seminario:
```bash
make docker-build
```

### 2. Prerequisitos antes de Entrenar
Necesitamos que el dataset se encuentre en la ruta `datasets`.
Deberiamos de tener los siguientes ficheros y carpetas:

```
datasets/
└── cifar10/
    ├── train/         # Directorio con las imágenes de entrenamiento
    ├── train.txt      # Archivo de metadatos para el entrenamiento
    ├── val/           # Directorio con las imágenes de validación
    └── val.txt        # Archivo de metadatos para la validación
```

Asegúrate de que la estructura del dataset coincida exactamente con la mostrada arriba para que el código funcione correctamente.

### 3. Ejecutar el contenedor
Una vez tenemos la imagen creada, crearemos el contenedor ejecutando en una terminal:
```bash
make docker-run
```
Esto creará el contenedor y nos introducirá dentro del mismo.

### 4. Acceder al contenedor
Dado que necesitamos otra terminal con el contenedor, podremos ejecutar:
```bash
make docker-enter-container
```

En este punto deberíamos de tener dos terminales corriendo dentro del contenedor de Docker.

### 5. Ejecutar MLFlow y Entrenamiento
En una de las terminales haremos:
```bash
make run-mlflow-ui
```
Esto nos permitirá levantar el servidor local de MLFlow. En este punto podremos acceder a la URL http://0.0.0.0:5000/ para visualizar MLFlow.

En la otra terminal ejecutaremos:
```bash
python main.py
```

En este punto deberíamos ver que el código empieza a entrenar.

Una vez estemos entrenando el modelo deberemos ver que se crea un experimento en MLFlow y que nos sale el run de entrenamiento.