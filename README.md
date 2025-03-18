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


### 1. Imagen de Docker
Para comenzar a trabajar, lo primero que deberiamos hacer es crear nuestra imagen de Docker. Sin embargo, y por limitaciones de tiempo, en este seminario se da la imagen de docker pre construida. Esto quiere decir que podemos usar esta imagen de Docker de ahora en adelante para crear nuestros contenedores.

La imagen de Docker preconstruida que vamos a usar esta alojada en el registry Docker Hub en la siguiente URL: https://hub.docker.com/repository/docker/alexlopezcifuentes/ipcv-mlops/general. Podemos ver como el comando make docker-run hace uso de la imagen de Docker alexlopezcifuentes/ipcv-mlops:latest.

Pese a no ser necesaria la creacion de la imagen de Docker el fichero Makefile contiene la instrucción para crear la imagen y se facilita el fichero Dockerfile.

### 2. Creacion del contenedor
Para empezar a trabajar deberemos de crear un contenedor de Docker usando la imagen proporcionada. Lo haremos con el comando: 
```bash
make docker-run
```
Veremos como, al no tener la iamgen disponible localmente, se comienza a descargar. Una vez concluido el proceso de descarga se creará el contenedor y se nos introducirá dentro del mismo.

### 3. Acceder al contenedor
En este punto deberiamos de tener una terminal dentro del contenedor de Docker. Sin embargo, para la realizacion del seminario necesitamos una segunda terminal dentro del contenedor. Para ello podremos ejecutar, desde otra terminal distinta, el siguiente comando:
```bash
make docker-enter-container
```
En este punto deberíamos de tener dos terminales corriendo dentro del contenedor de Docker, las denominaremos Terminal 1 y Terminal 2.

### 5. Ejecutar MLFlow.
En la Terminal 1 vamos a ejecutar la interfaz de MLFLow. Para ello haremos el siguiente comando:
```bash
make run-mlflow-ui
```
Esto nos permitirá levantar el servidor local de MLFlow. En este punto podremos acceder a la URL http://0.0.0.0:5000/ para visualizar MLFlow. A partir de este momento podremos minimizar la Terminal 1 puesto que no la volveremos a utilizar.

### 6. Ejecutar Entrenamiento

En esta sección se explican los detalles para ejecutar el entrenamiento. El objetivo de este seminario es hacer modificaciones sobre la version del dataset y algun hyperparametro del entrenamineto para comprobar las utilidades tanto de DVC como de MLFlow.

A partir de este punto usaremos en todo momento Terminal 2.

#### 1. Descarga del dataset
Vamos a utilizar DVC para descargar el dataset de entrenamiento. Uno de los puntos de usar DVC para la gestion de los datos es que nos permite utilizar Git como versionador de elementos. De este modo, en este repositorio podremos encontrar dos tags de git https://github.com/alexlopezcifuentes/IPCV-MLOPs-Seminary/tags

- cifar10v1.0.0
- cifar10v2.0.0

Estos dos tags en Git nos permiten acceder con facilidad a cada una de las versiones. Vamos a descargar la v1.0.0. Para ello ejecutaremos los sigueintes comandos desde la raiz del proyecto:
```bash
cd datasets
git checkout cifar10v1.0.0
dvc pull cifar10
```
En este punto comenzaremos a ver como se descarga el dataset. Una vez concluido deberiamos de tener los siguientes ficheros y carpetas:

```
datasets/
└── cifar10/
    ├── train/         # Directorio con las imágenes de entrenamiento
    ├── train.txt      # Archivo de metadatos para el entrenamiento
    ├── val/           # Directorio con las imágenes de validación
    └── val.txt        # Archivo de metadatos para la validación
```

Ahora podremos ejecutar el entrenamiento utilizando el comando:
```bash
python main.py
```

En este punto deberíamos ver que el código empieza a entrenar.

Una vez estemos entrenando el modelo deberemos ver que se crea un experimento en MLFlow y que nos sale el run de entrenamiento.