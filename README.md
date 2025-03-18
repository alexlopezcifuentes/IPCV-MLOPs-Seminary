# Instrucciones para Ejecutar el Código

## Requisitos Previos
- Docker instalado en tu sistema

## Verificación de Docker
Para comprobar que Docker está correctamente instalado, ejecuta:

```bash
docker run hello-world
```

Si la instalación es correcta, deberías ver la siguiente salida:

<details>
<summary>Ejemplo de Salida</summary>

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

</details>

## Ejecución del Proyecto

Para facilitar la ejecución, todos los comandos necesarios están incluidos en el archivo `Makefile`. Esto permite ejecutar comandos complejos mediante atajos simples.

### 1. Imagen de Docker

En este seminario utilizaremos una imagen de Docker preconfigurada, disponible en Docker Hub. La imagen se encuentra en: [https://hub.docker.com/repository/docker/alexlopezcifuentes/ipcv-mlops/general](https://hub.docker.com/repository/docker/alexlopezcifuentes/ipcv-mlops/general). Haremos esto debido a las limitaciones computacionales y de tiempo de las maquinas virtuales.

> **Nota**: Aunque no es necesario crear la imagen, el repositorio incluye el archivo `Dockerfile` y la instrucción correspondiente en el `Makefile` por si deseas examinarlos.

### 2. Creación del Contenedor

Para iniciar el trabajo, crea un contenedor Docker con la imagen proporcionada ejecutando:

```bash
make docker-run
```

Si la imagen no está disponible localmente, Docker la descargará automáticamente. Una vez completada la descarga, se creará el contenedor y accederás a él. Deberas observar una salida como:
<details>
<summary>Ejemplo de Salida</summary>

```text
(SeminarioIPCV) (base) ➜  Seminario MLOPs git:(cifar10_v1.0.0) ✗ make docker-run
# Run a Docker container from the image
# -it                         # Interactive mode with terminal
# --rm                        # Remove container when it exits
# --name mlops-seminary-container  # Name the running container
# -v "/home/alex/Personal/Seminario MLOPs:/app"           # Mount current directory to /app in container
# -p 5000:5000               # Map port 5000 of host to port 5000 of container
# alexlopezcifuentes/ipcv-mlops:latest      # Use the latest version of our image
docker run -it --rm --name mlops-seminary-container -v "/home/alex/Personal/Seminario MLOPs:/app" -p 5000:5000 alexlopezcifuentes/ipcv-mlops:latest
Unable to find image 'alexlopezcifuentes/ipcv-mlops:latest' locally
latest: Pulling from alexlopezcifuentes/ipcv-mlops
6e909acdb790: Pull complete 
a8053d65de8e: Pull complete 
806331b0d260: Pull complete 
ce054015c4fb: Pull complete 
105df2d2f12c: Pull complete 
207feb3a1b79: Pull complete 
769ed5670cb7: Pull complete 
66c3fd7d0fc0: Pull complete 
93c1da8775e1: Pull complete 
Digest: sha256:c5b04dc0e891e19224b442704d5d8a3d2280613017a01b4339875cc2d63313af
Status: Downloaded newer image for alexlopezcifuentes/ipcv-mlops:latest
root@3f3f51185b33:/app# 
```

</details>

### 3. Acceso al Contenedor

En este punto deberías tener una terminal dentro del contenedor Docker. Para el seminario, necesitarás una segunda terminal dentro del mismo contenedor. Para abrir esta segunda terminal, ejecuta en una nueva ventana de terminal:

```bash
make docker-enter-container
```

Ahora deberías tener dos terminales ejecutándose dentro del contenedor Docker, que llamaremos **Terminal 1** y **Terminal 2**.

### 4. Ejecutar MLFlow

En la **Terminal 1**, inicia la interfaz de MLFlow con:

```bash
make run-mlflow-ui
```

<details>
<summary>Ejemplo de Salida</summary>

```text
root@3f3f51185b33:/app# make run-mlflow-ui
# Run MLflow's tracking UI server
# --host 0.0.0.0              # Make server accessible on all network interfaces
# --port 5000                 # Set the port to 5000
mlflow ui --host 0.0.0.0 --port 5000
[2025-03-18 15:12:21 +0000] [57] [INFO] Starting gunicorn 23.0.0
[2025-03-18 15:12:21 +0000] [57] [INFO] Listening at: http://0.0.0.0:5000 (57)
[2025-03-18 15:12:21 +0000] [57] [INFO] Using worker: sync
[2025-03-18 15:12:21 +0000] [58] [INFO] Booting worker with pid: 58
[2025-03-18 15:12:21 +0000] [59] [INFO] Booting worker with pid: 59
[2025-03-18 15:12:21 +0000] [60] [INFO] Booting worker with pid: 60
[2025-03-18 15:12:21 +0000] [61] [INFO] Booting worker with pid: 61
```
</details>

Esto iniciará el servidor local de MLFlow. Podrás acceder a la interfaz web en [http://0.0.0.0:5000/](http://0.0.0.0:5000/). Después de iniciar MLFlow, puedes minimizar la **Terminal 1**, ya que no la utilizaremos más.

> **Nota**: Solo minimiza la terminal, no la cierres puesto que entonces no podras acceder a MLFlow.


### 5. Ejecutar Entrenamiento

A partir de este punto, utilizaremos exclusivamente la **Terminal 2** para todas las operaciones.

#### 5.1. Descarga del Dataset

Utilizaremos DVC para gestionar y descargar el dataset de entrenamiento. Este repositorio contiene dos versiones etiquetadas del dataset:

- `cifar10v1.0.0`: Primera version de CIFAR10 sin imagenes de algunas clases en el set de entrenamiento:
- `cifar10v2.0.0`

Para descargar la versión 1.0.0, ejecuta desde la raíz del proyecto:

```bash
cd datasets
git checkout cifar10_v1.0.0
dvc pull cifar10
```

Durante la descarga deberemos de ver la siguiente salida:

<details>
<summary>Ejemplo de Salida</summary>

```text
root@3f3f51185b33:/app/datasets# dvc pull cifar10.dvc 
Collecting                                                                                                                            |60.0k [00:02, 26.9kentry/s]
Fetching
Building workspace index                                                                                                                |1.00 [00:00,  935entry/s]
Comparing indexes                                                                                                                      |60.0k [00:00, 129kentry/s]
Applying changes                                                                                                                       |60.0k [00:17, 3.48kfile/s]
A       cifar10/
1 file added
```
</details>


Una vez completada la descarga, deberías tener la siguiente estructura de archivos:

```
datasets/
└── cifar10/
    ├── train/         # Directorio con las imágenes de entrenamiento
    ├── train.txt      # Archivo de metadatos para el entrenamiento
    ├── val/           # Directorio con las imágenes de validación
    └── val.txt        # Archivo de metadatos para la validación
```

#### 5.2. Entrenamiento del Modelo

Para iniciar el entrenamiento, ejecuta:

```bash
python main.py
```

Verás cómo comienza el proceso de entrenamiento mostrando la siguiente salida:

<details>
<summary>Ejemplo de Salida</summary>

```text
root@3f3f51185b33:/app# python main.py 
2025-03-18 15:16:08.219 | INFO     | src.dataloader:__init__:13 - Initiating Dataloader for stage: train
2025-03-18 15:16:08.219 | INFO     | src.dataloader:set_dataset:29 - Using cifar10 dataset
2025-03-18 15:16:08.357 | INFO     | src.dataloader:__init__:13 - Initiating Dataloader for stage: val
2025-03-18 15:16:08.357 | INFO     | src.dataloader:set_dataset:29 - Using cifar10 dataset
2025-03-18 15:16:08.439 | INFO     | src.model:__init__:84 - Instantiating Model
2025-03-18 15:16:08.439 | INFO     | src.model:__init__:94 - Device to be used: cpu
2025-03-18 15:16:08.440 | INFO     | src.model:set_model:112 - Initiating Model: alexnet
2025-03-18 15:16:08.441 | INFO     | src.loss:__init__:24 - Instantiating Loss
2025-03-18 15:16:08.441 | INFO     | src.loss:__init__:39 - Setting ignore_index to 11
2025-03-18 15:16:08.441 | INFO     | src.loss:set_loss:47 - Initiating loss: crossentropy
2025-03-18 15:16:08.441 | INFO     | src.optimizer:set_optimizer:61 - Initiating optimizer: sgd
2025-03-18 15:16:08.442 | INFO     | src.optimizer:set_scheduler:69 - Initiating scheduler: step
2025-03-18 15:16:08.442 | INFO     | src.runners:__init__:25 - Instantiating Runner
2025-03-18 15:16:08.443 | INFO     | src.runners:complete_train:48 - Starting complete training...
2025-03-18 15:16:08.443 | INFO     | src.runners:complete_train:51 - Epoch 1 of 5
2025-03-18 15:16:08.443 | INFO     | src.runners:complete_train:54 - Training step...
2025-03-18 15:16:09.065 | INFO     | src.runners:epoch_train:119 - [Epoch 1/5,     1/469]. Loss: 2.300. Accuracy: 0.125
2025-03-18 15:16:10.790 | INFO     | src.runners:epoch_train:119 - [Epoch 1/5,   101/469]. Loss: 2.034. Accuracy: 0.243
```
</details>

En MLFlow se creará automáticamente un experimento con la información de esta ejecución, que servirá como baseline para comparaciones posteriores.


#### 5.3. Actualización del Dataset

Para probar con la versión 2.0.0 del dataset, ejecuta:

```bash
cd datasets
git checkout cifar10_v2.0.0
dvc pull cifar10
```

Una vez descargada la nueva versión, puedes iniciar un nuevo entrenamiento con:

```bash
python main.py
```

Esto generará una nueva entrada en MLFlow que podrás comparar con el entrenamiento anterior.