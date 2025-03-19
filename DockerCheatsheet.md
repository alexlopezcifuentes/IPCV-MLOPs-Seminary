# Docker Cheatsheet

Esta guía contiene los comandos más útiles de Docker organizados por categorías.

## Gestión de Imágenes

### Listar imágenes

```bash
docker images
```
Muestra todas las imágenes Docker descargadas en tu sistema, incluyendo el nombre, etiqueta, ID, fecha de creación y tamaño.

```bash
docker image ls
```
Funcionalidad idéntica a `docker images`, pero usa la sintaxis más nueva de comandos.

### Buscar y descargar imágenes

```bash
docker search nombre_imagen
```
Busca imágenes disponibles en Docker Hub según el nombre proporcionado.

```bash
docker pull nombre_imagen:tag
```
Descarga una imagen específica desde Docker Hub, usando opcionalmente una etiqueta (tag) concreta (como "latest", "3.9", etc.).

### Eliminar imágenes

```bash
docker rmi nombre_imagen:tag
```
Elimina una imagen específica del sistema local. Si hay contenedores usando esta imagen, primero debes eliminar esos contenedores.

## Gestión de Contenedores

### Listar contenedores

```bash
docker ps
```
Muestra solo los contenedores que están actualmente en ejecución.

```bash
docker ps -a
```
Muestra todos los contenedores, tanto los que están ejecutándose como los detenidos.

### Operaciones básicas con contenedores

```bash
docker start id_contenedor
```
Inicia un contenedor previamente detenido, usando su ID o nombre.

```bash
docker stop id_contenedor
```
Detiene un contenedor en ejecución de forma ordenada, permitiendo que finalice sus procesos.

```bash
docker rm id_contenedor
```
Elimina permanentemente un contenedor. El contenedor debe estar detenido previamente.

### Crear y ejecutar contenedores

```bash
docker run [opciones] nombre_imagen:tag
```
Crea y arranca un nuevo contenedor basado en la imagen especificada, con opciones opcionales.

```bash
docker run -d nombre_imagen
```
Ejecuta el contenedor en segundo plano, devolviendo el control a la terminal.

```bash
docker run -p 8080:80 nombre_imagen
```
Conecta el puerto 8080 del host al puerto 80 del contenedor, permitiendo acceso externo.

```bash
docker run -v /ruta/local:/ruta/contenedor nombre_imagen
```
Vincula un directorio del host con uno del contenedor, permitiendo compartir datos entre ambos.

```bash
docker run --name mi_contenedor nombre_imagen
```
Crea un contenedor con un nombre específico en lugar de uno aleatorio.

```bash
docker run -it nombre_imagen bash
```
Ejecuta un contenedor con acceso interactivo a su terminal, útil para depuración o exploración.

### Monitoreo de contenedores

```bash
docker logs id_contenedor
```
Muestra los logs (salida estándar y de error) generados por un contenedor.

```bash
docker exec -it id_contenedor comando
```
Ejecuta un comando dentro de un contenedor que ya está en funcionamiento.

```bash
docker stats
```
Muestra estadísticas en tiempo real del uso de CPU, memoria y red de los contenedores en ejecución.

```bash
docker inspect id_contenedor
```
Proporciona información detallada sobre la configuración y estado de un contenedor específico.

### Otras operaciones con contenedores

```bash
docker commit id_contenedor nueva_imagen:tag
```
Crea una nueva imagen a partir del estado actual de un contenedor, útil para guardar cambios realizados.

## Limpieza y Mantenimiento

```bash
docker container prune
```
Elimina todos los contenedores detenidos, liberando espacio.

```bash
docker image prune
```
Elimina todas las imágenes que no están siendo utilizadas por ningún contenedor.

```bash
docker volume prune
```
Elimina todos los volúmenes que no están conectados a ningún contenedor.

```bash
docker system prune
```
Limpieza completa del sistema, eliminando todos los recursos que no se están utilizando actualmente.
