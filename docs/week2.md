# Progresión Semana 2: 1/07 - 6/07

## Objetivos de la semana

1. Convertir el problema de detección/clasificación de YOLOv8 en un problema de detección únicamente, extrayendo las bounding boxes detectadas a un formato de tensor para poder delimitar la sección que será procesada posteriormente por una red de clasificación. Se propone almacenar en un fichero `csv` los datos deferentes a la etiqueta de la imagen, que irán asociados a la imagen original. 
2. Realizar una aumentación de datos para el problema de detección únicamente, con la finalidad de mejorar la precisión del modelo. Esta aumentación de datos deberá ser aplicada tanto a las imágenes sin lesión como a las de con lesión. Tendrá las siguientes características:
    - Se aumentará un % determinado a datos con lesión y sin lesión, de tal forma que la aumentación de datos esté presente en ambas, pero con una mayor presencia en las imágenes con lesión. 
    - Dentro de las imágenes con lesión, aunque el problema sea méramente de detección, aplicarán de manera indexada las aumentaciones a las diferentes clases, de la forma que la clase con menor instancias sea la más aumentada -asegurando la aumentación en todas. 
    - Las transformaciones serán:
      - Variación ligera del brillo.
      - Variación ligera del contraste.
      - Rotación **muy ligera** de la imagen.
      - Translación **muy ligera** de la imagen. 
3. Investigar sobre redes convolucionales de clasificación o backbones que puedan servir con este propósito. (No diseñarla hasta que el objetivo 2 haya hecho que YOLO mejore su precisión, recall, y mAP). 
4. Documentar el proceso con la finalidad de detectar errores de diseño. 

## Aumentación de datos

- Se ha introducido un factor 2 a 1 la cantidad de imágenes con lesión, para poder tener una mejor representación de la clase minoritaria. 
- Se ha aumentado el brillo y contraste de las imágenes con lesión. Generando nuevos nombres. 
- Una misma imagen puede ser aumentada por veces por el mismo método. Por ello, se ha añadido un número secuencial al nombre. 

Cosas a hacer:
- Indexar la cantidad de instancias de cada clase en lesión para favorecer al label minoritario.
- Introducir nuevos tipos de aumentación espacial. 