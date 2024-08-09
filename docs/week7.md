# Progresión Semana 7: 5/08 - 9/08 

## Objetivos

- [ ] Ejecución del Algoritmo Genético en Picasso, análisis de resultados.
- [ ] Ajuste fino de hiperparámetros `momentum` y final learning rate `lrf`.
- [ ] Análisis comparativo de resultados.
- [ ] Entrenamiento del modelo con todas las clases.

## Resultados

### Algoritmo Genético

El Algoritmo Genético se ha podido ejecutar en el supercomputador Picasso. Para ello, se ha creado un entorno conda y se han instalado las dependencias del script que contiene la ejecución de este. Acto seguido, se ha enviado el siguiente trabajo:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=yolov8_tuning
#SBATCH --time=2-00:00:00
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint=dgx
#SBATCH --error=yolov8_tuning.%J.err
#SBATCH --output=yolov8_tuning.%J.out

# Activar el entorno Conda
conda activate yolov8_env

# Ejecutar el script de Python
python /mnt/home/users/tic_163_uma/mpascual/GA_Ultralytics/yolov8.py

# Desactivar el entorno
conda deactivate
```

### Ajuste fino momentum

![momentum_lower](../data/results/week7/finetuning_lower_momentum/results_plot.png)

|   Precision |   Recall |    mAP50 |   mAP50-95 |   Parameter |
|------------:|---------:|---------:|-----------:|------------:|
|    0.759264 | 0.225246 | 0.492959 |   0.344481 |        0.45 |
|    0.889484 | 0.171218 | 0.50142  |   0.367605 |        0.47 |
|    0.952451 | 0.165059 | 0.552674 |   0.412035 |        0.5  |
|    0.949717 | 0.155163 | 0.53838  |   0.397566 |        0.55 |
|    0.892981 | 0.152669 | 0.52778  |   0.389639 |        0.57 |
|    0.764502 | 0.271944 | 0.523313 |   0.389331 |        0.6  |
|    0.859672 | 0.231374 | 0.551428 |   0.42842  |        0.65 |
|    0.924703 | 0.221418 | 0.577646 |   0.442631 |        0.7  |
|    0.912033 | 0.192684 | 0.555481 |   0.4338   |        0.72 |
|    0.907615 | 0.177343 | 0.545272 |   0.420387 |        0.75 |
|    0.901547 | 0.170474 | 0.539411 |   0.417238 |        0.78 |

Como se puede observar, el ajuste fino del parámetro momentum ha mejorado el recall del modelo hasta 0.2719, con el valor `0.6`, por lo que se utilizará junto con `lr0=0.0001`.

### Ajuste fino lrf


### Resultado de Detección con todas las clases

Se ha generado un conjunto de datos que hace uso de todas las clases del conjunto de datos (aunque solo tenga el objetivo de detectar la lesión, no clasificar) y se ha comprobado el comportamiento de la red. El conjunto de datos tiene las siguientes características:

![distrib](../data/results/week7/results_all_labels/images_labels_nolesion_distribution.png)
![distrib2](../data/results/week7/results_all_labels/label_distribution_train.png)

Los resultados de este entrenamiento son los siguientes:

![results](../data/results/week7/results_all_labels/results.png)

Utilizando los hiperparámetros anteriormente ajustados, parece que el modelo tiene los mismos problemas que solo utilizando las clases más "detectables", y es que el recall disminuye a medida que las épocas aumentan, no mejora o se estabiliza. Es por esto que se tendrán que ajustar los hiperparámetros de manera independiente para el modelo entrenado con todo el conjunto de datos.