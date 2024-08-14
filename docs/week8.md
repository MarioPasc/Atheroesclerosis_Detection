# Progresión Semana 8: 12/07 - 16/07

## Objetivos de la Semana

- [X] Ejecución del Algoritmo Genético con nueva función de `fitness` para la mejora del recall
- [ ] Ajuste de número de épocas e iteraciones algoritmo genético

## Resultados GA

Como se comentó en la semana anterior, se ha modificado la función de fitness de los individuos al medio, dándole más peso a la métrica recall dentro de la media ponderada devuelta:

```python
    def fitness(self):
        """Model fitness as a weighted combination of metrics."""
        w = [0.0, 0.9, 0.0, 0.1]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return (np.array(self.mean_results()) * w).sum()
```

Para la ejecución del GA con esta nueva función de fitness en el SC Picasso se ha partido de la configuración óptima para el recall conseguida en el entrenamiento con la función de fitness por defecto, sin embargo, para que los resultados pudieran ser más fiables, pero que el tiempo de ejecución no supere los 3 días, se decidió dedicar `200` iteraciones a `50` épocas por iteración. Los resultados son los siguientes:

![data1](../data/results/week8/GA_Recall_config/detect/tune/mAP50-95_evolution.png)
![data2](../data/results/week8/GA_Recall_config/detect/tune/mAP50_evolution.png)
![data3](../data/results/week8/GA_Recall_config/detect/tune/precision_evolution.png)
![data4](../data/results/week8/GA_Recall_config/detect/tune/recall_evolution.png)

Como se puede observar, el recall ofrece una mejora sustancial, llegando a alcanzar un valor de `0.69` en la época 50 de ejecución. Este valor elevado, y que, si se observa la nube de puntos, parece ser un outlier, será tomado con precaución, ya que puede ser el producto de una configuración de pesos y sesgos afortunada en la red, es por esto que se ha probado a ejecutar por 120 épocas la configuración de hiperparámetros obtenida en esta época, con el fin de investigar si presenta una mejora tan optimista como la presentada en estos resultados. 

#### Resultados ejecución 120 épocas config83

![data5](../data/results/week8/GA_Recall_config/Recall_config_GA/recall_comparison.png)
![data5](../data/results/week8/GA_Recall_config/Recall_config_GA/precision_comparison.png)
![data5](../data/results/week8/GA_Recall_config/Recall_config_GA/map_50_comparison.png)
![data5](../data/results/week8/GA_Recall_config/Recall_config_GA/map_50_95_comparison.png)

Como se puede observar, los valores de recall presentan una mejora con respecto a los entrenamientos de hace unas semanas, presentando una estabilización de la métrica en entrenamiento al rededor de 0.5, y, en validación, al rededor de 0.33. Esto puede indicar que, efectivamente, el valor de 0.69 de recall obtenido mediante el GA podría deberse a una combinación afortunada de pesos y sesgos.

Para **evitar** que esto pueda pasar, o prevenirlo en la medida de lo posible, se va a realizar una ejecución del algoritmo genético partido de los valores de recall de la configuración 83 pero aumentando el número de épocas hasta `100` -y disminuyendo el número de iteraciones de manera consecuente para no aumentar demasiado el tiempo de ejecución. De esta forma se podrá mitigar el efecto aleatorio de los pesos y sesgos de la red, aumentando el número de épocas y dejando suficiente tiempo para que las métricas de estabilicen.  