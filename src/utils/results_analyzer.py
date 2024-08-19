import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import numpy as np
import os

class ComparativeAnalysis:
    def __init__(self, path_to_results: str) -> None:
        """
        Constructor para la clase ComparativeAnalysis

        :param path_to_results: Ruta de la carpeta con los archivos CSV de resultados
        """
        self.path_to_results = path_to_results
        self.train_data = pd.read_csv(os.path.join(self.path_to_results, 'results.csv'))
        self.val_data = pd.read_csv(os.path.join(self.path_to_results, 'validation_results.csv'))
        # Eliminar espacios en los nombres de las columnas
        self.train_data.columns = self.train_data.columns.str.strip()
        self.val_data.columns = self.val_data.columns.str.strip()
        # Renombrar columnas para alinearlas correctamente
        self.train_data.rename(columns={
            'epoch': 'epoch',
            'metrics/precision(B)': 'precision',
            'metrics/recall(B)': 'recall',
            'metrics/mAP50(B)': 'map_50',
            'metrics/mAP50-95(B)': 'map_50_95'
        }, inplace=True)
        self.val_data.rename(columns={
            'map_05': 'map_50',
            'map_05_95': 'map_50_95'
        }, inplace=True)
        self.val_data = self.val_data.drop(0, errors='ignore')
        self.val_data = self.val_data.sort_values(by='epoch')

    def plot_comparative_graphs(self) -> None:
        """
        Genera una serie de gráficas comparativas entre los datos de entrenamiento y validación
        """
        metrics = ['precision', 'recall', 'map_50', 'map_50_95']
        self._plot_violin(metrics)
        self._save_stats(metrics)
        for metric in metrics:
            self._plot_metric(metric)
    
    def _plot_metric(self, metric: str) -> None:
        """
        Genera una gráfica para la métrica especificada

        :param metric: Nombre de la métrica a graficar (e.g., 'accuracy', 'recall', 'map')
        """

        plt.figure(figsize=(10, 5))
        plt.plot(self.train_data['epoch'], self.train_data[metric], label=f'Train {metric.capitalize()}', marker='o')
        plt.plot(self.val_data['epoch'], self.val_data[metric], label=f'Validation {metric.capitalize()}', marker='x')
        plt.title(f'{metric.capitalize()} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.yticks(np.linspace(0, 1, 5))
        plt.grid(True)
    
        
        plt.savefig(f"{self.path_to_results}/{metric}_comparison.png")
        plt.show()

    def _plot_violin(self, metrics: List[str]) -> None:
        """
        Genera un gráfico de violín para las métricas especificadas

        :param metrics: Lista de nombres de métricas a graficar
        """
        # Crear DataFrame para violín plot
        violin_data = pd.DataFrame()

        for metric in metrics:
            train_series = pd.DataFrame({
                'value': self.train_data[metric],
                'metric': f'Train {metric}'
            })
            val_series = pd.DataFrame({
                'value': self.val_data[metric],
                'metric': f'Val {metric}'
            })
            violin_data = pd.concat([violin_data, train_series, val_series])

        plt.figure(figsize=(15, 7))
        sns.violinplot(x='metric', y='value', data=violin_data, inner=None)

        # Añadir puntos estadísticos
        for metric in metrics:
            train_values = self.train_data[metric]
            val_values = self.val_data[metric]

            train_stats = {
                'mean': train_values.mean(),
                'median': train_values.median(),
                'q1': train_values.quantile(0.25),
                'q3': train_values.quantile(0.75),
                'min': train_values.min(),
                'max': train_values.max(),
                'std': train_values.std()
            }
            val_stats = {
                'mean': val_values.mean(),
                'median': val_values.median(),
                'q1': val_values.quantile(0.25),
                'q3': val_values.quantile(0.75),
                'min': val_values.min(),
                'max': val_values.max(),
                'std': val_values.std()
            }

            for stat, color, marker in zip(['mean', 'median', 'q1', 'q3', 'min', 'max', 'std'],
                                           ['red', 'blue', 'green', 'green', 'black', 'black', 'orange'],
                                           ['o', 'x', 'd', 'd', 's', 's', '^']):
                plt.scatter(x=[f'Train {metric}'], y=[train_stats[stat]], color=color, s=20, label=f'{stat}' if stat not in plt.gca().get_legend_handles_labels()[1] else "")
                plt.scatter(x=[f'Val {metric}'], y=[val_stats[stat]], color=color, s=20, label=f'{stat}' if stat not in plt.gca().get_legend_handles_labels()[1] else "")

        plt.title('Metric Distributions with Statistical Points')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.path_to_results}/metrics_violin_plot.png")
        plt.show()

    def _save_stats(self, metrics: List[str]) -> None:
        """
        Guarda las estadísticas calculadas en un archivo .txt en formato Markdown

        :param metrics: Lista de nombres de métricas a guardar
        """
        stats_data = []
        for metric in metrics:
            train_values = self.train_data[metric]
            val_values = self.val_data[metric]

            train_stats = {
                'Metric': f'Train {metric}',
                'Mean': train_values.mean(),
                'Median': train_values.median(),
                'Q1': train_values.quantile(0.25),
                'Q3': train_values.quantile(0.75),
                'Min': train_values.min(),
                'Max': train_values.max(),
                'Std': train_values.std()
            }
            val_stats = {
                'Metric': f'Val {metric}',
                'Mean': val_values.mean(),
                'Median': val_values.median(),
                'Q1': val_values.quantile(0.25),
                'Q3': val_values.quantile(0.75),
                'Min': val_values.min(),
                'Max': val_values.max(),
                'Std': val_values.std()
            }
            stats_data.extend([train_stats, val_stats])

        stats_df = pd.DataFrame(stats_data)
        stats_df = stats_df.round(4)

        # Guardar como .txt en formato Markdown
        with open(f"{self.path_to_results}/metrics_statistics.md", 'w') as f:
            f.write(stats_df.to_markdown(index=False))

# Ejemplo de uso
analysis = ComparativeAnalysis(path_to_results='data/results/week9/GA_results_trainval')
analysis.plot_comparative_graphs()
