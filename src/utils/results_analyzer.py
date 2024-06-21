import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
import numpy as np
import os

class ComparativeAnalysis:
    def __init__(self, train_file: str, val_file: str) -> None:
        """
        Constructor para la clase ComparativeAnalysis

        :param train_file: Ruta del archivo CSV con los resultados de entrenamiento
        :param val_file: Ruta del archivo CSV con los resultados de validación
        """
        self.train_data = pd.read_csv(train_file)
        self.val_data = pd.read_csv(val_file)
        # Eliminar espacios en los nombres de las columnas
        self.train_data.columns = self.train_data.columns.str.strip()
        self.val_data.columns = self.val_data.columns.str.strip()
        # Renombrar columnas para alinearlas correctamente
        self.train_data.rename(columns={
            'epoch': 'epoch',
            'metrics/precision(B)': 'precision',
            'metrics/recall(B)': 'recall',
            'metrics/mAP50(B)': 'map_05',
            'metrics/mAP50-95(B)': 'map_05_95'
        }, inplace=True)
        self.val_data.rename(columns={
            'map_05': 'map_05',
            'map_05_95': 'map_05_95'
        }, inplace=True)
        self.val_data = self.val_data.drop(0, errors='ignore')
        self.val_data = self.val_data.sort_values(by='epoch')

    def plot_comparative_graphs(self) -> None:
        """
        Genera una serie de gráficas comparativas entre los datos de entrenamiento y validación
        """
        metrics = ['precision', 'recall', 'map_05', 'map_05_95']
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
        plt.savefig(f"data/results/week1/baseline_reducted_with_validation/{metric}_comparison.png")
        plt.show()

# Uso del código
train_file_path = 'data/results/week1/baseline_reducted_with_validation/results.csv'
val_file_path = 'data/results/week1/baseline_reducted_with_validation/validation_results.csv'

analysis = ComparativeAnalysis(train_file_path, val_file_path)
analysis.plot_comparative_graphs()
