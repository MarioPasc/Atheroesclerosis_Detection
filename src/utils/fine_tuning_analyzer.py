import os
import pandas as pd
import matplotlib.pyplot as plt

class FineTuningAnalyzer:

    def __init__(self, results_folder: str) -> None:
        """
        Inicializa la clase FineTuningAnalyzer con la carpeta de resultados.
        
        :param results_folder: Ruta a la carpeta que contiene los subdirectorios con los archivos results.csv
        """
        self.results_folder = results_folder
        self.metrics = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']

    def plot_results(self):
        """
        Genera subplots para cada métrica especificada mostrando la evolución a lo largo de las épocas para diferentes valores del parámetro.
        """
        folders = [f for f in os.listdir(self.results_folder) if os.path.isdir(os.path.join(self.results_folder, f))]
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.ravel()  # Aplanar la matriz de ejes para fácil iteración
        
        for idx, metric in enumerate(self.metrics):
            for folder in folders:
                csv_path = os.path.join(self.results_folder, folder, 'results.csv')
                if os.path.exists(csv_path):
                    this_results = pd.read_csv(csv_path)
                    this_results.columns = this_results.columns.str.strip()
                    if metric in this_results.columns:
                        label = folder.split('_')[-1]
                        
                        axs[idx].plot(this_results['epoch'], this_results[metric], label=label)
            axs[idx].set_title(metric)
            axs[idx].set_xlabel('Epoch')
            axs[idx].set_ylabel('Value')
            axs[idx].legend(loc='lower right')
            axs[idx].grid(True)

        plt.tight_layout()
        output_path = os.path.join(self.results_folder, 'results_plot.png')
        plt.savefig(output_path)
        plt.show()

def main():
    results_folder = "./data/results/week6/fine_tuning_lr0"
    analyzer = FineTuningAnalyzer(results_folder=results_folder)
    analyzer.plot_results()

if __name__ == "__main__":
    main()
