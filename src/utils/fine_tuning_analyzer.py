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
        self.metrics = {
            'metrics/precision(B)': 'Precision',
            'metrics/recall(B)': 'Recall',
            'metrics/mAP50(B)': 'mAP50',
            'metrics/mAP50-95(B)': 'mAP50-95'
        }

    def plot_results(self):
        """
        Genera subplots para cada métrica especificada mostrando la evolución a lo largo de las épocas para diferentes valores del parámetro.
        """
        folders = [f for f in os.listdir(self.results_folder) if os.path.isdir(os.path.join(self.results_folder, f))]
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.ravel()  # Aplanar la matriz de ejes para fácil iteración
        
        for idx, (metric, metric_name) in enumerate(self.metrics.items()):
            for folder in folders:
                csv_path = os.path.join(self.results_folder, folder, 'results.csv')
                if os.path.exists(csv_path):
                    this_results = pd.read_csv(csv_path)
                    this_results.columns = this_results.columns.str.strip()
                    if metric in this_results.columns:
                        label = folder.split('_')[-1]
                        axs[idx].plot(this_results['epoch'], this_results[metric], label=label)
            axs[idx].set_title(metric_name)
            axs[idx].set_xlabel('Epoch')
            axs[idx].set_ylabel('Value')
            axs[idx].legend(loc='lower right')
            axs[idx].grid(True)

        plt.tight_layout()
        output_path = os.path.join(self.results_folder, 'results_plot.png')
        plt.savefig(output_path)
        plt.show()

    def save_metrics_summary(self):
        """
        Guarda un resumen de las métricas en un archivo de texto en formato markdown.
        """
        folders = [f for f in os.listdir(self.results_folder) if os.path.isdir(os.path.join(self.results_folder, f))]
        summary_data = []

        for folder in folders:
            csv_path = os.path.join(self.results_folder, folder, 'results.csv')
            if os.path.exists(csv_path):
                this_results = pd.read_csv(csv_path)
                this_results.columns = this_results.columns.str.strip()
                averages = {}
                for metric, metric_name in self.metrics.items():
                    if metric in this_results.columns:
                        averages[metric_name] = this_results[metric].mean()
                averages['Parameter'] = folder.split('_')[-1]
                summary_data.append(averages)

        summary_df = pd.DataFrame(summary_data)
        markdown_table = summary_df.to_markdown(index=False)

        with open(os.path.join(self.results_folder, 'metrics_summary.md'), 'w') as f:
            f.write("# Metrics Summary\n")
            f.write(markdown_table)

def main():
    results_folder = "./data/results/week7/finetuning_lrf"
    analyzer = FineTuningAnalyzer(results_folder=results_folder)
    analyzer.plot_results()
    analyzer.save_metrics_summary()

if __name__ == "__main__":
    main()
