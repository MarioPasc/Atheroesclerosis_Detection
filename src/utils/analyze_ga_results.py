import os
import pandas as pd
from natsort import natsorted
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

class AnalyzerGA:

    def __init__(self, detect_path:str) -> None:
        self.path = detect_path

    def _generate_combined_results_phase1(self) -> None:
        df_all_results = pd.DataFrame(columns=['name', 'epoch', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
       'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)',
       'metrics/mAP50-95(B)', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
       'lr/pg0', 'lr/pg1', 'lr/pg2'])
        
        for file in os.listdir(self.path):
            if file != 'tune':
                df = pd.read_csv(os.path.join(self.path,file, 'results.csv'))
                metrics = df.iloc[99, :]
                metrics.index = [x.strip(" ") for x in metrics.index]
                metrics_df = metrics.to_frame().T
                metrics_df['name'] = file  
                df_all_results = pd.concat([df_all_results, metrics_df], ignore_index=True)

        df_all_results.to_csv(os.path.join(self.path, 'tune', 'combined_results.csv'))

    def _generate_combined_results_phase2(self) -> None:
        path_phase1 = os.path.join(self.path, 'tune', 'combined_results.csv')
        path_hyperparameters = os.path.join(self.path, 'tune', 'tune_results.csv')
        df_results = pd.read_csv(path_phase1)
        df_hyperparams = pd.read_csv(path_hyperparameters)
        
        # Ordenar el DataFrame de resultados según la columna 'name'
        df_results_sorted = df_results.loc[natsorted(df_results.index, key=lambda i: df_results.at[i, 'name'])].reset_index(drop=True)

        # Comprobar que el número de filas coincide
        if df_results_sorted.shape[0] != df_hyperparams.shape[0]:
            raise ValueError("El número de filas en los DataFrames no coincide.")
        
        # Concatenar los DataFrames por columnas (axis=1)
        df_combined = pd.concat([df_results_sorted, df_hyperparams], axis=1)
        
        df_combined.to_csv(os.path.join(self.path, 'tune', 'combined_results_all.csv'))

    def plot_metrics_vs_name(self) -> None:
        df_combined = os.path.join(self.path, 'tune', 'combined_results_all.csv')
        df_combined = pd.read_csv(df_combined)
        df_combined['name_number'] = df_combined['name'].apply(lambda x: '1' if x == 'ateroesclerosis_tuning' else x.replace('ateroesclerosis_tuning', ''))
        df_combined['name_number'] = df_combined['name_number'].astype(int)
        
        metrics = ['metrics/recall(B)', 'metrics/precision(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
        
        for metric in metrics:
            plt.figure(figsize=(12, 8))
            
            sc = plt.scatter(df_combined['name_number'], df_combined[metric], c=df_combined['fitness'], cmap='viridis', edgecolor='k')
            smooth_values = gaussian_filter1d(df_combined[metric], sigma=2)
            plt.plot(df_combined['name_number'], smooth_values, color='orange', linestyle='--')            
            # Encontrar el valor máximo y su posición
            max_value = df_combined[metric].max()
            max_index = df_combined[df_combined[metric] == max_value]['name_number'].iloc[0]
            
            plt.axvline(x=max_index, color='red', linestyle='--')
            plt.axhline(y=max_value, color='red', linestyle='--')
            
            plt.text(max_index-2, max_value+0.001, f'({max_index}, {max_value:.2f})', color='red', fontsize=10, ha='right')
            
            plt.xlabel('Configuración (Número)')
            plt.ylabel(metric)
            plt.title(f'Evolución de {metric} a través de la configuración')
            plt.xticks(ticks=[])  
            
            plt.colorbar(sc, label='fitness')
            
            plt.tight_layout()
            name = metric.split('/')[-1].strip('(B)')
            plt.savefig(os.path.join(self.path, 'tune', f'{name}_evolution.png'))

    def analyze(self) -> None:
        self._generate_combined_results_phase1()
        self._generate_combined_results_phase2()
        self.plot_metrics_vs_name()

def main() -> None:
    analyzer = AnalyzerGA(detect_path="./data/results/week8/GA_Recall_config/100_epochs")
    analyzer.analyze()

if __name__ == "__main__":
    main()


