import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict

class DatasetAnalyzer:
    def __init__(self, data_path: str, save_path: str, augmentations: List[str]) -> None:
        """
        Inicializa la clase DatasetAnalyzer con la ruta de datos, ruta de guardado y las aumentaciones.

        Args:
            data_path (str): Ruta al conjunto de datos.
            save_path (str): Ruta donde se guardarán las imágenes generadas.
            augmentations (List[str]): Lista con los nombres de las aumentaciones.
        """
        self.data_path = data_path
        self.save_path = save_path
        self.augmentations = augmentations
        self.sets = ['train', 'val', 'test']
        self.extensions_images = ['.png']
        self.extensions_labels = ['.txt']

    def count_files(self, path: str, extensions: List[str]) -> int:
        """
        Cuenta la cantidad de archivos en un directorio dado con las extensiones especificadas.

        Args:
            path (str): Ruta al directorio.
            extensions (List[str]): Lista de extensiones de archivos a contar.

        Returns:
            int: Número de archivos encontrados con las extensiones especificadas.
        """
        return len([f for f in os.listdir(path) if any(f.endswith(ext) for ext in extensions)])

    def analyze_dataset(self) -> None:
        """
        Realiza el análisis del dataset y genera las gráficas de distribución de labels y aumentaciones.
        """
        counts_images = {}
        counts_labels = {}
        counts_nolesion = {}
        counts_augmentations = {set_name: {aug: 0 for aug in self.augmentations} for set_name in self.sets}

        for set_name in self.sets:
            images_path = os.path.join(self.data_path, 'images', set_name)
            labels_path = os.path.join(self.data_path, 'labels', set_name)

            counts_images[set_name] = self.count_files(images_path, self.extensions_images)
            counts_labels[set_name] = self.count_files(labels_path, self.extensions_labels)

            # Contar las imágenes sin lesión
            counts_nolesion[set_name] = counts_images[set_name] - counts_labels[set_name]

            # Contar las imágenes aumentadas
            total_augmented = 0
            for image_file in os.listdir(images_path):
                if image_file.endswith('.png'):
                    for aug in self.augmentations:
                        if aug in image_file:
                            counts_augmentations[set_name][aug] += 1
                            total_augmented += 1
                            break

            # Calcular imágenes sin aumentación
            counts_augmentations[set_name]['none'] = counts_images[set_name] - total_augmented

        self._plot_images_labels_distribution(counts_images, counts_labels, counts_nolesion)
        self._plot_augmentations_heatmap(counts_augmentations)

    def _plot_images_labels_distribution(self, counts_images: Dict[str, int], counts_labels: Dict[str, int], counts_nolesion: Dict[str, int]) -> None:
        """
        Genera y guarda una gráfica de barras comparando el número de imágenes, labels y nolesion por set.

        Args:
            counts_images (Dict[str, int]): Diccionario con los conteos de imágenes por set.
            counts_labels (Dict[str, int]): Diccionario con los conteos de labels por set.
            counts_nolesion (Dict[str, int]): Diccionario con los conteos de imágenes sin lesión por set.
        """
        fig, ax = plt.subplots()
        bar_width = 0.25
        index = range(len(self.sets))

        bar1 = plt.bar(index, [counts_images[set_name] for set_name in self.sets], bar_width, label='Images')
        bar2 = plt.bar([i + bar_width for i in index], [counts_labels[set_name] for set_name in self.sets], bar_width, label='Labels')
        bar3 = plt.bar([i + 2 * bar_width for i in index], [counts_nolesion[set_name] for set_name in self.sets], bar_width, label='Nolesion Images')

        plt.xlabel('Dataset')
        plt.ylabel('Count')
        plt.title('Number of Images, Labels, and Nolesion Images per Dataset')
        plt.xticks([i + bar_width for i in index], self.sets)
        plt.legend()

        # Añadir los valores encima de cada barra
        for bars in [bar1, bar2, bar3]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom', fontsize=8)

        plt.savefig(os.path.join(self.save_path, 'images_labels_nolesion_distribution.png'))
        plt.close(fig)

    def _plot_augmentations_heatmap(self, counts_augmentations: Dict[str, Dict[str, int]]) -> None:
        """
        Genera y guarda un heatmap de las imágenes aumentadas por set y tipo de aumentación.

        Args:
            counts_augmentations (Dict[str, Dict[str, int]]): Diccionario con los conteos de aumentaciones por set y tipo.
        """
        augmentations_data = []
        for set_name in self.sets:
            for aug in self.augmentations + ['none']:
                augmentations_data.append([set_name, aug, counts_augmentations[set_name][aug]])

        df_augmentations = pd.DataFrame(augmentations_data, columns=['Set', 'Augmentation', 'Count'])
        df_pivot = df_augmentations.pivot_table(values='Count', index='Augmentation', columns='Set')

        # Convertir los valores a enteros
        df_pivot = df_pivot.astype(int)

        plt.figure(figsize=(8, 6))
        sns.heatmap(df_pivot, annot=True, fmt='d', cmap='YlGnBu')
        plt.title('Heatmap of Image Augmentations')

        plt.savefig(os.path.join(self.save_path, 'augmentations_heatmap.png'))
        plt.close()
