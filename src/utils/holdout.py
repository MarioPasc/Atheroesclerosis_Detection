import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List

class HoldOut:
    def __init__(self, csv_path: str, val_size: float, test_size: float, output_dir: str = './data/holdout') -> None:
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.val_size = val_size
        self.test_size = test_size
        self.df = pd.read_csv(csv_path)
        self.df['GroundTruthFile'] = self.df['GroundTruthFile'].fillna('nolesion')
        os.makedirs(output_dir, exist_ok=True)

    def filter_labels(self, labels: List[str]) -> pd.DataFrame:
        # Filtrar las muestras que tienen lesiones y cuyo label está en la lista proporcionada
        filtered_df = self.df[(self.df['Lesion'] == False) | (self.df['LesionLabel'].isin(labels))]
        return filtered_df

    def split_data(self, filtered_df: pd.DataFrame, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        unique_videos = filtered_df['Video_Paciente'].unique()
        train_videos, test_videos = train_test_split(unique_videos, test_size=self.test_size, random_state=random_state)
        train_videos, val_videos = train_test_split(train_videos, test_size=self.val_size/(1-self.test_size), random_state=random_state)
        
        train_df = filtered_df[filtered_df['Video_Paciente'].isin(train_videos)]
        val_df = filtered_df[filtered_df['Video_Paciente'].isin(val_videos)]
        test_df = filtered_df[filtered_df['Video_Paciente'].isin(test_videos)]
        
        return train_df, val_df, test_df

    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        for split_name, df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
            tqdm.pandas(desc=f'Processing {split_name} split')
            split_df = df[['SelectedFramesNonLesionVideo', 'SelectedFramesLesionVideo', 'GroundTruthFile', 'Lesion', 'LesionLabel']].copy()
            split_df['Frame_path'] = split_df.progress_apply(
                lambda row: row['SelectedFramesLesionVideo'] if row['Lesion'] else row['SelectedFramesNonLesionVideo'], axis=1
            )
            split_df['Groundtruth_path'] = split_df['GroundTruthFile'].apply(
                lambda x: x if isinstance(x, str) and x.startswith('/home/') else 'nolesion'
            )
            split_df['LesionLabel'] = split_df['LesionLabel'].fillna('nolesion')
            split_df = split_df[['LesionLabel', 'Frame_path', 'Groundtruth_path']]
            split_df.to_csv(os.path.join(self.output_dir, f"{split_name}.csv"), index=False)

    def generate_plots(self) -> None:
        os.makedirs('./figures', exist_ok=True)
        
        data_splits = ['train', 'val', 'test']
        lesions_count = {'train': [0, 0], 'val': [0, 0], 'test': [0, 0]}
        
        for split in tqdm(data_splits, desc='Generating plots'):
            df = pd.read_csv(os.path.join(self.output_dir, f"{split}.csv"))
            lesions_count[split][0] = df[df['LesionLabel'] != 'nolesion'].shape[0]  # Con lesión
            lesions_count[split][1] = df[df['LesionLabel'] == 'nolesion'].shape[0]  # Sin lesión

        labels = ['Train', 'Validation', 'Test']
        lesion_values = [lesions_count[split][0] for split in data_splits]
        no_lesion_values = [lesions_count[split][1] for split in data_splits]
        
        x = range(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots()
        ax.bar(x, no_lesion_values, width, label='Sin lesión')
        ax.bar([p + width for p in x], lesion_values, width, label='Con lesión')
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Número de frames')
        ax.set_title('Distribución de frames con y sin lesión en cada conjunto de datos')
        ax.set_xticks([p + width/2 for p in x])
        ax.set_xticklabels(labels)
        ax.legend()

        plt.savefig('./figures/lesion_distribution.png')

    def process(self, labels: List[str]) -> None:
        filtered_df = self.filter_labels(labels)
        train_df, val_df, test_df = self.split_data(filtered_df)
        self.save_splits(train_df, val_df, test_df)
        self.generate_plots()