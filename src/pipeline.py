from utils.data_loader import DataLoader
from utils.holdout import HoldOut
from utils.yolo_dataset_generator import DatasetGenerator
from utils.data_augmentation import DataAugmentor
from utils.dataset_analyzer import DatasetAnalyzer
import os 
import pandas as pd
from typing import List

def generate_dataset_csv(base_path:str, output_csv_path:str) -> pd.DataFrame:
    data_loader = DataLoader(base_path = base_path)
    df = data_loader.process_data()
    data_loader.save_to_csv(df, output_csv_path)
    data_loader.plot_lesion_label_distribution(df=df)
    return df

def holdout_dataset(csv_path:str, val_size:float, test_size:float, labels: List[str]) -> None:
    holdout = HoldOut(csv_path = csv_path, val_size = val_size, test_size = test_size)
    holdout.process(labels=labels)

def generate_dataset(train_csv:str, val_csv:str, test_csv:str, dataset_dir:str) -> None:
    dataset_generator = DatasetGenerator(train_csv=train_csv, val_csv=val_csv, test_csv=test_csv, dataset_dir=dataset_dir)
    dataset_generator.generate_dataset()

def data_augmentation(train_csv:str, val_csv:str, augmented_output:str, augmented_lesion_images:int, augmented_nolesion_images:int):
    augmentor = DataAugmentor(train_csv, val_csv, augmented_output)
    augmentor.augment_data(augmented_lesion_images, augmented_nolesion_images)

def analyze_dataset(dataset_path:str, save_path:str, augmentations:List[str]):
    analyzer = DatasetAnalyzer(dataset_path, save_path, augmentations)
    analyzer.analyze_dataset()

def main():
    base_path = "/home/mariopasc/Python/Datasets/Coronariografias/CADICA"
    output_csv_path = os.path.join('./data/info_dataset.csv')
    df = generate_dataset_csv(base_path = base_path,
                              output_csv_path = output_csv_path)
    print('Holdout')
    holdout_dataset(csv_path = './data/info_dataset.csv', val_size = .2, test_size = .2, 
                labels=["p70_90", "p90_98", "p99", "p100"])
    #"p0_20", "p20_50", "p50_70", 
    print('Applying data augmentation')
    augmented_output = '/home/mariopasc/Python/Datasets/Coronariografias/CADICA_Augmented'
    data_augmentation(train_csv='./data/holdout/train.csv', val_csv='./data/holdout/val.csv',
                      augmented_output= augmented_output,
                      augmented_lesion_images=1500, augmented_nolesion_images=900)
    print('Processing YOLOv8 Dataset')
    dataset_dir='/home/mariopasc/Python/Datasets/Coronariografias/CADICA_Detection'
    generate_dataset(train_csv=os.path.join(augmented_output, 'full_augmented_train.csv'),
                    val_csv=os.path.join(augmented_output, 'full_augmented_val.csv'),
                    test_csv='./data/holdout/test.csv',
                    dataset_dir=dataset_dir)

    augmentations = ['translation', 'contrast', 'brightness', 'xray_noise']  # Lista de aumentaciones
    analyze_dataset(dataset_path=dataset_dir, save_path="./figures/distribution",
                    augmentations=augmentations)


if __name__ == "__main__":
    main()
