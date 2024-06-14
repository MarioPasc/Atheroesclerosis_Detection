from utils.data_loader import DataLoader
from utils.holdout import HoldOut
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

def main():
    base_path = "/home/mariopasc/Python/Datasets/Coronariografias/CADICA"
    output_csv_path = os.path.join('./data/info_dataset.csv')

    df = generate_dataset_csv(base_path = base_path,
                              output_csv_path = output_csv_path)
    holdout_dataset(csv_path = './data/info_dataset.csv', val_size = .2, test_size = .2, 
                    labels=["p70_90", "p90_98", "p99", "p100"])

if __name__ == "__main__":
    main()
