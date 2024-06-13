from utils.data_loader import DataLoader
from utils.holdout import HoldOut
import os 


def main():
    base_path = "/home/mariopasc/Python/Datasets/Coronariografias/CADICA"
    output_csv_path = os.path.join('./data/info_dataset.csv')

    data_loader = DataLoader(base_path)
    df = data_loader.process_data()
    data_loader.save_to_csv(df, output_csv_path)

    holdout = HoldOut('./data/info_dataset.csv')
    holdout.process()

if __name__ == "__main__":
    main()
