from utils.data_loader import DataLoader
from utils.holdout import HoldOut
import os 


def main():
    base_path = "/home/mariopasc/Python/Datasets/Coronariografias/CADICA"
    output_csv_path = os.path.join('./data/info_dataset.csv')

    data_loader = DataLoader(base_path = base_path)
    df = data_loader.process_data()
    data_loader.save_to_csv(df, output_csv_path)

    holdout = HoldOut(csv_path = './data/info_dataset.csv',
                      val_size = .2, test_size = .2)
    holdout.process()

if __name__ == "__main__":
    main()
