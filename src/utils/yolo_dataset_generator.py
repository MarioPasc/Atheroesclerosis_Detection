import pandas as pd
import os
import shutil
from typing import Optional
from tqdm import tqdm 

class DatasetGenerator:
    """
    # Estos class mappings solo podrán ser usados cuando se entrene el modelo con todas las clases
    class_mappings = {
        "p0_20": 0,
        "p20_50": 1,
        "p50_70":2,
        "p70_90": 3,
        "p90_98": 4,
        "p99": 5,
        "p100": 6
    }
    """
    class_mappings = {
        "p70_90": 0,
        "p90_98": 1,
        "p99": 2,
        "p100": 3
    }
    def __init__(self, train_csv: str, val_csv: str, test_csv: str, dataset_dir: str):
        """
        Initialize the DatasetGenerator with paths to the CSV files and the output directory.

        :param train_csv: Path to the training CSV file.
        :param val_csv: Path to the validation CSV file.
        :param test_csv: Path to the test CSV file.
        :param dataset_dir: Path to the output directory where the dataset will be generated.
        """
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.dataset_dir = dataset_dir
        
        os.makedirs(self.dataset_dir, exist_ok=True)

        # Read the CSV files
        self.train_df = pd.read_csv(train_csv)
        self.val_df = pd.read_csv(val_csv)
        self.test_df = pd.read_csv(test_csv)
        
        # Define the required directory structure
        self.structure = {
            'images': ['train', 'val', 'test'],
            'labels': ['train', 'val', 'test']
        }
        
    def create_directories(self) -> None:
        """
        Create the directory structure for the YOLO dataset.
        """
        for key, subdirs in self.structure.items():
            for subdir in subdirs:
                path = os.path.join(self.dataset_dir, key, subdir)
                os.makedirs(path, exist_ok=True)
                
    def convert_bbox_format(self, bbox: str, img_width: int, img_height: int) -> str:
        """
        Convert bounding box from [x, y, w, h, class] to YOLO format.

        :param bbox: Bounding box in the format "x y w h class".
        :param img_width: Width of the image.
        :param img_height: Height of the image.
        :return: Bounding box in YOLO format "class x_center y_center width height".
        """
        x, y, w, h, cls = bbox.split()
        x, y, w, h = int(x), int(y), int(w), int(h)
        cls = self.class_mappings[cls]
        
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        w /= img_width
        h /= img_height
        
        return f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                
    def copy_files(self, df: pd.DataFrame, split: str) -> None:
        """
        Copy image and label files to their respective directories.
        
        :param df: DataFrame containing the file paths.
        :param split: The dataset split (train, val, or test).
        """
        for _, row in tqdm(df.iterrows(), desc='Generating YOLO Dataset...'):
            image_src = row['Frame_path']
            label_src = row['Groundtruth_path']
            
            image_dst = os.path.join(self.dataset_dir, 'images', split, os.path.basename(image_src))
            label_dst = os.path.join(self.dataset_dir, 'labels', split, os.path.basename(label_src).replace('.txt', '.txt'))
            
            shutil.copy(image_src, image_dst)
            
            if label_src != 'nolesion':
                try:
                    with open(label_src, 'r') as f:
                        bboxes = f.readlines()
                    
                    img_width, img_height = self.get_image_dimensions(image_src)
                    
                    with open(label_dst, 'w') as f:
                        for bbox in bboxes:
                            yolo_bbox = self.convert_bbox_format(bbox.strip(), img_width, img_height)
                            f.write(yolo_bbox + '\n')
                except FileNotFoundError:
                    print(f"Label file not found: {label_src}, skipping.")
            
    def get_image_dimensions(self, image_path: str) -> tuple:
        """
        Get the dimensions of an image.

        :param image_path: Path to the image file.
        :return: A tuple (width, height) of the image.
        """
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size
            
    def generate_dataset(self) -> None:
        """
        Generate the YOLO dataset by copying files to the appropriate directories.
        """
        self.create_directories()
        
        # Copy files for each split
        self.copy_files(self.train_df, 'train')
        self.copy_files(self.val_df, 'val')
        self.copy_files(self.test_df, 'test')


