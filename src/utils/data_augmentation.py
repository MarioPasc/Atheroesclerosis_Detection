import pandas as pd
from typing import List, Tuple
import random
import cv2
import numpy as np
import os
import shutil

class DataAugmentor:
    def __init__(self, train_path: str, val_path: str, base_output_path: str):
        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path)
        self.base_output_path = base_output_path

    def augment_data(self, lesion_ratio: int = 2):
        lesion_images = self.train_df[self.train_df['LesionLabel'] != 'nolesion']
        nolesion_images = self.train_df[self.train_df['LesionLabel'] == 'nolesion']

        augmented_images = []

        # Augment lesion images
        for _, row in lesion_images.iterrows():
            for _ in range(lesion_ratio):
                augmented_images.append(self.apply_augmentation(row, 'train'))

        # Augment no-lesion images
        for _, row in nolesion_images.iterrows():
            augmented_images.append(self.apply_augmentation(row, 'train'))

        augmented_df = pd.DataFrame(augmented_images, columns=self.train_df.columns)
        augmented_df.to_csv(os.path.join(self.base_output_path, 'augmented_train.csv'), index=False)

    def apply_augmentation(self, row: pd.Series, dataset_type: str) -> pd.Series:
        img_path = row['Frame_path']
        img = cv2.imread(img_path)
        augmentation_type = ""

        # Apply random augmentations
        if random.random() > 0.5:
            img = self.random_brightness(img)
            augmentation_type = "brightness"
        else:
            img = self.random_contrast(img)
            augmentation_type = "contrast"

        # Save augmented image
        new_img_path, new_img_name = self.generate_new_path(img_path, dataset_type, augmentation_type)
        cv2.imwrite(new_img_path, img)

        # Handle bounding box
        groundtruth_path = row['Groundtruth_path']
        if groundtruth_path != 'nolesion':
            new_gt_path = self.copy_bounding_box(groundtruth_path, new_img_name, dataset_type)
        else:
            new_gt_path = 'nolesion'

        new_row = row.copy()
        new_row['Frame_path'] = new_img_path
        new_row['Groundtruth_path'] = new_gt_path
        return new_row

    @staticmethod
    def random_brightness(img: np.ndarray) -> np.ndarray:
        value = random.uniform(0.9, 1.1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], value)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def random_contrast(img: np.ndarray) -> np.ndarray:
        alpha = random.uniform(0.9, 1.1)
        new_img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, 0)
        return new_img

    def generate_new_path(self, img_path: str, dataset_type: str, augmentation_type: str) -> Tuple[str, str]:
        # Extract patient and video info from the image path
        path_parts = img_path.split('/')
        patient_video = path_parts[-2]

        # Create the new directory path
        new_dir = os.path.join(self.base_output_path, dataset_type, 'augmented_images', patient_video)
        os.makedirs(new_dir, exist_ok=True)

        # Generate the new image path with incremental numbering
        img_name = os.path.basename(img_path).replace('.png', f'_{augmentation_type}')
        img_base_name = os.path.splitext(img_name)[0]
        counter = 1
        new_img_name = f"{img_base_name}_{counter}.png"
        new_img_path = os.path.join(new_dir, new_img_name)

        while os.path.exists(new_img_path):
            counter += 1
            new_img_name = f"{img_base_name}_{counter}.png"
            new_img_path = os.path.join(new_dir, new_img_name)

        return new_img_path, new_img_name

    def copy_bounding_box(self, gt_path: str, new_img_name: str, dataset_type: str) -> str:
        # Extract patient and video info from the new image path
        path_parts = new_img_name.split('_')
        patient_video = path_parts[0] + '_' + path_parts[1]

        # Create the new bounding box directory path
        new_dir = os.path.join(self.base_output_path, dataset_type, 'bounding_boxes', patient_video)
        os.makedirs(new_dir, exist_ok=True)

        # Generate the new bounding box path
        gt_name = new_img_name.replace('.png', '.txt')
        new_gt_path = os.path.join(new_dir, gt_name)

        # Copy the bounding box file
        shutil.copy(gt_path, new_gt_path)

        return new_gt_path

# Usage
train_path = './data/holdout/train.csv'
val_path = './data/holdout/val.csv'
base_output_path = '/home/mariopasc/Python/Datasets/Coronariografias/CADICA_Augmented'

augmentor = DataAugmentor(train_path, val_path, base_output_path)
augmentor.augment_data()
