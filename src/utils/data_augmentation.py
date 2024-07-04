import pandas as pd
from typing import List, Tuple
import random
import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

class DataAugmentor:
    def __init__(self, train_path: str, val_path: str, base_output_path: str):
        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path)
        self.base_output_path = base_output_path

    def augment_data(self, augmented_lesion_images: int, augmented_nolesion_images: int):
        # Augment training data
        self.augment_subset(self.train_df, 'train', augmented_lesion_images, augmented_nolesion_images)
        
        # Augment validation data
        self.augment_subset(self.val_df, 'val', augmented_lesion_images, augmented_nolesion_images)
        
        # Generate pre-augmentation and post-augmentation graphs
        self.generate_label_distribution_graph(self.train_df, 'train')
        self.generate_label_distribution_graph(self.val_df, 'val')

    def augment_subset(self, subset_df: pd.DataFrame, dataset_type: str, augmented_lesion_images: int, augmented_nolesion_images: int):
        lesion_images = subset_df[subset_df['LesionLabel'] != 'nolesion']
        nolesion_images = subset_df[subset_df['LesionLabel'] == 'nolesion']

        # Calculate distribution of lesion labels
        lesion_label_counts = lesion_images['LesionLabel'].value_counts()
        
        # Calculate augmentation factor, giving more weight to minority classes
        total_labels = len(lesion_label_counts)
        total_instances = lesion_label_counts.sum()
        
        # Calculate the weight for each label with a stronger bias towards minority classes
        weights = {label: (total_instances - count)**2 / total_instances**2 for label, count in lesion_label_counts.items()}
        
        # Normalize weights so that the sum of all augmented instances equals augmented_lesion_images
        total_weight = sum(weights.values())
        lesion_augmentation_counts = {label: int(augmented_lesion_images * weight / total_weight) for label, weight in weights.items()}

        augmented_images = []

        # Augment lesion images
        for label, count in lesion_augmentation_counts.items():
            label_images = lesion_images[lesion_images['LesionLabel'] == label]
            for _ in range(count):
                row = label_images.sample().iloc[0]
                augmented_images.append(self.apply_augmentation(row, dataset_type))

        # Augment no-lesion images to meet the specified number
        for _ in range(augmented_nolesion_images):
            row = nolesion_images.sample().iloc[0]
            augmented_images.append(self.apply_augmentation(row, dataset_type))

        augmented_df = pd.DataFrame(augmented_images, columns=subset_df.columns)
        augmented_df.to_csv(os.path.join(self.base_output_path, f'augmented_{dataset_type}.csv'), index=False)

        # Concatenate the original and augmented dataframes
        full_df = pd.concat([subset_df, augmented_df], ignore_index=True)
        full_df.to_csv(os.path.join(self.base_output_path, f'full_augmented_{dataset_type}.csv'), index=False)

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
        new_img_path, new_img_name = self.generate_new_path(img_path, dataset_type, augmentation_type, row['LesionLabel'])
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

    def generate_new_path(self, img_path: str, dataset_type: str, augmentation_type: str, lesion_label: str) -> Tuple[str, str]:
        # Extract patient and video info from the image path
        path_parts = img_path.split('/')
        patient_video = path_parts[-4] + "_" + path_parts[-3]

        # Determine the subdirectory based on lesion label
        sub_dir = 'lesion' if lesion_label != 'nolesion' else 'nolesion'
        base_dir = os.path.join(self.base_output_path, dataset_type, 'images', sub_dir, patient_video)
        os.makedirs(base_dir, exist_ok=True)

        # Generate the new image path with incremental numbering
        img_name = os.path.basename(img_path).replace('.png', f'_{augmentation_type}')
        img_base_name = os.path.splitext(img_name)[0]
        counter = 1
        new_img_name = f"{img_base_name}_{counter}.png"
        new_img_path = os.path.join(base_dir, new_img_name)

        while os.path.exists(new_img_path):
            counter += 1
            new_img_name = f"{img_base_name}_{counter}.png"
            new_img_path = os.path.join(base_dir, new_img_name)

        return new_img_path, new_img_name

    def copy_bounding_box(self, gt_path: str, new_img_name: str, dataset_type: str) -> str:
        # Extract patient and video info from the new image path
        path_parts = new_img_name.split('_')
        patient_video = path_parts[0] + '_' + path_parts[1]

        # Create the new bounding box directory path
        new_dir = os.path.join(self.base_output_path, dataset_type, 'labels', 'lesion', patient_video)
        os.makedirs(new_dir, exist_ok=True)

        # Generate the new bounding box path
        gt_name = new_img_name.replace('.png', '.txt')
        new_gt_path = os.path.join(new_dir, gt_name)

        # Copy the bounding box file
        shutil.copy(gt_path, new_gt_path)

        return new_gt_path

    def generate_label_distribution_graph(self, subset_df: pd.DataFrame, dataset_type: str):
        # Count labels pre-augmentation
        pre_aug_counts = subset_df['LesionLabel'].value_counts()

        # Load full augmented data
        augmented_df = pd.read_csv(os.path.join(self.base_output_path, f'full_augmented_{dataset_type}.csv'))
        post_aug_counts = augmented_df['LesionLabel'].value_counts()

        # Ensure all labels are in both counts
        all_labels = set(pre_aug_counts.index).union(set(post_aug_counts.index))
        pre_aug_counts = pre_aug_counts.reindex(all_labels, fill_value=0)
        post_aug_counts = post_aug_counts.reindex(all_labels, fill_value=0)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        index = range(len(all_labels))
        bar_width = 0.35

        bar1 = ax.bar(index, pre_aug_counts.values, bar_width, label='Pre-Augmentation')
        bar2 = ax.bar([i + bar_width for i in index], post_aug_counts.values, bar_width, label='Post-Augmentation')

        ax.set_xlabel('Labels')
        ax.set_ylabel('Count')
        ax.set_title(f'Label Distribution Pre and Post Augmentation ({dataset_type.capitalize()})')
        ax.set_xticks([i + bar_width / 2 for i in index])
        ax.set_xticklabels(all_labels, rotation=45)
        ax.legend()

        plt.tight_layout()
        plt_path = os.path.join(self.base_output_path, f'label_distribution_{dataset_type}.png')
        plt.savefig(plt_path)
        plt.close()

# Usage
train_path = './data/holdout/train.csv'
val_path = './data/holdout/val.csv'
base_output_path = '/home/mariopasc/Python/Datasets/Coronariografias/CADICA_Augmented'

augmentor = DataAugmentor(train_path, val_path, base_output_path)
augmentor.augment_data(augmented_lesion_images=500, augmented_nolesion_images=100)
