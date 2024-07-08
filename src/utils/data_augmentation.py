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
        aug_type = random.choice(['brightness', 'contrast', 'translation'])
        
        if aug_type == 'brightness':
            img = self.random_brightness(img)
            augmentation_type = "brightness"
        elif aug_type == 'contrast':
            img = self.random_contrast(img)
            augmentation_type = "contrast"
        elif aug_type == 'translation':
            tx = random.randint(-20, 20)
            ty = random.randint(-20, 20)
            if row['Groundtruth_path'] != 'nolesion':
                img, bbox_coords = self.random_translation(img, row['Groundtruth_path'], tx, ty)
                augmentation_type = "translation"
            else:
                img = self.translate_image_only(img, tx, ty)
                augmentation_type = "translation"

        # Save augmented image
        new_img_path, new_img_name = self.generate_new_path(img_path, dataset_type, augmentation_type, row['LesionLabel'])
        cv2.imwrite(new_img_path, img)

        # Handle bounding box
        groundtruth_path = row['Groundtruth_path']
        if groundtruth_path != 'nolesion':
            if augmentation_type == "translation":
                new_gt_path = self.save_translated_bounding_box(bbox_coords, new_img_name, dataset_type)
            else:
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

    def random_translation(self, img: np.ndarray, gt_path: str, tx: int, ty: int) -> Tuple[np.ndarray, List[int]]:
        """
        Aplica una traslación aleatoria a la imagen y ajusta la bounding box.
        """
        # Leer la bounding box desde el archivo
        with open(gt_path, 'r') as f:
            bbox = f.readline().strip().split()
            x, y, w, h, label = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), str(bbox[4])

        # Calcular el tamaño del borde que necesitamos añadir para evitar los bordes negros
        top = max(0, ty)
        bottom = max(0, -ty)
        left = max(0, tx)
        right = max(0, -tx)

        # Añadir bordes a la imagen usando la técnica de replicar bordes
        image_with_border = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)

        # Crear la matriz de traslación y aplicar la traslación a la imagen con bordes
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image_with_border = cv2.warpAffine(image_with_border, translation_matrix, (img.shape[1] + left + right, img.shape[0] + top + bottom))

        # Recortar la imagen trasladada a su tamaño original
        translated_image = translated_image_with_border[top:top + img.shape[0], left:left + img.shape[1]]

        # Trasladar las coordenadas de la bounding box
        new_x, new_y, new_w, new_h = self.translate_bounding_box(x, y, w, h, tx, ty)

        return translated_image, [new_x, new_y, new_w, new_h, label]

    @staticmethod
    def translate_point(x, y, tx, ty):
        """
        Aplica la traslación a un punto (x, y).
        """
        new_x = x + tx
        new_y = y + ty
        return new_x, new_y

    @staticmethod
    def translate_bounding_box(x, y, width, height, tx, ty):
        """
        Traslada las coordenadas de la bounding box.
        """
        new_x = x + tx
        new_y = y + ty
        return new_x, new_y, width, height

    def translate_image_only(self, img: np.ndarray, tx: int, ty: int) -> np.ndarray:
        """
        Aplica una traslación aleatoria solo a la imagen.
        """
        # Calcular el tamaño del borde que necesitamos añadir para evitar los bordes negros
        top = max(0, ty)
        bottom = max(0, -ty)
        left = max(0, tx)
        right = max(0, -tx)

        # Añadir bordes a la imagen usando la técnica de replicar bordes
        image_with_border = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)

        # Crear la matriz de traslación y aplicar la traslación a la imagen con bordes
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image_with_border = cv2.warpAffine(image_with_border, translation_matrix, (img.shape[1] + left + right, img.shape[0] + top + bottom))

        # Recortar la imagen trasladada a su tamaño original
        translated_image = translated_image_with_border[top:top + img.shape[0], left:left + img.shape[1]]

        return translated_image

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

    def save_translated_bounding_box(self, bbox_coords: List[int], new_img_name: str, dataset_type: str) -> str:
        # Extract patient and video info from the new image path
        path_parts = new_img_name.split('_')
        patient_video = path_parts[0] + '_' + path_parts[1]

        # Create the new bounding box directory path
        new_dir = os.path.join(self.base_output_path, dataset_type, 'labels', 'lesion', patient_video)
        os.makedirs(new_dir, exist_ok=True)

        # Generate the new bounding box path
        gt_name = new_img_name.replace('.png', '.txt')
        new_gt_path = os.path.join(new_dir, gt_name)

        # Save the new bounding box coordinates to the new file
        with open(new_gt_path, 'w') as f:
            # 4th position in the label
            f.write(f"{bbox_coords[0]} {bbox_coords[1]} {bbox_coords[2]} {bbox_coords[3]} {bbox_coords[4]}")

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
