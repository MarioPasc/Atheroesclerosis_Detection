import pandas as pd
import matplotlib.pyplot as plt
import os

class DatasetAnalyzer:
    def __init__(self, train_path: str, augmented_train_path: str):
        self.train_df = pd.read_csv(train_path)
        self.augmented_train_df = pd.read_csv(augmented_train_path)

    def compare_lesion_counts(self):
        # Create a column to indicate presence of lesion
        self.train_df['LesionPresence'] = self.train_df['LesionLabel'] != 'nolesion'
        self.augmented_train_df['LesionPresence'] = self.augmented_train_df['LesionLabel'] != 'nolesion'

        # Count lesions and no lesions in the original and augmented datasets
        pre_counts = self.train_df['LesionPresence'].value_counts()
        post_counts = self.augmented_train_df['LesionPresence'].value_counts()

        # Create a DataFrame for plotting
        compare_df = pd.DataFrame({
            'pre-augmentation': pre_counts,
            'post-augmentation': post_counts
        }).fillna(0)

        # Rename index for better clarity
        compare_df.index = ['No Lesion', 'Lesion']

        # Plot the data
        fig, ax = plt.subplots()
        compare_df.plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
        ax.set_title('Lesion vs No Lesion Counts Before and After Augmentation')
        ax.set_ylabel('Count')
        ax.set_xlabel('Lesion Presence')
        plt.xticks(rotation=0)

        # Save the figure
        output_dir = './figures/augmentation'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'lesion_comparison.png'))
        plt.close()

    def compare_lesion_label_counts(self):
        # Filter only lesion images
        pre_lesion_df = self.train_df[self.train_df['LesionLabel'] != 'nolesion']
        post_lesion_df = self.augmented_train_df[self.augmented_train_df['LesionLabel'] != 'nolesion']

        # Count lesion labels in the original and augmented datasets
        pre_counts = pre_lesion_df['LesionLabel'].value_counts()
        post_counts = post_lesion_df['LesionLabel'].value_counts()

        # Define lesion categories
        lesion_categories = ['p70_90', 'p90_98', 'p99', 'p100']
        
        # Create a DataFrame for plotting
        compare_df = pd.DataFrame({
            'pre-augmentation': [pre_counts.get(label, 0) for label in lesion_categories],
            'post-augmentation': [post_counts.get(label, 0) for label in lesion_categories]
        }, index=lesion_categories)

        # Plot the data
        fig, ax = plt.subplots()
        compare_df.plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
        ax.set_title('Lesion Label Counts Before and After Augmentation')
        ax.set_ylabel('Count')
        ax.set_xlabel('LesionLabel')
        plt.xticks(rotation=0)

        # Save the figure
        output_dir = './figures/augmentation'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'lesion_label_comparison.png'))
        plt.close()

# Usage
train_path = './data/holdout/train.csv'
augmented_train_path = '/home/mariopasc/Python/Datasets/Coronariografias/CADICA_Augmented/augmented_train.csv'

analyzer = DatasetAnalyzer(train_path, augmented_train_path)
analyzer.compare_lesion_counts()
analyzer.compare_lesion_label_counts()
