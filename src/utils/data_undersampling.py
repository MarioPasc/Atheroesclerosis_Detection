import os
import pandas as pd
import matplotlib.pyplot as plt

class DataUndersampling:
    def __init__(self, holdout_path: str, class_dict: dict):
        """
        Class to perform undersampling on dataset splits based on a given class distribution.

        Args:
            holdout_path (str): Path to the folder containing train.csv, val.csv, and test.csv.
            class_dict (dict): Dictionary where keys are class names and values are lists with 3 integers
                               representing the percentage of images to ignore in [train, val, test] splits.
        """
        self.holdout_path = holdout_path
        self.class_dict = class_dict

        # Paths to the CSV files
        self.train_csv = os.path.join(self.holdout_path, 'train.csv')
        self.val_csv = os.path.join(self.holdout_path, 'val.csv')
        self.test_csv = os.path.join(self.holdout_path, 'test.csv')

        # DataFrames to hold pre- and post-undersampling data
        self.pre_undersampling_df = None
        self.post_undersampling_df = None

    def apply_undersampling(self):
        """
        Apply undersampling to the train, val, and test splits based on the specified percentages.
        """
        # Load the CSV files and save the original data for comparison later
        self.pre_undersampling_df = pd.concat([
            pd.read_csv(self.train_csv),
            pd.read_csv(self.val_csv),
            pd.read_csv(self.test_csv)
        ])

        # Apply undersampling based on the class_dict
        train_df = self.undersample_class(pd.read_csv(self.train_csv), split_type='train')
        val_df = self.undersample_class(pd.read_csv(self.val_csv), split_type='val')
        test_df = self.undersample_class(pd.read_csv(self.test_csv), split_type='test')

        # Save the modified CSV files
        train_df.to_csv(self.train_csv, index=False)
        val_df.to_csv(self.val_csv, index=False)
        test_df.to_csv(self.test_csv, index=False)

        # Save the post-undersampling data for comparison
        self.post_undersampling_df = pd.concat([train_df, val_df, test_df])

        # Plot the undersampling results
        class_labels = ["p0_20", "p20_50", "p50_70", "p70_90", "p90_98", "p99", "p100"]
        self.plot_undersampling_results(class_labels)

    def undersample_class(self, df: pd.DataFrame, split_type: str) -> pd.DataFrame:
        """
        Undersample the specified class for a given split (train, val, test).

        Args:
            df (pd.DataFrame): DataFrame for the specific split (train, val, test).
            split_type (str): The type of split ('train', 'val', 'test').

        Returns:
            pd.DataFrame: The DataFrame after applying undersampling.
        """
        # Map split_type to the corresponding index in the class_dict value
        split_index_map = {'train': 0, 'val': 1, 'test': 2}
        split_index = split_index_map[split_type]

        # Loop over each class in the class_dict
        for class_name, percentages in self.class_dict.items():
            # Get the percentage to ignore for this split
            percentage_to_ignore = percentages[split_index]

            if percentage_to_ignore > 0:
                # Filter the DataFrame for the specific class
                class_df = df[df['LesionLabel'] == class_name]

                # Calculate the number of rows to remove
                num_to_remove = int(len(class_df) * (percentage_to_ignore / 100))

                # Randomly select the rows to remove
                rows_to_remove = class_df.sample(n=num_to_remove, random_state=42).index

                # Drop the selected rows from the original DataFrame
                df = df.drop(rows_to_remove)

        return df

    def plot_undersampling_results(self, class_labels: list):
        """
        Generate a barplot comparing the number of instances pre-undersampling vs post-undersampling.

        Args:
            class_labels (list): List of class labels to include on the x-axis.

        Saves:
            The generated plot as a PNG file in './figures/undersampling/'.
        """

        # Calculate the total number of instances per class label pre- and post-undersampling
        pre_counts = self.pre_undersampling_df['LesionLabel'].value_counts().reindex(class_labels, fill_value=0)
        post_counts = self.post_undersampling_df['LesionLabel'].value_counts().reindex(class_labels, fill_value=0)

        # Plot the results
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.35
        index = range(len(class_labels))

        # Create the bars
        ax.bar(index, pre_counts, bar_width, label='Pre-Undersampling', color='#1f77b4')
        ax.bar([i + bar_width for i in index], post_counts, bar_width, label='Post-Undersampling', color='#ff7f0e')

        # Set the labels and title
        ax.set_xlabel('Class Labels')
        ax.set_ylabel('Number of Instances')
        ax.set_title('Comparison of Instances Pre- and Post-Undersampling')
        ax.set_xticks([i + bar_width / 2 for i in index])
        ax.set_xticklabels(class_labels)
        ax.legend()

        # Add grid and adjust layout
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the figure
        output_dir = './figures/undersampling'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'undersampling_comparison.png'))


