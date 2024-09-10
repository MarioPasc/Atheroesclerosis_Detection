import os
import pandas as pd
import matplotlib.pyplot as plt
import natsort

class FineTuningAnalyzer:

    def __init__(self, results_folder: str) -> None:
        """
        Initializes the FineTuningAnalyzer class with the results folder.
        
        :param results_folder: Path to the folder containing subdirectories with results.csv files.
        """
        self.results_folder = results_folder
        self.metrics = {
            'metrics/precision(B)': 'Precision',
            'metrics/recall(B)': 'Recall',
            'metrics/mAP50(B)': 'mAP50',
            'metrics/mAP50-95(B)': 'mAP50-95'
        }

    def plot_results(self):
        """
        Generates subplots for each specified metric showing the evolution over epochs for different parameter values.
        The legend values are sorted naturally in ascending order.
        """
        folders = [f for f in os.listdir(self.results_folder) if os.path.isdir(os.path.join(self.results_folder, f))]
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.ravel()  # Flatten the axis matrix for easy iteration
        
        for idx, (metric, metric_name) in enumerate(self.metrics.items()):
            sorted_labels = []
            sorted_data = []
            
            # Collect all data and labels for sorting
            for folder in folders:
                csv_path = os.path.join(self.results_folder, folder, 'results.csv')
                if os.path.exists(csv_path):
                    this_results = pd.read_csv(csv_path)
                    this_results.columns = this_results.columns.str.strip()
                    if metric in this_results.columns:
                        label = folder.split('_')[-1]
                        sorted_labels.append(label)
                        sorted_data.append(this_results)

            # Sort the labels naturally (handle floats and scientific notation)
            sorted_indices = natsort.index_natsorted(sorted_labels, alg=natsort.ns.FLOAT)
            sorted_labels = [sorted_labels[i] for i in sorted_indices]
            sorted_data = [sorted_data[i] for i in sorted_indices]

            # Plot the sorted data
            for label, data in zip(sorted_labels, sorted_data):
                axs[idx].plot(data['epoch'], data[metric], label=label)
            
            axs[idx].set_title(metric_name)
            axs[idx].set_xlabel('Epoch')
            axs[idx].set_ylabel('Value')
            axs[idx].legend(loc='lower right')
            axs[idx].grid(True)

        plt.tight_layout()
        output_path = os.path.join(self.results_folder, 'results_plot.png')
        plt.savefig(output_path)
        plt.show()

    def save_metrics_summary(self, sort_by: str = 'Recall'):
        """
        Saves a summary of metrics in a markdown file.
        The summary is sorted by the specified metric in descending order.
        
        :param sort_by: Metric to sort by (e.g., 'Precision', 'Recall', 'mAP50', 'mAP50-95')
        """
        folders = [f for f in os.listdir(self.results_folder) if os.path.isdir(os.path.join(self.results_folder, f))]
        summary_data = []

        for folder in folders:
            csv_path = os.path.join(self.results_folder, folder, 'results.csv')
            if os.path.exists(csv_path):
                this_results = pd.read_csv(csv_path)
                this_results.columns = this_results.columns.str.strip()
                averages = {}
                for metric, metric_name in self.metrics.items():
                    if metric in this_results.columns:
                        averages[metric_name] = this_results[metric].mean()
                averages['Parameter'] = folder.split('_')[-1]
                summary_data.append(averages)

        summary_df = pd.DataFrame(summary_data)

        # Check if the sort_by column is present in the summary
        if sort_by in summary_df.columns:
            summary_df = summary_df.sort_values(by=sort_by, ascending=False)

        markdown_table = summary_df.to_markdown(index=False)

        with open(os.path.join(self.results_folder, 'metrics_summary.md'), 'w') as f:
            f.write("# Metrics Summary\n")
            f.write(markdown_table)

def main():
    results_folder = "./data/results/week11/Tune_lr0_yolov8"
    analyzer = FineTuningAnalyzer(results_folder=results_folder)
    analyzer.plot_results()
    
    # You can specify the metric to sort by here, e.g., 'Recall'
    analyzer.save_metrics_summary(sort_by='Recall')

if __name__ == "__main__":
    main()
