import numpy as np
import os
import glob
import shutil
import pandas as pd
import logging
from typing import Dict, List
import yaml
import ultralytics

# Limites de los hiperparámetros
HYPERPARAMETER_LIMITS = {
    'momentum': (0.4, 0.8),
    'lr0': (0.0, 1.0e-3),
    'lrf': (0.0, 1.0e-2), 
    'warmup_epochs': (2.5, 4),
    'warmup_momentum': (0.0, 1.0),
    'box': (7.0, 9.0), 
    'cls': (0.5, 0.8),
    'dfl': (0.9, 1.3)
}

# Configurar logging para errores
logging.basicConfig(filename='hyperparameter_tuning.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def _adjust_hyperparameter(hyperparam_name: str, base_value: float, coef: float) -> float:
    """
    Adjusts the hyperparameter value based on a Gaussian distribution centered on the base value,
    while ensuring the value stays within predefined limits for that hyperparameter.
    
    Args:
        hyperparam_name (str): The name of the hyperparameter being adjusted.
        base_value (float): The base value of the hyperparameter (center of the Gaussian).
        coef (float): Exploration-exploitation coefficient. Lower values focus on exploitation (closer values),
                      higher values encourage exploration (further values).
    
    Returns:
        float: The adjusted hyperparameter value constrained within the min and max limits.
    """
    min_value, max_value = HYPERPARAMETER_LIMITS.get(hyperparam_name, (None, None))
    
    if min_value is None or max_value is None:
        raise ValueError(f"Limits for hyperparameter '{hyperparam_name}' are not defined.")
    
    std_dev = base_value * coef  # Scale standard deviation by the base value and the coefficient
    adjusted_value = np.random.normal(loc=base_value, scale=std_dev)
    adjusted_value = max(min_value, min(adjusted_value, max_value))
    
    return adjusted_value

class HyperparameterTuning:
    
    def __init__(self, output: str, model: str, config_yaml_path: str, yaml_params_path: str) -> None:
        """
        Initializes the HyperparameterTuning class.
        
        Args:
            output (str): Path to the output folder where results will be stored.
            model (str): Model path or name for YOLO.
            config_yaml_path (str): Path to the YOLO configuration file (yaml).
            yaml_params_path (str): Path to the yaml file with hyperparameters.
        """
        self.output_folder = output
        self.model = ultralytics.YOLO(model=model)
        self.config = config_yaml_path
        self.default_params = self._load_yaml_to_dict(yaml_params_path)  # Base hyperparameters from YAML
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def _load_yaml_to_dict(self, yaml_path: str) -> Dict:
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    
    def _train(self, hyperparameters: Dict[str, float], run_name: str, epochs_per_iteration: int) -> None:
        try:
            params = {**self.default_params, **hyperparameters, 'save_period': 1, 'name': run_name, 'epochs': epochs_per_iteration}
            print(f"Training model with hyperparameters: {hyperparameters}, epochs: {epochs_per_iteration}")
            self.model.train(**params)
            self.val(run_name, hyperparameters)
        except Exception as e:
            logging.error(f"Error during training with hyperparameters {hyperparameters}: {str(e)}")
            print(f"An error occurred during training. Check 'hyperparameter_tuning.log' for details.")

    def _move_validation_logs(self, run_name: str) -> None:
        """
        Moves all `val*` folders into a `validation_log` folder within the run directory.
        
        Args:
            run_name (str): The name of the current run.
        """
        detect_folder = "./runs/detect"
        run_folder = os.path.join(detect_folder, run_name)
        validation_log_folder = os.path.join(run_folder, 'validation_log')
        
        if not os.path.exists(validation_log_folder):
            os.makedirs(validation_log_folder)
        
        # Move all val* folders
        folders = [folder for folder in os.listdir(detect_folder) if folder.startswith('val')]
        for val_folder in folders:
            val_folder_path = os.path.join(detect_folder, val_folder)  # Full path to the val folder
            if os.path.isdir(val_folder_path):  # Check if it's indeed a folder
                shutil.move(val_folder_path, validation_log_folder)  # Move the entire folder
                print(f"Moved {val_folder} to {validation_log_folder}")

        print(f"Moved all validation folders to {validation_log_folder}")

    def val(self, run_name: str, hyperparameters: Dict[str, float], tuning_csv: str = 'tuning_results.csv') -> None:
        """
        Validates the model, saves validation results for the hyperparameter, and computes aggregated metrics.
        
        Args:
            run_name (str): Name of the current run (used to locate the folder with weights).
            hyperparameters (Dict[str, float]): Hyperparameters used for this run.
            tuning_csv (str): Path to the CSV file where tuning results will be stored.
        """
        try:
            weights_folder = f"./runs/detect/{run_name}/weights"
            weight_files = sorted([file for file in glob.glob(os.path.join(weights_folder, '*.pt')) if 'last.pt' not in file])
            weight_files.append(os.path.join(weights_folder, 'last.pt'))  # Include last.pt

            # Store validation metrics
            metrics_data = []
            best_weight_file = os.path.join(weights_folder, 'best.pt')
            
            validation_results_path = os.path.join(f"./runs/detect/{run_name}", "validation_results.csv")

            for weight_file in weight_files:
                model_batch = ultralytics.YOLO(weight_file)
                results = model_batch.val(imgsz=640, conf=0.001, plots=True, save_json=True)

                map_50 = results.box.map50
                map_50_95 = results.box.map
                file_name = os.path.basename(weight_file)

                metrics_data.append({
                    'File_name': file_name,
                    'map50_95': map_50_95,
                    'map50': map_50
                })

            # Save validation results for this hyperparameter run
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics.to_csv(validation_results_path, mode='w', header=True, index=False)
            print(f"Validation results saved to {validation_results_path}")

            # Calculate mean and last metrics, excluding 'best.pt'
            df_without_best = df_metrics[df_metrics['File_name'] != 'best.pt']
            map50_95_mean = df_without_best['map50_95'].mean()
            map50_95_last = df_metrics[df_metrics['File_name'] == 'last.pt']['map50_95'].values[0]
            map50_mean = df_without_best['map50'].mean()
            map50_last = df_metrics[df_metrics['File_name'] == 'last.pt']['map50'].values[0]

            # Save final metrics in tuning CSV
            final_metrics = {
                'Hyperparameter': list(hyperparameters.keys())[0],
                'Value': list(hyperparameters.values())[0],
                'map50_95_mean': map50_95_mean,
                'map50_95_last': map50_95_last,
                'map50_mean': map50_mean,
                'map50_last': map50_last
            }

            final_df = pd.DataFrame([final_metrics])
            final_df.to_csv(tuning_csv, mode='a', header=not os.path.exists(tuning_csv), index=False)
            print(f"Tuning results saved to {tuning_csv}")

            # Move validation logs
            self._move_validation_logs(run_name)

            # Save the best.pt file and remove the weights folder
            if os.path.exists(best_weight_file):
                new_best_file_name = f'{list(hyperparameters.keys())[0]}_{run_name}_best.pt'
                shutil.copy(best_weight_file, os.path.join(self.output_folder, new_best_file_name))
                print(f"Saved best.pt as '{new_best_file_name}'")

            shutil.rmtree(weights_folder)
            print(f"Removed weights folder: {weights_folder}")

        except Exception as e:
            logging.error(f"Error during validation of run {run_name}: {str(e)}")
            print(f"An error occurred during validation. Check 'hyperparameter_tuning.log' for details.")

    def tune_hyperparameters(self, hyperparameters: Dict[str, List[float]], coef: float = 0.1, num_iterations: int = 10, 
                             epochs_per_iteration: int = 5, random: bool = True) -> None:
        for hyperparam_name, values in hyperparameters.items():
            base_value = self.default_params.get(hyperparam_name, None)
            if base_value is None:
                print(f"Hyperparameter {hyperparam_name} not found in the base YAML parameters.")
                continue

            if random:
                print(f"\nTuning hyperparameter: {hyperparam_name} with base value: {base_value} using random values.")
                for iteration in range(num_iterations):
                    run_name = f'{hyperparam_name}_run_{iteration}'
                    adjusted_value = _adjust_hyperparameter(hyperparam_name, base_value, coef)
                    print(f"Iteration {iteration + 1}/{num_iterations}: Adjusted {hyperparam_name} to {adjusted_value}")
                    adjusted_hyperparameters = {hyperparam_name: adjusted_value}
                    self._train(adjusted_hyperparameters, run_name, epochs_per_iteration)
            else:
                print(f"\nTuning hyperparameter: {hyperparam_name} with provided values.")
                for iteration, value in enumerate(values):
                    run_name = f'{hyperparam_name}_run_{iteration}'
                    print(f"Iteration {iteration + 1}/{len(values)}: Using provided value {value} for {hyperparam_name}")
                    provided_hyperparameters = {hyperparam_name: value}
                    self._train(provided_hyperparameters, run_name, epochs_per_iteration)

        print(f"\nTuning completed.")


def random_values() -> None:
    hyperparameters = {'lr0': [], 'momentum': []}
    tuner = HyperparameterTuning(output='output_folder', model='yolov8l.pt', config_yaml_path='config.yaml', yaml_params_path='params.yaml')

    # Usar generación aleatoria con coeficiente de exploración 0.2, 10 iteraciones y 5 épocas por iteración
    tuner.tune_hyperparameters(hyperparameters, coef=0.2, num_iterations=10, epochs_per_iteration=5, random=True)

def fixed_values() -> None:

    # Hiperparámetros de la mejor configuración base
    lr0 = 1.0e-05
    lrf = 0.00829
    momentum = 0.70064
    weight_decay = 0.00048
    warmup_epochs = 3.66787
    warmup_momentum = 0.78696
    warmup_bias_lr = 0.1
    box = 8.57719
    cls = 0.68361
    dfl = 1.19862

    hyperparameters = {
        'lr0': [5.0e-6, 7.5e-6, lr0, 5.0e-5, 7.5e-5],  # Distribución simétrica para lr0
        'lrf': [0.004, 0.00614, lrf, 0.012, 0.016],  # Distribución simétrica para lrf
        'momentum': [0.68, 0.69, momentum, 0.71, 0.72],  # Distribución simétrica para momentum
        'weight_decay': [0.00024, 0.00036, weight_decay, 0.0006, 0.00072],  # Simétrico para weight_decay
        #'warmup_epochs': [2.5, 3.0, warmup_epochs, 4.0, 4.5],  # Simétrico para warmup_epochs
        'warmup_momentum': [0.69, 0.74, warmup_momentum, 0.83, 0.88],  # Simétrico para warmup_momentum
        'warmup_bias_lr': [0.05, 0.075, warmup_bias_lr, 0.15, 0.175],  # Simétrico para warmup_bias_lr
        'box': [7.8, 8.18, box, 8.98, 9.38],  # Simétrico para box
        'cls': [0.61, 0.65, cls, 0.72, 0.75],  # Simétrico para cls
        'dfl': [1.05, 1.12, dfl, 1.27, 1.34]  # Simétrico para dfl
    }

    tuner = HyperparameterTuning(output='output_tuning', model='yolov8l.pt', config_yaml_path='config.yaml', yaml_params_path='args.yaml')

    # Usar los valores proporcionados en la lista con 5 épocas por iteración
    tuner.tune_hyperparameters(hyperparameters, epochs_per_iteration=100, random=False)


fixed_values()