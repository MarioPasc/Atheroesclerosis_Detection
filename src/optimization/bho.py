import optuna
from ultralytics import YOLO
from typing import Dict, Tuple, List

# Global default parameters for YOLO training
default_params = {
    'data': None,  # This will be set dynamically in the class
    'epochs': 100,
    'time': None,
    'patience': 100,
    'batch': 4,
    'imgsz': 640,
    'save': True,
    'save_period': 1,
    'cache': False,
    'device': None,
    'workers': 8,
    'project': None,
    'name': "ateroesclerosis_training",
    'exist_ok': False,
    'pretrained': True,
    'verbose': True,
    'seed': 42,
    'deterministic': True,
    'single_cls': True, 
    'rect': False,
    'cos_lr': True,
    'resume': False,
    'amp': True,
    'fraction': 1.0,
    'profile': False,
    'freeze': None,
    'plots': True,
    'optimizer': 'Adam',
    'iou': 0.5,  # Standardized IoU value
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'label_smoothing': 0.0,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'pose': 12.0,
    'kobj': 1.0,
    'dropout': 0.0,
    'val': True,
    # Augmentation parameters
    'augment': False,
    'hsv_h': 0.0,
    'hsv_s': 0.0,
    'hsv_v': 0.0,
    'degrees': 0.0,
    'translate': 0.0,
    'scale': 0.0,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.0,
    'mosaic': 0.0,
    'close_mosaic': 0,
    'mixup': 0.0,
    'copy_paste': 0.0,
    'erasing': 0.0,
    'crop_fraction': 0.0, 
    'auto_augment': "",
    'bgr': 0.0, 
}

class BHOYOLO:
    def __init__(self, model: str, hyperparameters: Dict[Tuple[str, str], List], data: str, epochs: int = 100, img_size: int = 640) -> None:
        """
        Implementation of Bayesian Hyperparameter Optimization (BHO) for YOLOv8.

        params:
            - model (str): YOLO model as a string, e.g., 'yolov8', 'yolov9'
            - hyperparameters (Dict[Tuple[str, str], List]): Hyperparameters to optimize. 
                e.g., {('lr0', 'log'): [1e-6, 1e-3], ('momentum', 'uniform'): [0.55, 0.8], ('batch_size', 'categorical'): [16, 32, 64]}
            - data (str): Path to the dataset YAML file
            - epochs (int): Number of epochs for each training trial
            - img_size (int): Image size for training
        """

        # Set the model based on input, defaulting to YOLOv8
        self.model_name = model
        if model == "yolov8":
            self.model = 'yolov8l.pt'
        elif model == "yolov9":
            self.model = 'yolov9e.pt'
        elif model == "yolov10":
            self.model = 'yolov10l.pt'
        else:
            raise ValueError(f"Model {model} not recognized.")

        # Store hyperparameters and training details
        self.hyperparameters = hyperparameters
        self.yaml_path = data  # Store dataset path
        self.epochs = epochs
        self.img_size = img_size

    def _extract_hyperparameters(self, trial) -> Dict[str, float]:
        """
        Function to extract hyperparameters from the search space using the trial object.

        params:
            - trial: The Optuna trial object for suggesting values

        returns:
            - A dictionary of hyperparameters with values suggested by the trial
        """
        final_hyperparam = {}

        # Loop over the hyperparameters and their function type
        for (hyperparam, func_type), values in self.hyperparameters.items():
            if func_type == "log":
                final_hyperparam[hyperparam] = trial.suggest_loguniform(hyperparam, values[0], values[1])
            elif func_type == "uniform":
                final_hyperparam[hyperparam] = trial.suggest_uniform(hyperparam, values[0], values[1])
            elif func_type == "categorical":
                final_hyperparam[hyperparam] = trial.suggest_categorical(hyperparam, values)
            else:
                raise ValueError(f"Unknown function type {func_type} for hyperparameter {hyperparam}")

        return final_hyperparam

    def _train_model(self, trial) -> float:
        """
        Objective function that trains the YOLO model with the suggested hyperparameters and returns the mAP50-95.

        params:
            - trial: The Optuna trial object to suggest hyperparameters

        returns:
            - mAP50-95: The validation metric (mAP) after training
        """

        # Extract hyperparameters for the trial
        hyperparams = self._extract_hyperparameters(trial)
        
        # Merge with default_params
        params = {**default_params, **hyperparams}
        params['data'] = self.yaml_path  # Ensure the dataset path is correct
        params['epochs'] = self.epochs

        # Initialize the YOLO model
        model = YOLO(self.model)

        # Train the model using the merged hyperparameters
        results = model.train(**params)

        # Retrieve the mAP50-95 from results
        map_50_95 = results.box.map 

        # Log intermediate values for the trial
        trial.report(map_50_95, step=self.epochs)

        # Early stopping condition if intermediate values are poor
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return map_50_95

    def optimize(self, n_trials: int = 50) -> None:
        """
        Runs the Bayesian Optimization for the defined number of trials.

        params:
            - n_trials (int): Number of optimization trials to run
        """

        # Create an Optuna study to maximize the mAP50-95
        study = optuna.create_study(direction="maximize")

        # Run optimization
        study.optimize(self._train_model, n_trials=n_trials)

        # Print best results after optimization
        print(f"Best trial mAP50-95: {study.best_trial.value}")
        print(f"Best hyperparameters: {study.best_trial.params}")

        # Optionally, save the study for future reference
        study.trials_dataframe().to_csv("study_results.csv", index=False)


def main() -> None:
    model = "yolov8"
    data = "src/models/config.yaml"
    
    # Define the hyperparameter search space
    hyperparameters = {
        ('lr0', 'log'): [1e-6, 1e-3],
        ('momentum', 'uniform'): [0.55, 0.95],
        ('weight_decay', 'log'): [1e-5, 1e-2]
    }

    # Instantiate the BHOYOLO class
    bho_yolo = BHOYOLO(model=model, hyperparameters=hyperparameters, data=data, epochs=3)

    # Run optimization
    bho_yolo.optimize(n_trials=5)  # Adjust the number of trials as needed


if __name__ == "__main__":
    main()
