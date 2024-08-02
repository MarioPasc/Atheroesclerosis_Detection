import pandas as pd
import os
import ultralytics
import glob as glob
import itertools

class Detection_YOLOv8:

    def __init__(self, yaml_path: str, model_path: str) -> None:
        if model_path == "":
            model_path = 'yolov8l.pt'
        self.model = ultralytics.YOLO(model=model_path)
        self.yaml_path = yaml_path

    def train(self, hyperparameters: dict) -> None:
        # Merge default hyperparameters with the provided ones
        default_params = {
            'data': self.yaml_path,
            'epochs': 120,
            'imgsz': 512,
            'save': True,
            'save_period': -1,
            'name': "ateroesclerosis_training",
            'verbose': True,
            'seed': 42,
            'single_cls': True,
            'plots': True,
            'cos_lr': True,
            'lr0': 0.0001,
            'lrf': 0.01,
            'momentum': 0.85,
            'weight_decay': 0.001,
            'optimizer': 'AdamW',
            'warmup_epochs': 1,
            'label_smoothing': 0.1,
            'dropout': 0.05,
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
            'batch': 4
        }
        params = {**default_params, **hyperparameters}
        self.model.train(**params)

    def tune(self) -> None:
        results_tuning = self.model.tune(data=self.yaml_path, epochs=30, iterations=300, save=False, plots=False, val=False,
                                         name="ateroesclerosis_tuning", seed=42, single_cls=True,
                                         augment=False,
                                         hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0,
                                         scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0,
                                         mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0, erasing=0.0)

    def val(self) -> None:
        weights_folder = "./runs/detect/ateroesclerosis_training/weights"
        results_list = []

        # Get all the weight files, excluding 'last.pt'
        weight_files = sorted([file for file in glob.glob(os.path.join(weights_folder, '*.pt')) if 'last.pt' not in file])
        weight_files.append(os.path.join(weights_folder, 'last.pt'))  # Include last.pt
        weight_files.insert(0, os.path.join(weights_folder, 'best.pt'))  # Include best.pt

        for epoch, weight in enumerate(weight_files):
            if os.path.basename(weight) == 'best.pt':
                epoch_num = 0
            elif os.path.basename(weight) == 'last.pt':
                epoch_num = 100
            else:
                epoch_num = int(os.path.basename(weight).replace('epoch', '').replace('.pt', ''))

            model_batch = ultralytics.YOLO(weight)
            results = model_batch.val(imgsz=640, conf=0.01, plots=True)

            # Collect results
            precision_b = results.box.mp  # precisión media
            recall_b = results.box.mr  # recall medio
            map_05_b = results.box.map50  # mAP para th=.5
            map_05_95_b = results.box.map  # mAP para th=.5-.95

            results_list.append({
                "epoch": epoch_num,
                "weight": os.path.basename(weight),
                "precision": precision_b,
                "recall": recall_b,
                "map_05": map_05_b,
                "map_05_95": map_05_95_b
            })

        # Save results to a CSV file
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(os.path.join('./runs/detect/ateroesclerosis_training', 'validation_results.csv'), index=False)
        print("Validation results saved to 'validation_results.csv'")

class YOLOv8_Tuning:

    def __init__(self, detection_model: Detection_YOLOv8, base_path: str = './runs/detect') -> None:
        self.detection_model = detection_model
        self.base_path = base_path

    def train_with_hyperparameter_grid(self, lr0_values: list, momentum_values: list, lrf_values: list) -> None:
        for lr0, momentum, lrf in itertools.product(lr0_values, momentum_values, lrf_values):
            experiment_name = f"finetune_mom{momentum}_lr0{lr0}_lrf{lrf}"
            print(f"Training with lr0 = {lr0}, momentum = {momentum}, lrf = {lrf}")
            hyperparameters = {
                'lr0': lr0,
                'momentum': momentum,
                'lrf': lrf,
                'name': experiment_name
            }
            self.detection_model.train(hyperparameters)

def main() -> int:
    detection_model = Detection_YOLOv8(model_path="", yaml_path="./config.yaml")
    tuner = YOLOv8_Tuning(detection_model)

    lr0_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    momentum_values = [0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98]
    lrf_values = [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5]
    tuner.train_with_hyperparameter_grid(lr0_values, momentum_values, lrf_values)

    return 0

if __name__ == "__main__":
    main()
