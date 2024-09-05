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
            'epochs': 80,
            'time': None,
            'patience': 100,
            'batch':16,
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
            'single_cls': False, # Desactivamos esta opción para ver cómo afecta el balanceo de datos
            'rect': False,
            'cos_lr': True,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'plots': True,
            'optimizer': 'Adam',
            'iou': 0.5, # YOLO lo tiene por defecto en .7, lo dejaremos en .5 como estándar
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
            # Augmentation parameters, set to False
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

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject: str, body: str, to_email: str):
    from_email = "mario.pg02@gmail.com"  
    from_password = "dddw jysk gbaj knhm "  

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, from_password)
    text = msg.as_string()
    server.sendmail(from_email, to_email, text)
    server.quit()

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

    def train_with_different_lr0(self, lr0_values: list) -> None:
        for lr0 in lr0_values:
            experiment_name = f"finetune_lr0{lr0}"
            print(f"Training with lr0 = {lr0}")
            hyperparameters = {
                'lr0': lr0,
                'name': experiment_name
            }
            self.detection_model.train(hyperparameters)

    def train_with_different_momentum(self, momentum_values: list) -> None:
        for momentum in momentum_values:
            experiment_name = f"finetune_mom{momentum}"
            print(f"Training with momentum = {momentum}")
            hyperparameters = {
                'momentum': momentum,
                'name': experiment_name
            }
            self.detection_model.train(hyperparameters)

    def train_with_different_lrf(self, lrf_values: list) -> None:
        for lrf in lrf_values:
            experiment_name = f"finetune_lrf{lrf}"
            print(f"Training with lrf = {lrf}")
            hyperparameters = {
                'lrf': lrf,
                'name': experiment_name
            }
            self.detection_model.train(hyperparameters)

def main() -> int:
    detection_model = Detection_YOLOv8(model_path="", yaml_path="./config.yaml")
    tuner = YOLOv8_Tuning(detection_model)

    lr0_values = [0.001, 0.005, 0.01, 0.015, 0.02]
    momentum_values = [0.6, 0.65, 0.7, 0.72, 0.75, 0.78]
    lrf_values = [0.01, 0.025, 0.05, 0.075, 0.1, 0.25]

    # Train with different lr0 values
    tuner.train_with_different_lr0(lr0_values)
    send_email(
        subject="Training LR0 Complete",
        body="The training with different LR0 values has been completed.",
        to_email="mario.pg02@gmail.com"  
    )
    # Train with different momentum values
    #tuner.train_with_different_momentum(momentum_values)
    #send_email(
    #    subject="Training Momentum Complete",
    #    body="The training with different Momentum values has been completed.",
    #    to_email="mario.pg02@gmail.com"  
    #)
    # Train with different lrf values
    #tuner.train_with_different_lrf(lrf_values)
    #send_email(
    #    subject="Training LRF Complete",
    #    body="The training with different LRF values has been completed.",
    #    to_email="mario.pg02@gmail.com"  
    #)

if __name__ == "__main__":
    main()
