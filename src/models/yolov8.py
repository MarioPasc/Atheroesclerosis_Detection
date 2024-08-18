import external.ultralytics as ultralytics
import os
import glob
import pandas as pd

class Detection_YOLOv8:

    def __init__(self, yaml_path:str, model_path:str) -> None:
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
            'save_period': 1,
            'name': "ateroesclerosis_training",
            'verbose': True,
            'seed': 42,
            'single_cls': True,
            'plots': True,
            'optimizer': 'AdamW',
            'deterministic': True,
            'cos_lr': True,
            'amp': True,
            'dropout': 0.0,
            'iou': 0.7,
            'lr0': 2.0e-5,
            'lrf': 0.00438,
            'momentum': 0.71169,
            'weight_decay': 0.00043,
            'warmup_epochs': 3.9316,
            'warmup_momentum': 0.2153,
            'warmup_bias_lr': 0.1,
            'label_smoothing': 0.0,
            'box': 4.29656,
            'cls': 0.35645,
            'dfl': 1.98713,
            'pose': 12.0,
            'kobj': 1.0,
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
            'batch': 16
        }
        params = {**default_params, **hyperparameters}
        self.model.train(**params)
        
    def tune(self) -> None:
        results_tuning = self.model.tune(data=self.yaml_path, epochs=90, iterations=100, save=True, plots=True, val=True,
                                         name="ateroesclerosis_tuning",seed=42, single_cls=True, cos_lr=True, deterministic = True,
                                         box = 3.77048, cls = 0.39766, dfl=1.88035, lr0=2.0e-05, lrf=0.00603, momentum=0.77038, 
                                         weight_decay = 0.00053, warmup_epochs = 3.40376, warmup_momentum = 0.1909, imgsz = 512,
                                         optimizer = 'AdamW', augment=False, crop_fraction = 0.0, iou = 0.7, 
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


def main() -> int:
    model = Detection_YOLOv8(model_path="", yaml_path="./config.yaml")
    model.train()
    model.val()
    #model.tune()
    return 0

if __name__ == "__main__":
    main()