import ultralytics
import os

class Detection_YOLOv8:

    def __init__(self, yaml_path:str, model_path:str) -> None:
        if model_path == "":
            model_path = 'yolov8l.pt'
        self.model = ultralytics.YOLO(model=model_path)
        self.yaml_path = yaml_path

    def train(self) -> None:
        results_train = self.model.train(data=self.yaml_path, epochs=3, imgsz=640,
                                         save=True, save_period=1,
                                         name="ateroesclerosis_training", verbose=True,
                                         seed=42, single_cls=True, plots=True,
                                         augment=False, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0,
                                         scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0, mosaic=0.0,
                                         close_mosaic=0, mixup=0.0, copy_paste=0.0, auto_augment="", erasing=0.0)


def main() -> int:
    model = Detection_YOLOv8(model_path="", yaml_path="./config.yaml")
    model.train()
    return 0

if __name__ == "__main__":
    main()