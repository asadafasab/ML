from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import cv2
import tqdm
import numpy as np
import detectron2
from detectron2.utils.logger import setup_logger
import pyautogui
setup_logger()


# Initialize predictor
cfg = get_cfg()
cfg.merge_from_file("config.yml")
predictor = DefaultPredictor(cfg)
# Initialize visualizer
v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN).set(thing_classes=["fist","mid","ok hand"]), ColorMode.IMAGE)
v = VideoVisualizer(MetadataCatalog.get(
    cfg.DATASETS.TRAIN).set(thing_classes=["fist","mid","ok hand"]), ColorMode.IMAGE)

cam = cv2.VideoCapture(0)

def move(d):
    pyautogui.keyDown("left")
    pyautogui.keyUp("up")
    pyautogui.keyUp("right")

while True:
    _, frame = cam.read()

    if cv2.waitKey(1) == 27:  # Esc
        break

    outputs = predictor(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    visualization = v.draw_instance_predictions(
        frame, outputs["instances"].to("cpu"))

    if len(outputs["instances"].pred_classes):
        if int(outputs["instances"].pred_classes[0])==0:
            print(f"0: left")
            pyautogui.keyDown("left")
            pyautogui.keyUp("up")
            pyautogui.keyUp("right")
        elif int(outputs["instances"].pred_classes[0])==1:
            print(f"1: jump")
            pyautogui.keyDown("up")
            pyautogui.keyUp("left")
            pyautogui.keyUp("right")
        elif int(outputs["instances"].pred_classes[0])==2:
            print(f"2: right")
            pyautogui.keyDown("right")
            pyautogui.keyUp("left")
            pyautogui.keyUp("up")
    else:
        pyautogui.keyUp("right")
        pyautogui.keyUp("left")
        pyautogui.keyUp("up")

    visualization = cv2.cvtColor(
        visualization.get_image(), cv2.COLOR_RGB2BGR)

    cv2.imshow("O", visualization)


cam.release()
cv2.destroyAllWindows()
















