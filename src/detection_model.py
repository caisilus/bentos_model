
from ultralytics import YOLO

MODEL_PATH = "./model/best.pt"
import cv2

class Detection_model:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.thereshhold = 0.5


    def detect(self, img_path):
        detection_results = self.model(img_path)[0]
        confs = detection_results.boxes.conf.tolist()
        #(l,u,r,d)
        boxes = detection_results.boxes.data.tolist()
        detections = [{'box': [int(b) for b in box[:4]], 'conf': conf} for box, conf in zip(boxes, confs) if conf > self.thereshhold]

        return {'image': detection_results.orig_img, 'detections': detections}
    
    def show_detections(self, detections_data):
        image_np = detections_data['image']
        for det in detections_data['detections']:

            self.draw_rectangle_with_name(image_np, det['box'], det['name'])
        return detections_data
    
    def draw_rectangle_with_name(self, image_np, coordinates, name):
        x1, y1, x2, y2 = coordinates

        cv2.rectangle(image_np, (x2, y2), (x1, y1), (0, 255, 0), 10)
        cv2.putText(image_np, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 0), 10)

