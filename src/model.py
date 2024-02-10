from src.utils import convert_image_to_base64
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
        detections = [{'box': box[:4], 'conf': conf} for box, conf in zip(boxes, confs) if conf > self.thereshhold]

        return {'orig': detection_results.orig_img, 'detections': detections}
    
    def get_name_for(self, detections_data):
        for det in detections_data['detections']:
            det['name'] = 'name'
        return detections_data
    
    def show_detections(self, detections_data):
        image_np = detections_data['orig']
        for det in detections_data['detections']:

            self.draw_rectangle_with_name(image_np, det['box'], det['name'])
        return detections_data
    
    def draw_rectangle_with_name(self, image_np, coordinates, name):
        coordinates = [int(c) for c in coordinates]
        x1, y1, x2, y2 = coordinates

        cv2.rectangle(image_np, (x2, y2), (x1, y1), (0, 255, 0), 10)
        cv2.putText(image_np, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 0), 10)

    def predict(self, url_image):
        detections_data = self.detect(url_image)
        detections_data = self.get_name_for(detections_data)
        detections_data = self.show_detections(detections_data)
        detections_data['orig'] = convert_image_to_base64(detections_data['orig'])
        return detections_data

