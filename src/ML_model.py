from src.detection_model import Detection_model
from src.benthos_clip_model import BenthosClip
from PIL import Image
import matplotlib.pyplot as plt
from src.utils import convert_image_to_base64

class ML_model():
    def __init__(self):
        self.detection_model = Detection_model()
        self.clip_model = BenthosClip()
        self.verbose = True

    def handle_images(self, images):
        all_data = []
        for image in images:
            detections_data = self.detection_model.detect(image)
            for det in detections_data['detections']:
                box = det['box']
                det_img = Image.fromarray(detections_data['image'][box[1]:box[3],box[0]:box[2]], 'RGB')
                all_info, true_name = self.clip_model(det_img)
                print(true_name)
                det['name'] = true_name
            detections_data = self.detection_model.show_detections(detections_data)
            if self.verbose:
                print(all_info)
                plt.imshow(detections_data['image'])
            detections_data['image'] = convert_image_to_base64(detections_data['image'])
        all_data.append(detections_data)
        return all_data




