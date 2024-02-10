import cv2
import base64

def convert_image_to_base64(image_np):
    _, buffer = cv2.imencode('.jpg', image_np)
    jpg_as_text = base64.b64encode(buffer)
    base64_string = jpg_as_text.decode('utf-8')
    return base64_string