import torch
import open_clip
from PIL import Image
import json
import numpy as np

names_path = './azov_sea_benthos_names.json'

class BenthosClip():
    def __init__(self, checkpoint = 'hf-hub:imageomics/bioclip'):
        with open(names_path,'r') as f:
            list_of_names = json.load(f)
        self.names_decoder = {}
        for i, n in enumerate(list_of_names['names']):
            self.names_decoder[i] = n

        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(checkpoint)
        self.tokenizer = open_clip.get_tokenizer(checkpoint)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.names = self.tokenizer(list_of_names['names'])

    def __call__(self, image: Image): #TODO do batch inference
        image = self.preprocess_val(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(self.names)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        text_probs = text_probs.numpy()
        max_id = np.argmax(text_probs[0])
        true_name = self.names_decoder[max_id]
        res = {}
        for i, p in enumerate(text_probs[0]):
            res[self.names_decoder[i]] = p

        full_res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}   
        return full_res, true_name
