import torch
import open_clip

from typing import List
from PIL import Image

class BenthosClip():
    def __init__(self, species_names: List[str], checkpoint = 'hf-hub:imageomics/bioclip'):
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(checkpoint)
        self.tokenizer = open_clip.get_tokenizer(checkpoint)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.names = self.tokenizer(species_names)

    def __call__(self, image: Image): #TODO do batch inference
        image = self.preprocess_val(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(self.names)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        print("Label probs:", text_probs)
        return torch.argmax(text_probs)
