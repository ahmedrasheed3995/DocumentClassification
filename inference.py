import os

from PIL import Image

import torch
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3Tokenizer, LayoutLMv3Processor, LayoutLMv3ForSequenceClassification, AdamW
# import matplotlib.pyplot as plt


label2idx = {'resume': 0, 'scientific_publication': 1, 'email': 2}

feature_extractor = LayoutLMv3FeatureExtractor()
tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
processor = LayoutLMv3Processor(feature_extractor, tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LayoutLMv3ForSequenceClassification.from_pretrained("models/saved_model")
model.to(device)

query = os.path.join("/image_dir", os.getenv('IMAGE'))
image = Image.open(query).convert("RGB")
encoded_inputs = processor(image, return_tensors="pt").to(device)
outputs = model(**encoded_inputs)
preds = torch.softmax(outputs.logits, dim=1).tolist()[0]
pred_labels = {label:pred for label, pred in zip(label2idx.keys(), preds)}
print(pred_labels)

# plt.imshow(np.array(image))
# plt.show()

