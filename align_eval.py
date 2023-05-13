import requests
import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel
from torchvision.datasets import CIFAR10, CIFAR100

processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
model = AlignModel.from_pretrained("kakaobrain/align-base")

test_dataset = CIFAR100(root='./data', train=False, download=True)

# test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

random_int = torch.randint(0, len(test_dataset), (1,)).item()

sample = test_dataset[random_int]
image = sample[0]

classes = test_dataset.classes

inputs = processor(text=classes, images=image, return_tensors="pt")

# print(inputs)

with torch.no_grad():
    outputs = model(**inputs)

# this is the image-text similarity score
logits_per_image = outputs.logits_per_image

# we can take the softmax to get the label probabilities
probs = logits_per_image.softmax(dim=1)


print("predicted class:", classes[torch.argmax(probs).item()])
print("true class:", classes[sample[1]])