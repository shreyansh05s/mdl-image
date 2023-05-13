import requests
import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm.auto import tqdm
import json
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
model = AlignModel.from_pretrained("kakaobrain/align-base")

model.to(device)
model.eval()

with open("cifar100_classes.json", "r") as f:
    classes = json.load(f)


def transform_image_ALIGN(image):
    image = processor(images=image, return_tensors="pt")
    image = {k: v.squeeze() for k, v in image.items()}
    return image


classes_processed = processor(text=classes, return_tensors="pt")
classes_processed.to(device)

test_dataset = CIFAR100(
    root="./data", train=False, download=True, transform=transform_image_ALIGN
)


test_dataloader = DataLoader(
    test_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True
)


print(classes_processed)
eval_accuracy = 0
steps = 1
for data in tqdm(test_dataloader):
    inputs = data[0]
    inputs["pixel_values"] = inputs["pixel_values"].to(device)
    classes_processed.to(device)
    inputs.update(classes_processed)

    with torch.no_grad():
        outputs = model(**inputs)

    label = data[1].to(device)

    # this is the image-text similarity score
    logits_per_image = outputs.logits_per_image

    probs = logits_per_image.softmax(dim=1)

    temp_eval_accuracy = torch.sum(torch.argmax(probs, dim=1) == label) / len(label)

    eval_accuracy += temp_eval_accuracy
    if steps % 100 == 0:
        tqdm.write(f"Accuracy: {eval_accuracy / steps}")
    steps += 1

print(eval_accuracy / len(test_dataset))


print("predicted class:", classes[torch.argmax(probs).item()])
print("true class:", classes[sample[1]])
