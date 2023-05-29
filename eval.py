import torch
from torch import nn
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from transformers import (
    AutoModelForImageClassification,
    ViTImageProcessor,
    AutoFeatureExtractor,
    AlignModel,
    AlignProcessor,
    ViTForImageClassification,
    ViTConfig,
    Swinv2ForImageClassification,
    Swinv2Config,
    AutoImageProcessor,
)
import argparse
import json
import os
from model import ImageClassifier
from functools import partial
from torchvision import transforms


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open("cifar100_classes.json", "r") as f:
    cifar100_classes = json.load(f)


def transform_image(image):
    image = extractor(images=image, return_tensors="pt")
    image["pixel_values"] = image["pixel_values"].squeeze()
    return image

# define a transform function to preprocess for ResNet
def transform_image_RESNET(image):
    image = transforms.ToTensor()(image)
    image = transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))(image)
    return image


def transform_image_ALIGN(image):
    image = extractor(images=image, return_tensors="pt")
    image = {k: v.squeeze() for k, v in image.items()}
    return image


models = {
    "VIT": {
        "model": AutoModelForImageClassification.from_pretrained(
            "Ahmed9275/Vit-Cifar100"
        ),
        "extractor": ViTImageProcessor.from_pretrained("Ahmed9275/Vit-Cifar100"),
        "transform_image": transform_image,
    },
    "SWIN": {
        "model": AutoModelForImageClassification.from_pretrained(
            "MazenAmria/swin-small-finetuned-cifar100"
        ),
        "extractor": ViTImageProcessor.from_pretrained(
            "MazenAmria/swin-small-finetuned-cifar100"
        ),
        "transform_image": transform_image,
    },
    "VIT-ASAM": {
        "model": {
            "class": ViTForImageClassification,
            "pretrained": "google/vit-base-patch16-224-in21k",
            "freeze": False,
        },
        "config": ViTConfig,
        "extractor": AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        ),
        "transform_image": transform_image,
        "model_dir": os.path.join(
            "models", "vit", "effortless-wildflower-54", "vit_cifar100.pth"
        ),
    },
    "SWIN-ASAM": {
        "model": {
            "class": Swinv2ForImageClassification,
            "pretrained": "microsoft/swinv2-large-patch4-window12-192-22k",
            "freeze": True,
        },
        "config": Swinv2Config,
        "extractor": AutoImageProcessor.from_pretrained(
            "microsoft/swinv2-large-patch4-window12-192-22k"
        ),
        "transform_image": transform_image,
        "model_dir": os.path.join(
            "models", "swin", "denim-cosmos-61", "swin_cifar100.pth"
        ),
    },
    "RESNET": {
        "model": torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True
        ),
        "transform_image": transform_image_RESNET
    },
    # will not be used in the paper
    # "ALIGN": {
    #     "model": AlignModel.from_pretrained("kakaobrain/align-base"),
    #     "extractor": AlignProcessor.from_pretrained("kakaobrain/align-base"),
    #     "transform_image": transform_image_ALIGN,
    # },
}


def get_dataset(transform_image=None, batch_size=16):
    test_dataset = CIFAR100(
        root="./data", train=False, download=True, transform=transform_image
    )

    if batch_size is None:
        test_dataloader = None
    else:
        test_dataloader = DataLoader(
            test_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True
        )

    return test_dataset, test_dataloader


# define a helper function to calculate the accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="VIT")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    default_batch_size = 16

    args = parse_args()
    model = models[args.model]["model"]

    if "model_dir" in models[args.model]:
        config = models[args.model]["config"].from_pretrained(
            models[args.model]["model"]["pretrained"]
        )
        config.num_labels = 100
        model = ImageClassifier(
            num_classes=100, model=models[args.model]["model"], config=config
        )

        model.load_state_dict(torch.load(models[args.model]["model_dir"]))

    extractor = (
        models[args.model]["extractor"]
        if "extractor" in models[args.model]
        else None
    )
    transform_image = (
        models[args.model]["transform_image"]
        if "transform_image" in models[args.model]
        else None
    )
    batch_size = (
        models[args.model]["batch_size"]
        if "batch_size" in models[args.model]
        else default_batch_size
    )

    if args.model == "ALIGN":
        classes_processed = extractor(text=cifar100_classes, return_tensors="pt")

    model.to(device)
    model.eval()

    test_dataset, test_dataloader = get_dataset(transform_image, batch_size)

    if batch_size is None:
        dataset = test_dataset
    else:
        dataset = test_dataloader

    eval_accuracy = 0
    nb_eval_steps = 0
    for batch in tqdm(dataset):
        if type(batch[0]) == dict:
            b_input_ids = batch[0]
            if "pixel_values" in b_input_ids:
                b_input_ids["pixel_values"] = b_input_ids["pixel_values"].to(device)
            else:
                b_input_ids = {k: v.to(device) for k, v in b_input_ids.items()}

        else:
            b_input_ids = batch[0].to(device)

        if type(batch[1]) == int:
            b_labels = torch.tensor([batch[1]]).to(device)
        else:
            b_labels = batch[1].to(device)

        if args.model == "ALIGN":
            classes_processed.to(device)
            b_input_ids.update(classes_processed)

        with torch.no_grad():
            if args.model == "RESNET":
                outputs = model(b_input_ids)
            else:
                outputs = model(**b_input_ids)

        if args.model == "RESNET":
            logits = outputs
        else:
            try:
                logits = outputs.logits
            except:
                logits_per_image = outputs.logits_per_image

                # we can take the softmax to get the label probabilities
                logits = logits_per_image.softmax(dim=1)

        tmp_eval_accuracy = torch.sum(torch.argmax(logits, dim=1) == b_labels) / len(
            b_labels
        )

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
        if nb_eval_steps % 100 == 0:
            tqdm.write("Accuracy: {}".format(eval_accuracy / nb_eval_steps))

    print("Accuracy: {}".format(eval_accuracy / nb_eval_steps))
