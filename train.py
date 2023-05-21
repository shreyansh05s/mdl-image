from transformers import AutoTokenizer, AlignProcessor
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import (
    ViTForImageClassification,
    AutoImageProcessor,
    ViTConfig,
    CvtConfig,
    CvtForImageClassification,
    AutoFeatureExtractor,
    Swinv2ForImageClassification,
    Swinv2Config,
    ViTHybridForImageClassification,
    ViTHybridConfig,
    ViTHybridImageProcessor
)
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import torch.multiprocessing as mp
import wandb
from sam import SAM
import argparse
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models = {
    "vit": {
        "class": ViTForImageClassification,
        "config": ViTConfig,
        "pretrained": "google/vit-base-patch16-224-in21k",
    },
    "cvt": {
        "class": CvtForImageClassification,
        "config": CvtConfig,
        "pretrained": "microsoft/cvt-21-384-22k",
    },
    "swin": {
        "class": Swinv2ForImageClassification,
        "config": Swinv2Config,
        "pretrained": "microsoft/swinv2-large-patch4-window12-192-22k",
    },
    "vit_hybrid": {
        "class": ViTHybridForImageClassification,
        "config": ViTHybridConfig,
        "pretrained": "google/vit-hybrid-base-bit-384"
    }
}

# fix seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


# for each image, we need to pass the image through the processor
# and get the image features before passing it to the model
# so we can create a transform function to do this
class TransformImage:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image):
        image = self.image_processor(image, return_tensors="pt")
        image["pixel_values"] = image["pixel_values"].squeeze()
        return image


def label_transform_function(label):
    return {"labels": torch.tensor(label)}


class ImageClassifier(nn.Module):
    def __init__(self, num_classes, args):
        super(ImageClassifier, self).__init__()
        self.model = models[args.model]["class"].from_pretrained(
            models[args.model]["pretrained"],
            config=config,
            ignore_mismatched_sizes=True,
        )

        print("Model initialized")
        print(
            "Number of trainable parameters: ",
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        )

    def forward(self, **out):
        outputs = self.model(**out)
        return outputs


# funtction to select the optimizer
def get_optimizer(method="adam", lr=2e-4, **kargs):
    if method == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # elif method == "LION":
    #     optimizer = Lion(model.parameters(), lr=lr, weight_decay=1e-2)
    elif method == "sam":
        base_optimizer = (
            torch.optim.SGD
        )  # define an optimizer for the "sharpness-aware" update
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            lr=lr,
            adaptive=True,
            momentum=0.9,
            weight_decay=0.0005,
            rho=2.0,
        )
    return optimizer


def get_scheduler(method="step", args=None):
    if method == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    elif method == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    return scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    # parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", type=str, default="step")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="sam")
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--model", type=str, default="vit")

    args = parser.parse_args()

    wandb.init(project="vit-cifar100")
    wandb.config.update(args)

    # get run name from wandb
    run_name = wandb.run.name

    model_dir = os.path.join("models", args.model, run_name)

    # create a directory with the run name to save the model
    os.mkdir(model_dir)

    dir_name = model_dir

    mp.set_start_method("spawn", force=True)

    # transform = transforms.Compose(
    #     [
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(size=32, padding=4),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         transform_function,
    #     ]
    # )

    if args.model == "cvt":
        image_processor = AutoFeatureExtractor.from_pretrained(
            "microsoft/cvt-21-384-22k"
        )
    elif args.model == "vit_hybrid":
        image_processor = ViTHybridImageProcessor.from_pretrained(
            "google/vit-hybrid-base-bit-384"
        )
    else:
        image_processor = AutoImageProcessor.from_pretrained(
            models[args.model]["pretrained"]
        )
    model = models[args.model]["processor"] = image_processor

    transform_image = TransformImage(image_processor=image_processor)

    transform_processor = transforms.Compose([transform_image])

    train_dataset = CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=transform_processor,
        target_transform=label_transform_function,
    )
    test_dataset = CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=transform_processor,
        target_transform=label_transform_function,
    )

    batch_size = args.batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    config = models[args.model]["config"].from_pretrained(
        models[args.model]["pretrained"]
    )
    config.num_labels = 100

    model = ImageClassifier(100, args)
    model.to(device)

    epochs = args.epochs

    optimizer = get_optimizer(method=args.optimizer, lr=args.lr)

    scheduler = get_scheduler(method=args.scheduler, args=args)

    # progress bar
    pbar = tqdm(range(epochs))

    for epoch in pbar:
        if args.train:
            model.train()

            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)

                labels["labels"] = labels["labels"].to(device)

                optimizer.zero_grad()

                outputs = model(
                    pixel_values=images["pixel_values"], labels=labels["labels"]
                )

                loss = outputs.loss
                loss.backward()
                if args.optimizer == "sam":
                    optimizer.first_step(zero_grad=True)
                    model(
                        pixel_values=images["pixel_values"], labels=labels["labels"]
                    ).loss.backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.step()
                scheduler.step()

                wandb.log(
                    {
                        "loss": loss.item(),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "epoch": epoch + 1,
                        "step": i + 1,
                    }
                )

                if (
                    (i + 1) % 100 / batch_size == 0
                    or i + 1 == 2
                    or i + 1 == len(train_loader)
                ):
                    pbar.set_description(
                        f"Epoch {epoch+1}, Step {i+1}, Loss: {loss.item():.4f}, Learning rate: {scheduler.get_last_lr()[0]:.7f}"
                    )

        print("Evaluating the model")

        model.eval()
        total = 0
        correct = 0
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels["labels"] = labels["labels"].to(device)
            with torch.no_grad():
                outputs = model(
                    pixel_values=images["pixel_values"], labels=labels["labels"]
                )
            # add validation loss to wandb
            # wandb.log({"val_loss": outputs.loss.item()})
            tmp_eval_accuracy = torch.sum(
                torch.argmax(outputs.logits, dim=1) == labels["labels"]
            ) / labels["labels"].size(0)

            total += 1
            correct += tmp_eval_accuracy

        tqdm.write(
            f"Accuracy of the model on the test images: {100 * correct / total}%"
        )

        # save the model after each epoch
        torch.save(
            model.state_dict(),
            "{}/{}_cifar100_epoch_{}.pth".format(dir_name, args.model, epoch + 1),
        )
        wandb.log({"accuracy": 100 * correct / total})

    # save the model
    torch.save(model.state_dict(), "{}/{}_cifar100.pth".format(dir_name, args.model))
    # wandb.save("vit_cifar100.pth")
    wandb.finish()
