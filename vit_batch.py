from transformers import AutoTokenizer, AlignProcessor
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import ViTForImageClassification, AutoImageProcessor, ViTConfig
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import torch.multiprocessing as mp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# for each image, we need to pass the image through the processor
# and get the image features before passing it to the model
# so we can create a transform function to do this
def transform_function(image):
    image_processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    image = image_processor(image, return_tensors="pt")
    image["pixel_values"] = image["pixel_values"].squeeze()
    return image


def label_transform_function(label):
    return {"labels": torch.tensor(label)}


class vitClassification(nn.Module):
    def __init__(self, num_classes):
        super(vitClassification, self).__init__()
        self.model = ViTForImageClassification(config)

    def forward(self, **out):
        outputs = self.model(**out)
        return outputs


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    transform_processor = transforms.Compose([transform_function])

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

    batch_size = 16

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
    config.num_labels = 100

    model = vitClassification(100)
    model.to(device)

    epochs = 5

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)

    for epoch in tqdm(range(epochs)):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)

            labels["labels"] = labels["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                pixel_values=images["pixel_values"], labels=labels["labels"]
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (i + 1) % 100 / batch_size == 0 or i+1==2:
                # print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1} * {batch_size}/{len(train_dataset)}], Loss: {loss.item():.4f}')
                # display loss, learning rate, steps and epoch
                tqdm.write(
                    f"Epoch [{epoch+1}/{epochs}], Step [{(i+1) * batch_size}/{len(train_dataset)}], Loss: {loss.item():.4f}, Learning rate: {scheduler.get_last_lr()[0]:.7f}"
                )
            

    # save the model
    torch.save(model.state_dict(), "vit_cifar100.pth")

    # # evaluate the model
    # model.eval()
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for images, labels in tqdm(test_loader):
    #         images = images.to(device)
    #         labels["labels"] = labels["labels"].to(device)
    #         outputs = model(pixel_values=images["pixel_values"], labels=labels["labels"])
    #         _, predicted = torch.max(outputs.logits.data, 1)
    #         total += labels["labels"].size(0)
    #         correct += (predicted == labels["labels"]).sum().item()

    #     print(f"Accuracy of the model on the test images: {100 * correct / total}%")
