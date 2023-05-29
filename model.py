import torch
import torch.nn as nn
from transformers import (
    ViTForImageClassification,
    CvtForImageClassification,
    Swinv2ForImageClassification,
    ViTHybridForImageClassification,
    ViTMSNForImageClassification,
)

class ImageClassifier(nn.Module):
    def __init__(self, num_classes, model, config):
        super(ImageClassifier, self).__init__()
        self.model = model["class"].from_pretrained(
            model["pretrained"],
            config=config,
            ignore_mismatched_sizes=True,
        )

        # freeze the model except the last layer
        if model["freeze"]:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True

        print("Model initialized")
        print(
            "Number of trainable parameters: ",
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        )

    def forward(self, **out):
        outputs = self.model(**out)
        return outputs
