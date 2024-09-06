import torch.nn as nn
import torch.nn.functional as functional
import torchvision.models as models

from transformers import RobertaConfig
from transformers import RobertaModel


class ImageEncoder(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        if pretrained:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            self.resnet = models.resnet50()

        self.resnet.fc = nn.Linear(in_features=2048, out_features=768)

    def forward(self, images):
        return functional.normalize(self.resnet(images))


class TextEncoder(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        if pretrained:
            self.roberta = RobertaModel.from_pretrained('weights/develop/roberta')
        else:
            self.roberta = RobertaModel(RobertaConfig.from_pretrained('weights/develop/roberta'))

    def forward(self, texts, masks):
        return functional.normalize(self.roberta(texts, masks).pooler_output)


class ContrastModel(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained)
        self.text_encoder = TextEncoder(pretrained)

    def forward(self, images, texts, masks):
        return self.image_encoder(images), self.text_encoder(texts, masks)

    def encode_image(self, images):
        return self.image_encoder(images)

    def encode_text(self, texts, masks):
        return self.text_encoder(texts, masks)
