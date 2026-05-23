import torch.nn as nn
import timm


class MobileNetV4ShipClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        model_name = "hf_hub:timm/mobilenetv4_conv_small.e2400_r224_in1k"

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)