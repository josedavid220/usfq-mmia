import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
)


class ObjectCountingModel(nn.Module):
    def __init__(self, num_classes=2, version="v2"):  # [0 background, 1 person]
        super(ObjectCountingModel, self).__init__()

        if version == "v2":
            self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights)
        else:
            self.weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
            self.model = fasterrcnn_resnet50_fpn(weights=self.weights)

        for param in self.model.parameters():
            param.requires_grad_(False)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        self.model.roi_heads.box_predictor = (
            models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        )

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return self.model(images, targets)
        return self.model(images)
