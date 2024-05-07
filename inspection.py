import torch

x = torch.cuda.mem_get_info()
print(x)

truth = torch.cuda.is_available()
print("Cuda available: ", truth)

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import MaskRCNN

backbone  = resnet_fpn_backbone("resnet101", pretrained=True, trainable_layers=3)
model = MaskRCNN(backbone=backbone, num_classes=2, min_size=256, max_size=256)

model = model.cuda()
model.train()
x = torch.rand(3, 3, 256, 256).to("cuda")

try:
    out = model(x)
    print("Bravo, model run successfully!")
except:
    print("Model not successfully run!")
