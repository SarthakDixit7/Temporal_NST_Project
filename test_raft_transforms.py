import torch
from torchvision.models.optical_flow import Raft_Large_Weights

weights = Raft_Large_Weights.DEFAULT
transforms = weights.transforms()

# Test with random [0,1] tensor like ToTensor() produces
img1 = torch.rand(1, 3, 256, 256)
img2 = torch.rand(1, 3, 256, 256)

img1_t, img2_t = transforms(img1, img2)
print("Input shape:", img1.shape, "range:", img1.min().item(), img1.max().item())
print("Output shape:", img1_t.shape, "range:", img1_t.min().item(), img1_t.max().item())
