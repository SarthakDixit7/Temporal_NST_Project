import torch
from pathlib import Path
import lpips

from models.transform_net import TransformNetwork
from data.dataset import VideoDataset
from evaluate import to_lpips_range

@torch.no_grad()
def main():
    device = torch.device('cuda')
    baseline_net = TransformNetwork().to(device)
    baseline_net.load_state_dict(torch.load('/mnt/fast0/sd2885/checkpoints_baseline/transform_net_iter_79800.pth', map_location=device)['model_state_dict'])
    baseline_net.eval()
    
    temporal_net = TransformNetwork().to(device)
    temporal_net.load_state_dict(torch.load('/mnt/fast0/sd2885/checkpoints/transform_net_iter_78000.pth', map_location=device)['model_state_dict'])
    temporal_net.eval()
    
    dataset = VideoDataset('/mnt/fast0/sd2885/data/sintel/training/final')
    loss_fn = lpips.LPIPS(net="vgg").to(device)
    loss_fn.eval()
    
    print("Pair | Seq | Frame | Base_LPIPS | Temp_LPIPS")
    for i in range(10):
        prev_path, curr_path = dataset.pairs[i]
        seq = Path(curr_path).parent.name
        frame = Path(curr_path).stem
        
        _, frame_t = dataset[i]
        frame_t = frame_t.unsqueeze(0).to(device)
        
        base_stylized = baseline_net(frame_t)
        temp_stylized = temporal_net(frame_t)
        
        orig_t_lpips = to_lpips_range(frame_t.float())
        
        base_lpips = loss_fn(base_stylized.float(), orig_t_lpips).mean().item()
        temp_lpips = loss_fn(temp_stylized.float(), orig_t_lpips).mean().item()
        
        print(f"{i:4d} | {seq:6s} | {frame:10s} | {base_lpips:.6f} | {temp_lpips:.6f}")

if __name__ == "__main__":
    main()
