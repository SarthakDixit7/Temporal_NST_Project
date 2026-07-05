import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from models.transform_net import TransformNetwork

def resize_longest_edge(img, max_size=1024):
    """Scale image so its longest side is at most `max_size` pixels."""
    width, height = img.size
    longest = max(width, height)
    if longest <= max_size:
        return img
    scale = max_size / longest
    new_width = int(width * scale)
    new_height = int(height * scale)
    return img.resize((new_width, new_height), Image.BICUBIC)

def run_realesrgan(input_path, output_path):
    """
    Placeholder for Real-ESRGAN upscaling.
    In a real implementation, you would use:
    
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    ... initialize model and upsample ...
    """
    print(f"[*] Placeholder: Upscaling {input_path} with Real-ESRGAN -> {output_path}")
    print("[*] To enable Real-ESRGAN, please install the required packages (realesrgan, basicsr) and integrate the API.")
    pass

def main():
    parser = argparse.ArgumentParser(description="High Resolution Inference Pipeline (Downscale -> Stylize -> Upscale)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained TransformNetwork checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save final output")
    parser.add_argument("--max_size", type=int, default=1024, help="Maximum size for longest edge before stylization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Image and Downscale
    print(f"[*] Loading image: {args.image_path}")
    img = Image.open(args.image_path).convert("RGB")
    original_size = img.size
    print(f"[*] Original size: {original_size}")
    
    img = resize_longest_edge(img, max_size=args.max_size)
    print(f"[*] Resized to: {img.size} for TransformNetwork")
    
    # Transform to tensor (C, H, W) and add batch dim
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    
    # 2. Load TransformNetwork and Stylize
    print(f"[*] Loading TransformNetwork from {args.checkpoint}")
    transform_net = TransformNetwork().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    transform_net.load_state_dict(state_dict)
    transform_net.eval()
    
    with torch.no_grad():
        print("[*] Running stylization...")
        output = transform_net(img_tensor)
        
        # Shift Tanh output [-1.0, 1.0] back to [0.0, 1.0] and clamp
        output = (output + 1.0) / 2.0
        output = torch.clamp(output, 0.0, 1.0)
    
    # Save intermediate stylized output
    # We strip out the batch dimension
    output_image = output.squeeze(0)
    
    temp_output_path = args.output_path.replace(".", "_stylized_temp.")
    # If there was no extension in output_path, just append
    if temp_output_path == args.output_path:
        temp_output_path += "_stylized_temp.png"
        
    save_image(output_image, temp_output_path)
    print(f"[*] Saved intermediate stylized image to: {temp_output_path}")
    
    # 3. Upscale with Real-ESRGAN (Placeholder)
    print("[*] Passing to Real-ESRGAN for upscaling...")
    run_realesrgan(temp_output_path, args.output_path)
    
    print("[*] Inference high res pipeline complete.")

if __name__ == "__main__":
    main()