import os

os.environ["TORCH_HOME"] = "/mnt/fast0/sd2885/cache/"

import argparse
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import CHECKPOINT_ROOT, STYLE_ROOT, VIDEO_ROOT
from data.dataset import StyleDataset, VideoDataset
from models.temporal import compute_occlusion_mask, load_frozen_raft, warp_image
from models.transform_net import TransformNetwork
from models.vgg_loss import VGG19LossNetwork, gram_matrix


def make_dataloaders(
    video_root,
    style_root,
    batch_size=4,
    num_workers=0,
):
    video_ds = VideoDataset(video_root)
    style_ds = StyleDataset(style_root)

    video_loader = DataLoader(
        video_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    style_loader = DataLoader(
        style_ds, batch_size=1, shuffle=True, num_workers=num_workers, drop_last=False
    )
    return video_loader, style_loader


def tv_loss(image_batch):
    horizontal = torch.mean(torch.abs(image_batch[:, :, :, :-1] - image_batch[:, :, :, 1:]))
    vertical = torch.mean(torch.abs(image_batch[:, :, :-1, :] - image_batch[:, :, 1:, :]))
    return horizontal + vertical


def save_checkpoint(model, checkpoint_dir, iteration):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"transform_net_iter_{iteration}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def train(
    dataset_dir,
    style_dir,
    epochs=1,
    batch_size=8,
    num_workers=4,
    content_weight=1000.0,
    style_weight=1e10,
    tv_weight=1e-3,
    temporal_weight=1000.0,
    checkpoint_interval=500,
    checkpoint_dir="checkpoints",
    style_mode=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    transform_net = TransformNetwork().to(device)
    vgg = VGG19LossNetwork().to(device)
    raft = load_frozen_raft().to(device)

    vgg.eval()
    raft.eval()

    for param in vgg.parameters():
        param.requires_grad = False

    optimizer = Adam(transform_net.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss(reduction="mean")

    video_loader, style_loader = make_dataloaders(
        video_root=dataset_dir, style_root=style_dir, batch_size=batch_size, num_workers=num_workers
    )
    style_iter = cycle(style_loader)

    transform_net.train()

    if style_mode == 'pointillism':
        tv_weight = 0.0
        style_layers = ["relu1_2", "relu2_2", "relu3_3"]
    elif style_mode == 'impressionism':
        tv_weight = 1e-3
        style_layers = ["relu1_2", "relu2_2", "relu3_3", "relu4_1", "relu5_1"]
    else:
        style_layers = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]

    global_step = 0

    for epoch in range(epochs):
        for frame_t_minus_1, frame_t in video_loader:
            style_images = next(style_iter)

            frame_t_minus_1 = frame_t_minus_1.to(device)
            frame_t = frame_t.to(device)
            style_images = style_images.to(device)
            if style_images.shape[0] == 1 and frame_t.shape[0] > 1:
                style_images = style_images.expand(frame_t.shape[0], -1, -1, -1).contiguous()

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                with autocast(enabled=use_amp):
                    flow_forward = raft(frame_t_minus_1 * 255.0, frame_t * 255.0)[-1]
                    flow_backward = raft(frame_t * 255.0, frame_t_minus_1 * 255.0)[-1]
                    occlusion_mask = compute_occlusion_mask(flow_forward, flow_backward)

            with autocast(enabled=use_amp):
                y_prev = transform_net(frame_t_minus_1)
                y_curr = transform_net(frame_t)
                warped_y_prev = warp_image(y_prev, flow_forward)

                content_target, _ = vgg(frame_t)
                _, style_targets = vgg(style_images)
                stylized_curr_for_vgg = (y_curr + 1.0) / 2.0
                generated_content, generated_styles = vgg(stylized_curr_for_vgg)

            content_loss = mse_loss(generated_content.float(), content_target.float())

            style_loss = torch.zeros((), device=device, dtype=torch.float32)
            for layer_name in style_layers:
                generated_gram = gram_matrix(generated_styles[layer_name])
                target_gram = gram_matrix(style_targets[layer_name])
                style_loss = style_loss + mse_loss(generated_gram.float(), target_gram.float())

            tv_penalty = tv_loss(stylized_curr_for_vgg.float()) 
            temporal_loss = mse_loss(
                occlusion_mask.float() * y_curr.float(),
                occlusion_mask.float() * warped_y_prev.float(),
            )

            loss = (
                (content_weight * content_loss)
                + (style_weight * style_loss)
                + (tv_weight * tv_penalty)
                + (temporal_weight * temporal_loss)
            )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1

            if checkpoint_interval and global_step % checkpoint_interval == 0:
                save_checkpoint(transform_net, checkpoint_dir, global_step)

            # Calculate weighted components for logging
            weighted_content = content_weight * content_loss.item()
            weighted_style = style_weight * style_loss.item()
            weighted_tv = tv_weight * tv_penalty.item()
            weighted_temporal = temporal_weight * temporal_loss.item()

            print(
                f"epoch={epoch + 1} step={global_step} "
                f"content={weighted_content:.4f} "
                f"style={weighted_style:.4f} "
                f"tv={weighted_tv:.4f} "
                f"temporal={weighted_temporal:.4f} "
                f"total={loss.item():.4f}"
            )

    save_checkpoint(transform_net, checkpoint_dir, global_step)
    return transform_net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a temporal neural style transfer model.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=VIDEO_ROOT,
        help="Path to the sequential video frame dataset (MPI Sintel).",
    )
    parser.add_argument(
        "--style_dir",
        type=str,
        default=STYLE_ROOT,
        help="Path to the style dataset (WikiArt).",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--content_weight", type=float, default=10000.0, help="Weight for content loss.")
    parser.add_argument("--style_weight", type=float, default=1e10, help="Weight for style loss.")
    parser.add_argument("--tv_weight", type=float, default=1e-3, help="Weight for total variation loss.")
    parser.add_argument(
        "--temporal_weight", type=float, default=1e6, help="Weight for temporal consistency loss."
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=500, help="Steps between saving checkpoints."
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=CHECKPOINT_ROOT, help="Directory to save checkpoints."
    )
    parser.add_argument(
        "--style_mode", type=str, choices=["pointillism", "impressionism"], default=None,
        help="Dynamic configuration for specific artistic styles."
    )

    args = parser.parse_args()
    train(
        dataset_dir=args.dataset_dir,
        style_dir=args.style_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        temporal_weight=args.temporal_weight,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        style_mode=args.style_mode,
    )