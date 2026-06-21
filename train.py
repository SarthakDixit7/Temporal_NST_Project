
import os
import re

os.environ["TORCH_HOME"] = "/mnt/fast0/sd2885/cache/"

import argparse
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader

from config import CHECKPOINT_ROOT, DATA_ROOT
from data.dataset import CocoDataset, StyleDataset
from models.transform_net import TransformNetwork
from models.vgg_loss import VGG19LossNetwork, gram_matrix


def make_dataloaders(
    coco_root,
    style_root,
    batch_size=4,
    num_workers=0,
):
    coco_ds = CocoDataset(coco_root)
    style_ds = StyleDataset(style_root)

    coco_loader = DataLoader(
        coco_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    style_loader = DataLoader(
        style_ds, batch_size=1, shuffle=True, num_workers=num_workers, drop_last=False
    )
    return coco_loader, style_loader


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
    content_weight=1.0,
    style_weight=1000000.0,
    tv_weight=1e-4,
    checkpoint_interval=500,
    checkpoint_dir="checkpoints",
    resume_checkpoint=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    transform_net = TransformNetwork().to(device)

    global_step = 0
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        transform_net.load_state_dict(torch.load(resume_path, map_location=device))
        match = re.search(r"transform_net_iter_(\d+)\.pth$", resume_path.name)
        if match is None:
            raise ValueError(
                f"Could not infer training step from checkpoint filename: {resume_path.name}"
            )
        global_step = int(match.group(1))
        print(f"Resuming training from checkpoint {resume_path} at step {global_step}")

    vgg = VGG19LossNetwork().to(device)
    vgg.eval()

    for param in vgg.parameters():
        param.requires_grad = False

    optimizer = Adam(transform_net.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    coco_loader, style_loader = make_dataloaders(
        coco_root=dataset_dir, style_root=style_dir, batch_size=batch_size, num_workers=num_workers
    )
    style_iter = cycle(style_loader)

    transform_net.train()

    for epoch in range(epochs):
        for content_images in coco_loader:
            style_images = next(style_iter)

            content_images = content_images.to(device)
            style_images = style_images.to(device)

            optimizer.zero_grad(set_to_none=True)

            generated_images = transform_net(content_images)

            # Shift the [-1.0, 1.0] Tanh output to [0.0, 1.0] to match target distributions
            mapped_generated_images = (generated_images + 1.0) / 2.0

            with torch.no_grad():
                content_target, _ = vgg(content_images.float())
                _, style_targets = vgg(style_images.float())

            # Pass the mapped images to VGG so loss computations are calculated on identical ranges
            generated_content, generated_styles = vgg(mapped_generated_images.float())

            content_loss = mse_loss(generated_content, content_target)

            style_loss = torch.zeros((), device=device, dtype=torch.float32)
            style_layers = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
            for layer_name in style_layers:
                generated_gram = gram_matrix(generated_styles[layer_name])
                target_gram = gram_matrix(style_targets[layer_name])
                style_loss = style_loss + mse_loss(generated_gram, target_gram)

            total_variation = tv_loss(mapped_generated_images.float())
            loss = (content_weight * content_loss) + (style_weight * style_loss) + (tv_weight * total_variation)

            loss.backward()
            optimizer.step()

            if global_step % 10 == 0:
                first_layer_weight_sum = next(transform_net.parameters()).sum().item()
                print(f"Step {global_step} - Weight Signature: {first_layer_weight_sum:.6f}")

            global_step += 1

            if checkpoint_interval and global_step % checkpoint_interval == 0:
                save_checkpoint(transform_net, checkpoint_dir, global_step)

            print(
                f"epoch={epoch + 1} step={global_step} "
                f"content={content_loss.item():.4f} "
                f"style={(style_weight * style_loss).item():.4f} "
                f"tv={total_variation.item():.4f} total={loss.item():.4f}"
            )

    save_checkpoint(transform_net, checkpoint_dir, global_step)
    return transform_net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural style transfer model.")
    parser.add_argument("--dataset_dir", type=str, default=DATA_ROOT, help="Path to the COCO dataset.")
    parser.add_argument(
        "--style_dir",
        type=str,
        default="/mnt/fast0/sd2885/data/my_style/",
        help="Path to the style dataset (WikiArt).",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--content_weight", type=float, default=1.0, help="Weight for content loss.")
    parser.add_argument("--style_weight", type=float, default=1000000.0, help="Weight for style loss.")
    parser.add_argument("--tv_weight", type=float, default=1e-4, help="Weight for total variation loss.")
    parser.add_argument(
        "--checkpoint_interval", type=int, default=500, help="Steps between saving checkpoints."
    )
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_ROOT, help="Directory to save checkpoints.")
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to a .pth checkpoint to resume training from.",
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
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        resume_checkpoint=args.resume_checkpoint,
    )

