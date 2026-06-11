from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data.dataset import CocoDataset, StyleDataset
from models.transform_net import TransformNetwork
from models.vgg_loss import VGG19LossNetwork, gram_matrix


def make_dataloaders(
    coco_root="data/train2014",
    wikiart_root="data/wikiart_hf",
    batch_size=4,
    num_workers=0,
):
    coco_ds = CocoDataset(coco_root)
    style_ds = StyleDataset(wikiart_root)

    coco_loader = DataLoader(coco_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    style_loader = DataLoader(style_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
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
    epochs=1,
    batch_size=4,
    num_workers=0,
    content_weight=1.0,
    style_weight=10.0,
    tv_weight=1e-6,
    checkpoint_interval=500,
    checkpoint_dir="checkpoints",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform_net = TransformNetwork().to(device)
    vgg = VGG19LossNetwork().to(device)
    vgg.eval()

    for param in vgg.parameters():
        param.requires_grad = False

    optimizer = Adam(transform_net.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    coco_loader, style_loader = make_dataloaders(batch_size=batch_size, num_workers=num_workers)
    style_iter = cycle(style_loader)

    global_step = 0
    transform_net.train()

    for epoch in range(epochs):
        for content_images in coco_loader:
            style_images = next(style_iter)

            content_images = content_images.to(device)
            style_images = style_images.to(device)

            optimizer.zero_grad(set_to_none=True)

            generated_images = transform_net(content_images)

            with torch.no_grad():
                content_target, _ = vgg(content_images)
                style_content_target, style_targets = vgg(style_images)

            generated_content, generated_styles = vgg(generated_images)

            content_loss = mse_loss(generated_content, content_target)

            style_loss = 0.0
            style_layers = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
            for layer_name in style_layers:
                generated_gram = gram_matrix(generated_styles[layer_name])
                target_gram = gram_matrix(style_targets[layer_name])
                style_loss = style_loss + mse_loss(generated_gram, target_gram)

            total_variation = tv_loss(generated_images)
            loss = (content_weight * content_loss) + (style_weight * style_loss) + (tv_weight * total_variation)

            loss.backward()
            optimizer.step()

            global_step += 1

            if checkpoint_interval and global_step % checkpoint_interval == 0:
                save_checkpoint(transform_net, checkpoint_dir, global_step)

            print(
                f"epoch={epoch + 1} step={global_step} "
                f"content={content_loss.item():.4f} style={style_loss.item():.4f} "
                f"tv={total_variation.item():.4f} total={loss.item():.4f}"
            )

    save_checkpoint(transform_net, checkpoint_dir, global_step)
    return transform_net


if __name__ == "__main__":
    train()
