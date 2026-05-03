"""Loads and transforms data for the pix2pix model."""

import os

import PIL.Image as Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

from config import Config, DataConfig


class MapDataset(Dataset):
    """Dataset class for maps data where each file contains sat|map halves."""

    def __init__(self, image_dir: str, is_train: bool, data_config: DataConfig):
        super().__init__()
        self.is_train = is_train
        self.data_config = data_config
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.image_paths = sorted(
            [
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if f.lower().endswith((".jpg", ".png"))
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def _paired_transform(self, sat: Image.Image, road_map: Image.Image):
        """Apply identical deterministic preprocessing to both sat/map halves."""
        out_size = (self.data_config.img_size, self.data_config.img_size)
        sat = TF.resize(sat, out_size, interpolation=transforms.InterpolationMode.BICUBIC)
        road_map = TF.resize(road_map, out_size, interpolation=transforms.InterpolationMode.BICUBIC)
        sat = self.normalize(self.to_tensor(sat))
        road_map = self.normalize(self.to_tensor(road_map))
        return sat, road_map

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        # Keep paired alignment: split first, then apply the same resize/normalize path.
        sat, road_map = create_pairs(image)
        sat, road_map = self._paired_transform(sat, road_map)
        return sat, road_map


def create_pairs(img: Image.Image):
    """Split image into satellite and road map halves."""
    img_w, img_h = img.size
    half_w = img_w // 2
    return img.crop((0, 0, half_w, img_h)), img.crop((half_w, 0, img_w, img_h))


def build_train_and_val_dataloaders(config: Config):
    cuda_available = torch.cuda.is_available()
    g = torch.Generator()
    g.manual_seed(config.data.seed)
    train_dataset = MapDataset(config.data.train_dir, True, config.data)
    val_dataset = MapDataset(config.data.val_dir, False, config.data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        generator=g,
        pin_memory=cuda_available,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=cuda_available,
    )
    return train_loader, val_loader


def build_eval_dataloader(config: Config):
    """Build an evaluation loader for val or an explicit held-out directory."""
    cuda_available = torch.cuda.is_available()
    eval_dir = config.data.eval_dir or config.data.val_dir
    eval_dataset = MapDataset(eval_dir, False, config.data)
    return DataLoader(
        eval_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=cuda_available,
    )


