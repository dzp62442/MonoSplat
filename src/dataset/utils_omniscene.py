import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image


def _ensure_hwc3(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[2] == 3:
        return image
    if image.shape[2] == 1:
        return np.concatenate([image, image, image], axis=2)
    if image.shape[2] == 4:
        color = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:].astype(np.float32) / 255.0
        blended = color * alpha + 255.0 * (1.0 - alpha)
        return blended.clip(0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported image shape {image.shape}")


def load_info(info: dict) -> tuple[str, np.ndarray, np.ndarray]:
    """Extract image path and camera transforms from a bin entry."""
    img_path = info["data_path"]
    c2w = info["sensor2lidar_transform"]

    lidar2cam_r = np.linalg.inv(info["sensor2lidar_rotation"])
    lidar2cam_t = info["sensor2lidar_translation"] @ lidar2cam_r.T
    w2c = np.eye(4)
    w2c[:3, :3] = lidar2cam_r.T
    w2c[3, :3] = -lidar2cam_t
    return img_path, c2w, w2c


def load_conditions(
    img_paths: Sequence[str],
    resolution: Sequence[int],
    *,
    is_input: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load RGB images, masks, and normalized intrinsics."""

    def maybe_resize(image: Image.Image, ck: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image.height == resolution[0] and image.width == resolution[1]:
            return np.array(image), ck
        fx, fy, cx, cy = ck[0, 0], ck[1, 1], ck[0, 2], ck[1, 2]
        scale_h = resolution[0] / image.height
        scale_w = resolution[1] / image.width
        ck = np.array(
            [
                [fx * scale_w, 0, cx * scale_w],
                [0, fy * scale_h, cy * scale_h],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        resized = image.resize((resolution[1], resolution[0]))
        return np.array(resized), ck

    imgs, masks, intrinsics = [], [], []
    for img_path in img_paths:
        param_path = (
            img_path.replace("samples", "samples_param_small")
            .replace("sweeps", "sweeps_param_small")
            .replace(".jpg", ".json")
        )
        with Path(param_path).open("r") as f:
            params = json.load(f)
        ck = np.array(params["camera_intrinsic"], dtype=np.float32)

        image_path = (
            img_path.replace("samples", "samples_small")
            .replace("sweeps", "sweeps_small")
        )
        image = Image.open(image_path)
        image_np, ck = maybe_resize(image, ck)
        ck[0, :] = ck[0, :] / resolution[1]
        ck[1, :] = ck[1, :] / resolution[0]

        imgs.append(_ensure_hwc3(image_np))
        intrinsics.append(ck)

        if is_input:
            masks.append(np.ones(resolution, dtype=bool))
        else:
            mask_path = (
                image_path.replace("sweeps_small", "sweeps_mask_small")
                .replace("samples_small", "samples_mask_small")
                .replace(".jpg", ".png")
            )
            mask_image = Image.open(mask_path).convert("L")
            if mask_image.size != (resolution[1], resolution[0]):
                mask_image = mask_image.resize((resolution[1], resolution[0]), Image.BILINEAR)
            mask = np.array(mask_image).astype(np.float32) / 255.0
            masks.append(mask > 0.5)

    imgs_tensor = (
        torch.from_numpy(np.stack(imgs, axis=0))
        .permute(0, 3, 1, 2)
        .float()
        / 255.0
    )
    masks_tensor = torch.from_numpy(np.stack(masks, axis=0)).bool()
    intrinsics_tensor = torch.as_tensor(np.stack(intrinsics, axis=0), dtype=torch.float32)

    return imgs_tensor, masks_tensor, intrinsics_tensor
