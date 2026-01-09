import copy
import json
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from einops import repeat
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler
from .utils_omniscene import load_conditions, load_info


@dataclass
class DatasetOmniSceneCfg(DatasetCfgCommon):
    name: Literal["omniscene"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    test_times_per_scene: int
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    train_times_per_scene: int = 1
    highres: bool = False


class DatasetOmniScene(Dataset):
    """Dataset loader for the OmniScene (nuScenes derivative) benchmark.

    The loader mimics the implementation used in depthsplat: each bin token
    contains six surround-view cameras, where the key-frame images act as
    context views and the subsequent frames (indices 1 and 2) serve as targets.
    """

    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    data_version: str = "interp_12Hz_trainval"
    dataset_prefix: str = "/datasets/nuScenes"

    def __init__(
        self,
        cfg: DatasetOmniSceneCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        load_rel_depth: bool | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler

        self.data_root = Path(cfg.roots[0])
        self.resolution = tuple(cfg.image_shape)

        self.near = cfg.near if cfg.near != -1 else 0.1
        self.far = cfg.far if cfg.far != -1 else 1000.0

        self.load_rel_depth = stage == "test" if load_rel_depth is None else load_rel_depth
        if stage != "test":
            self.load_rel_depth = False

        self.bin_tokens = self._load_bin_tokens(stage)

    def _load_bin_tokens(self, stage: Stage) -> list[str]:
        bins_path = self.data_root / self.data_version
        if stage == "train":
            file_name = "bins_train_3.2m.json"
            with (bins_path / file_name).open("r") as f:
                return json.load(f)["bins"]
        if stage == "val":
            file_name = "bins_val_3.2m.json"
            with (bins_path / file_name).open("r") as f:
                tokens = json.load(f)["bins"]
            # Match depthsplat's visualization subset: every 3000-th entry from the first 30k bins.
            return tokens[:30000:3000][:10]
        if stage == "test":
            file_name = "bins_val_3.2m.json"
            with (bins_path / file_name).open("r") as f:
                tokens = json.load(f)["bins"]
            # Mini-test split used during training: every 14th entry, capped at 2048 bins.
            return tokens[0::14][:2048]
        raise ValueError(f"Unsupported stage {stage} for OmniScene dataset.")

    def __len__(self) -> int:
        return len(self.bin_tokens)

    def __getitem__(self, index: int):
        bin_token = self.bin_tokens[index]
        bin_path = (
            self.data_root
            / self.data_version
            / "bin_infos_3.2m"
            / f"{bin_token}.pkl"
        )
        with bin_path.open("rb") as f:
            bin_info = pkl.load(f)

        sensor_info_center = {
            sensor: bin_info["sensor_info"][sensor][0]
            for sensor in self.camera_types + ["LIDAR_TOP"]
        }

        # Key-frame inputs.
        input_img_paths, input_c2ws = [], []
        for cam in self.camera_types:
            info = copy.deepcopy(sensor_info_center[cam])
            img_path, c2w, _ = load_info(info)
            img_path = img_path.replace(self.dataset_prefix, str(self.data_root))
            input_img_paths.append(img_path)
            input_c2ws.append(c2w)
        input_c2ws = torch.as_tensor(input_c2ws, dtype=torch.float32)

        input_imgs, input_masks, input_intrinsics, input_rel_depths = load_conditions(
            input_img_paths,
            self.resolution,
            is_input=True,
            load_rel_depth=self.load_rel_depth,
        )

        # Additional render views from non-key frames.
        output_img_paths, output_c2ws = [], []
        frame_num = len(bin_info["sensor_info"]["LIDAR_TOP"])
        if frame_num < 3:
            raise ValueError(f"Bin {bin_token} only contains {frame_num} frames.")

        render_indices = [[1, 2]] * len(self.camera_types)
        for cam_id, cam in enumerate(self.camera_types):
            for frame_index in render_indices[cam_id]:
                info = copy.deepcopy(bin_info["sensor_info"][cam][frame_index])
                img_path, c2w, _ = load_info(info)
                img_path = img_path.replace(self.dataset_prefix, str(self.data_root))
                output_img_paths.append(img_path)
                output_c2ws.append(c2w)
        output_c2ws = torch.as_tensor(output_c2ws, dtype=torch.float32)

        output_imgs, output_masks, output_intrinsics, output_rel_depths = load_conditions(
            output_img_paths,
            self.resolution,
            is_input=False,
            load_rel_depth=self.load_rel_depth,
        )

        # Append the input views to the targets for supervision and visualization.
        output_imgs = torch.cat([output_imgs, input_imgs], dim=0)
        output_masks = torch.cat([output_masks, input_masks], dim=0)
        output_c2ws = torch.cat([output_c2ws, input_c2ws], dim=0)
        output_intrinsics = torch.cat([output_intrinsics, input_intrinsics], dim=0)
        if output_rel_depths is not None and input_rel_depths is not None:
            output_rel_depths = torch.cat(
                [output_rel_depths, input_rel_depths], dim=0
            )

        context = {
            "extrinsics": input_c2ws,
            "intrinsics": input_intrinsics,
            "image": input_imgs,
            "near": repeat(
                torch.tensor(self.near, dtype=torch.float32),
                "-> v",
                v=len(input_c2ws),
            ),
            "far": repeat(
                torch.tensor(self.far, dtype=torch.float32),
                "-> v",
                v=len(input_c2ws),
            ),
            "index": torch.arange(len(input_c2ws), dtype=torch.int64),
        }

        target = {
            "extrinsics": output_c2ws,
            "intrinsics": output_intrinsics,
            "image": output_imgs,
            "near": repeat(
                torch.tensor(self.near, dtype=torch.float32),
                "-> v",
                v=len(output_c2ws),
            ),
            "far": repeat(
                torch.tensor(self.far, dtype=torch.float32),
                "-> v",
                v=len(output_c2ws),
            ),
            "index": torch.arange(len(output_c2ws), dtype=torch.int64),
            "masks": output_masks,
        }
        if output_rel_depths is not None:
            target["rel_depth"] = output_rel_depths

        return {
            "context": context,
            "target": target,
            "scene": bin_token,
        }
