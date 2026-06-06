"""
Self-contained inference script for ReSplat on COLMAP-processed datasets.

Usage with --data_dir + --scene_name (for datasets with scene subdirectories):
    python scripts/infer_colmap.py \
        --model_preset dl3dv_8v_512x960 \
        --data_dir path/to/colmap_data \
        --scene_name SCENE_NAME \
        --output_dir path/to/output \
        --save_images --save_video --save_ply

Usage with --scene_path (single COLMAP scene directory):
    # 8-view high-res model
    python scripts/infer_colmap.py --model_preset dl3dv_8v_512x960 \
        --scene_path path/to/colmap_scene \
        --output_dir path/to/output --save_images --save_ply

    # 16-view high-res model
    python scripts/infer_colmap.py --model_preset dl3dv_16v_540x960 \
        --scene_path path/to/colmap_scene \
        --output_dir path/to/output --save_images --save_ply

Usage without presets (manual config):
    python scripts/infer_colmap.py \
        --scene_path path/to/colmap_scene \
        --checkpoint pretrained/resplat-base-dl3dv-512x960-view8-8179ed87.pth \
        --experiment dl3dv \
        --num_context 8 --num_refine 4 --max_resolution 960 \
        --output_dir path/to/output \
        --save_images --save_ply

Available presets:
    dl3dv_8v_512x960      - 8-view base model, high-res (recommended)
    dl3dv_16v_540x960     - 16-view base model, high-res
    dl3dv_8v_256x448      - 8-view base model, low-res
    dl3dv_16v_256x448     - 16-view base model, low-res
    dl3dv_32v_256x448     - 32-view base model, low-res
    dl3dv_8v_256x448_small - 8-view small (ViT-S) model
    dl3dv_8v_256x448_large - 8-view large (ViT-L) model, init-only
"""

import argparse
import collections
import json
import os
import struct
import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torchvision.transforms as tf
from PIL import Image
from src.misc.stablize_camera import render_stabilization_path

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")


# =============================================================================
# Model Presets
# =============================================================================

MODEL_PRESETS = {
    # === High-resolution models (Phase 2+) ===
    "dl3dv_8v_512x960": {
        "overrides": [],
        "num_context": 8,
        "num_refine": 4,
        "max_resolution": 960,
        "checkpoint": "pretrained/resplat-base-dl3dv-512x960-view8-8179ed87.pth",
    },
    "dl3dv_16v_540x960": {
        "overrides": [
            "model.encoder.gaussian_adapter.gaussian_scale_max=3.",
            "model.encoder.depth_pred_half_res=true",
            "model.encoder.no_crop_image=true",
        ],
        "num_context": 16,
        "num_refine": 2,
        "max_resolution": 960,
        "checkpoint": "pretrained/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth",
    },
    # === Low-resolution models (Phase 1) ===
    "dl3dv_8v_256x448": {
        "overrides": [],
        "num_context": 8,
        "num_refine": 4,
        "max_resolution": 448,
        "checkpoint": "pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth",
    },
    "dl3dv_16v_256x448": {
        "overrides": [],
        "num_context": 16,
        "num_refine": 4,
        "max_resolution": 448,
        "checkpoint": "pretrained/resplat-base-dl3dv-256x448-view16-f38bf984.pth",
    },
    "dl3dv_32v_256x448": {
        "overrides": [],
        "num_context": 32,
        "num_refine": 4,
        "max_resolution": 448,
        "checkpoint": "pretrained/resplat-base-dl3dv-256x448-view32-439b63a6.pth",
    },
    # === Small/Large backbone variants ===
    "dl3dv_8v_256x448_small": {
        "overrides": [
            "model.encoder.monodepth_vit_type=vits",
            "model.encoder.gaussian_regressor_channels=256",
        ],
        "num_context": 8,
        "num_refine": 4,
        "max_resolution": 448,
        "checkpoint": "pretrained/resplat-small-dl3dv-256x448-view8-548993fe.pth",
    },
    "dl3dv_8v_256x448_large": {
        "overrides": [
            "model.encoder.monodepth_vit_type=vitl",
            "model.encoder.gaussian_regressor_channels=768",
        ],
        "num_context": 8,
        "num_refine": 0,
        "max_resolution": 448,
        "checkpoint": "pretrained/resplat-large-dl3dv-256x448-view8-62f1703a.pth",
    },
}


# =============================================================================
# COLMAP Loading Functions
# =============================================================================

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
ColmapCamera = collections.namedtuple(
    "ColmapCamera", ["id", "model", "width", "height", "params"]
)
BaseImage = collections.namedtuple(
    "ColmapImage",
    ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"],
)

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)


class ColmapImage(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_extrinsics_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = ColmapImage(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def read_extrinsics_text(path):
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [
                        tuple(map(float, elems[0::3])),
                        tuple(map(float, elems[1::3])),
                    ]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = ColmapImage(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def read_intrinsics_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params
            )
            cameras[camera_id] = ColmapCamera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def read_intrinsics_text(path):
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = ColmapCamera(
                    id=camera_id,
                    model=model,
                    width=width,
                    height=height,
                    params=params,
                )
    return cameras


# =============================================================================
# Scene Loading
# =============================================================================


def load_colmap_scene(scene_path, sparse_dir="sparse/0", images_dir="images"):
    """Load COLMAP reconstruction and convert to ReSplat format.

    Returns dict with:
        image_paths: list[str] - full paths to images
        image_names: list[str] - image filenames
        c2w: np.ndarray [N, 4, 4] - camera-to-world matrices (OpenCV convention)
        intrinsics: np.ndarray [N, 3, 3] - normalized intrinsic matrices
        image_sizes: list[tuple[int, int]] - (width, height) per image
    """
    sparse_path = os.path.join(scene_path, sparse_dir)

    # Try binary format first, fall back to text
    cameras_bin = os.path.join(sparse_path, "cameras.bin")
    cameras_txt = os.path.join(sparse_path, "cameras.txt")
    images_bin = os.path.join(sparse_path, "images.bin")
    images_txt = os.path.join(sparse_path, "images.txt")

    if os.path.exists(cameras_bin) and os.path.exists(images_bin):
        cam_intrinsics = read_intrinsics_binary(cameras_bin)
        cam_extrinsics = read_extrinsics_binary(images_bin)
        print(f"Loaded COLMAP binary format from {sparse_path}")
    elif os.path.exists(cameras_txt) and os.path.exists(images_txt):
        cam_intrinsics = read_intrinsics_text(cameras_txt)
        cam_extrinsics = read_extrinsics_text(images_txt)
        print(f"Loaded COLMAP text format from {sparse_path}")
    else:
        raise FileNotFoundError(
            f"No COLMAP reconstruction found in {sparse_path}. "
            "Expected cameras.bin/txt and images.bin/txt"
        )

    images_root = os.path.join(scene_path, images_dir)

    # Sort images by name for deterministic ordering
    sorted_images = sorted(cam_extrinsics.values(), key=lambda x: x.name)

    image_paths = []
    image_names = []
    c2w_list = []
    intrinsics_list = []
    image_sizes = []
    point3D_ids_list = []

    for img in sorted_images:
        # Check image exists
        img_path = os.path.join(images_root, img.name)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found, skipping: {img_path}")
            continue

        # Get camera intrinsics
        cam = cam_intrinsics[img.camera_id]

        # Only support PINHOLE and SIMPLE_PINHOLE (undistorted images)
        if cam.model == "PINHOLE":
            fx, fy, cx, cy = cam.params[:4]
        elif cam.model == "SIMPLE_PINHOLE":
            f, cx, cy = cam.params[:3]
            fx = fy = f
        else:
            print(
                f"Warning: Unsupported camera model '{cam.model}' for image "
                f"{img.name}, skipping. Only PINHOLE and SIMPLE_PINHOLE are "
                f"supported. Please undistort images first using COLMAP."
            )
            continue

        # Build normalized intrinsic matrix (fx/width, fy/height, cx/width, cy/height)
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = fx / cam.width
        K[1, 1] = fy / cam.height
        K[0, 2] = cx / cam.width
        K[1, 2] = cy / cam.height

        # Build W2C matrix from COLMAP qvec + tvec
        # COLMAP: qvec is world-to-camera rotation, tvec is translation in camera frame
        R = qvec2rotmat(img.qvec)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = img.tvec

        # Invert to get C2W (camera-to-world)
        # COLMAP is already in OpenCV convention (Y-down, Z-forward)
        c2w = np.linalg.inv(w2c).astype(np.float32)

        # Extract valid point3D_ids (filter out -1 which means unmatched)
        valid_pts = img.point3D_ids[img.point3D_ids >= 0]

        image_paths.append(img_path)
        image_names.append(img.name)
        c2w_list.append(c2w)
        intrinsics_list.append(K)
        image_sizes.append((cam.width, cam.height))
        point3D_ids_list.append(set(valid_pts.tolist()))

    if len(image_paths) == 0:
        raise RuntimeError(f"No valid images found in {images_root}")

    return {
        "image_paths": image_paths,
        "image_names": image_names,
        "c2w": np.stack(c2w_list, axis=0),  # [N, 4, 4]
        "intrinsics": np.stack(intrinsics_list, axis=0),  # [N, 3, 3]
        "image_sizes": image_sizes,
        "point3D_ids": point3D_ids_list,  # list of sets, one per image
    }


# =============================================================================
# Frame Subsetting
# =============================================================================


def subset_scene_data(scene_data, start_frame, frame_distance):
    """Subset scene_data to frames [start_frame, start_frame + frame_distance).

    Returns a new scene_data dict with only the selected frames.
    Indices are 0-based in the returned data (transparent to downstream code).
    """
    end_frame = min(start_frame + frame_distance, len(scene_data["image_paths"]))
    indices = list(range(start_frame, end_frame))

    return {
        "image_paths": [scene_data["image_paths"][i] for i in indices],
        "image_names": [scene_data["image_names"][i] for i in indices],
        "c2w": scene_data["c2w"][indices],
        "intrinsics": scene_data["intrinsics"][indices],
        "image_sizes": [scene_data["image_sizes"][i] for i in indices],
        "point3D_ids": [scene_data["point3D_ids"][i] for i in indices],
    }


# =============================================================================
# View Selection
# =============================================================================


def farthest_point_sample(xyz, npoint):
    """Farthest point sampling on camera positions.

    Adapted from src/dataset/view_sampler/view_sampler_bounded_v2.py

    Args:
        xyz: [B, N, 3] point cloud data
        npoint: number of samples
    Returns:
        centroids: [B, npoint] sampled indices
    """
    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10

    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    barycenter = torch.sum(xyz, 1)
    barycenter = barycenter / xyz.shape[1]
    barycenter = barycenter.view(B, 1, 3)

    dist = torch.sum((xyz - barycenter) ** 2, -1)
    farthest = torch.max(dist, 1)[1]

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def select_context_views(c2w, num_context, strategy="fps"):
    """Select context view indices from all available views.

    Args:
        c2w: [N, 4, 4] camera-to-world matrices
        num_context: number of context views to select
        strategy: "fps" | "uniform"
    Returns:
        context_indices: sorted numpy array of indices
    """
    N = len(c2w)
    if num_context >= N:
        return np.arange(N)

    if strategy == "fps":
        positions = torch.tensor(c2w[:, :3, 3], dtype=torch.float32).unsqueeze(0)
        indices = farthest_point_sample(positions, num_context)[0].numpy()
        return np.sort(indices)
    elif strategy == "uniform":
        return np.linspace(0, N - 1, num_context, dtype=int)
    else:
        raise ValueError(f"Unknown context selection strategy: {strategy}")


def select_target_views(num_images, context_indices, target_selection="remaining",
                        num_target=None):
    """Select target views for rendering.

    Args:
        num_images: total number of images
        context_indices: indices used as context
        target_selection: "remaining" | "all"
        num_target: optional max number of target views
    Returns:
        target_indices: numpy array of indices
    """
    if target_selection == "remaining":
        all_indices = np.arange(num_images)
        target_indices = np.setdiff1d(all_indices, context_indices)
    elif target_selection == "all":
        target_indices = np.arange(num_images)
    else:
        raise ValueError(f"Unknown target selection: {target_selection}")

    if num_target is not None and len(target_indices) > num_target:
        step = len(target_indices) / num_target
        selected = [int(i * step) for i in range(num_target)]
        target_indices = target_indices[selected]

    return target_indices


# =============================================================================
# Image Loading and Preprocessing
# =============================================================================


def compute_target_shape(orig_h, orig_w, max_resolution=960, image_shape=None):
    """Compute target image shape, ensuring divisibility by 64.

    The encoder requires H and W to be divisible by
    shim_patch_size * downscale_factor = 16 * 4 = 64.
    """
    DIVISOR = 64

    if image_shape is not None:
        h, w = image_shape
    else:
        scale = max_resolution / max(orig_h, orig_w)
        if scale < 1.0:
            h = int(orig_h * scale)
            w = int(orig_w * scale)
        else:
            h, w = orig_h, orig_w

    # Round down to nearest multiple of DIVISOR
    h = (h // DIVISOR) * DIVISOR
    w = (w // DIVISOR) * DIVISOR

    assert h > 0 and w > 0, (
        f"Resolution too small after rounding to multiple of {DIVISOR}: "
        f"{h}x{w}. Increase --max_resolution."
    )
    return h, w


def load_and_preprocess_images(image_paths, target_h, target_w):
    """Load images and resize to target resolution.

    Args:
        image_paths: list of image file paths
        target_h, target_w: target dimensions (must be divisible by 64)
    Returns:
        images: [V, 3, H, W] float32 tensor in [0, 1]
    """
    to_tensor = tf.ToTensor()
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = img.resize((target_w, target_h), Image.LANCZOS)
        images.append(to_tensor(img))
    return torch.stack(images)


# =============================================================================
# Pose Utilities
# =============================================================================


def camera_normalization(pivotal_pose, poses):
    """Align all poses relative to a reference pose.

    Adapted from src/dataset/dataset_dl3dv.py:camera_normalization

    Args:
        pivotal_pose: [1, 4, 4] reference camera pose
        poses: [N, 4, 4] all camera poses
    Returns:
        normalized poses [N, 4, 4]
    """
    camera_norm_matrix = torch.inverse(pivotal_pose)
    poses = torch.bmm(camera_norm_matrix.repeat(poses.shape[0], 1, 1), poses)
    return poses


# =============================================================================
# Model Construction
# =============================================================================


def build_model(experiment, checkpoint, num_refine, image_shape, overrides,
                device, no_strict_load=True):
    """Build model using Hydra compose API and load checkpoint.

    Returns: (encoder, decoder, data_shim) all on device
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # Clear any existing Hydra state
    GlobalHydra.instance().clear()

    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        hydra_overrides = [
            f"+experiment={experiment}",
            "mode=test",
            f"model.encoder.num_refine={num_refine}",
            f"dataset.image_shape=[{image_shape[0]},{image_shape[1]}]",
            f"dataset.ori_image_shape=[{image_shape[0]},{image_shape[1]}]",
            f"output_dir=outputs/colmap_inference",
        ]
        hydra_overrides.extend(overrides)
        cfg_dict = compose(config_name="main", overrides=hydra_overrides)

    # Import src modules (outside beartype hook is fine for inference)
    from src.config import load_typed_root_config
    from src.dataset.data_module import get_data_shim
    from src.global_cfg import set_cfg
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper

    set_cfg(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)

    # Build encoder and decoder
    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)

    # Build ModelWrapper to load checkpoint with correct key prefixes
    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        None,  # encoder_visualizer
        decoder,
        [],  # losses (not needed for inference)
        None,  # step_tracker
    )

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model_wrapper.load_state_dict(ckpt, strict=not no_strict_load)
    model_wrapper = model_wrapper.to(device).eval()

    data_shim = get_data_shim(model_wrapper.encoder)

    return model_wrapper.encoder, model_wrapper.decoder, data_shim


# =============================================================================
# Batch Construction
# =============================================================================


def move_to_device(data, device):
    """Recursively move tensors in a nested dict to device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    return data


def build_batch(
    context_images,
    target_images,
    context_c2w,
    target_c2w,
    context_K,
    target_K,
    near,
    far,
    scene_name,
    device,
):
    """Build a batch dict matching ReSplat's BatchedExample format.

    All tensors get batch dimension B=1. Poses are aligned to the middle
    context view.
    """
    Vc = len(context_c2w)
    Vt = len(target_c2w)

    # Align all poses to middle context view
    all_c2w = torch.cat([context_c2w, target_c2w], dim=0)
    mid_idx = Vc // 2
    all_c2w = camera_normalization(context_c2w[mid_idx : mid_idx + 1], all_c2w)
    context_c2w_aligned = all_c2w[:Vc]
    target_c2w_aligned = all_c2w[Vc:]

    batch = {
        "context": {
            "image": context_images.unsqueeze(0),  # [1, Vc, 3, H, W]
            "extrinsics": context_c2w_aligned.unsqueeze(0),  # [1, Vc, 4, 4]
            "intrinsics": context_K.unsqueeze(0),  # [1, Vc, 3, 3]
            "near": torch.full((1, Vc), near),
            "far": torch.full((1, Vc), far),
            "index": torch.arange(Vc).unsqueeze(0),
        },
        "target": {
            "image": target_images.unsqueeze(0),  # [1, Vt, 3, H, W]
            "extrinsics": target_c2w_aligned.unsqueeze(0),  # [1, Vt, 4, 4]
            "intrinsics": target_K.unsqueeze(0),  # [1, Vt, 3, 3]
            "near": torch.full((1, Vt), near),
            "far": torch.full((1, Vt), far),
            "index": torch.arange(Vc, Vc + Vt).unsqueeze(0),
        },
        "scene": [scene_name],
    }

    return move_to_device(batch, device)


# =============================================================================
# Inference
# =============================================================================


@torch.no_grad()
def run_inference(encoder, decoder, batch, num_refine, render_chunk_size=10,
                  save_depth=False):
    """Run ReSplat inference.

    Following the pattern from src/model/model_wrapper.py test_step.

    Returns:
        gaussians: Gaussians dataclass
        rendered: [Vt, 3, H, W] rendered images tensor
        visualization_dump: dict with depth data (if save_depth=True), else None
    """
    _, _, _, h, w = batch["target"]["image"].shape

    visualization_dump = {} if save_depth else None

    # 1. Initial forward pass (encoder)
    print("Running encoder forward pass...")
    gaussians_out = encoder(batch["context"], global_step=0, deterministic=False,
                            visualization_dump=visualization_dump)

    if isinstance(gaussians_out, dict):
        condition_features = gaussians_out.get("condition_features", None)
        gaussians = gaussians_out["gaussians"]
    else:
        gaussians = gaussians_out
        condition_features = None

    # 2. Refinement (if enabled)
    if num_refine > 0 and condition_features is not None:
        print(f"Running refinement ({num_refine} iterations)...")
        refine_output = encoder.forward_update(
            batch["context"],
            batch["target"],
            condition_features,
            gaussians,
            decoder,
            None,  # context_remain
        )
        gaussians = refine_output["gaussian"][-1]

    # 3. Render target views (chunked for memory)
    Vt = batch["target"]["extrinsics"].shape[1]
    print(f"Rendering {Vt} target views...")
    all_colors = []
    all_depths = []
    for i in range(0, Vt, render_chunk_size):
        end = min(i + render_chunk_size, Vt)
        output = decoder.forward(
            gaussians,
            batch["target"]["extrinsics"][:, i:end],
            batch["target"]["intrinsics"][:, i:end],
            batch["target"]["near"][:, i:end],
            batch["target"]["far"][:, i:end],
            (h, w),
            depth_mode=None,
        )
        all_colors.append(output.color[0])  # [chunk, 3, H, W]
        all_depths.append(output.depth[0])  # [chunk, H, W]

    rendered = torch.cat(all_colors, dim=0)  # [Vt, 3, H, W]
    rendered_depth = torch.cat(all_depths, dim=0)  # [Vt, H, W]
    print(f"Rendered {Vt} views at {h}x{w}")

    return gaussians, rendered, rendered_depth, visualization_dump


# =============================================================================
# Output Saving
# =============================================================================


def save_outputs(
    rendered, gaussians, batch, output_dir, image_names,
    save_images=True, save_ply=False, save_depth=False,
    context_image_names=None, context_images=None,
    visualization_dump=None, rendered_depth=None, max_save_images=10,
):
    """Save rendered images, depth maps, and Gaussians (PLY)."""
    os.makedirs(output_dir, exist_ok=True)

    # Save rendered images (at most max_save_images, evenly spaced)
    if save_images:
        images_dir = os.path.join(output_dir, "rendered")
        os.makedirs(images_dir, exist_ok=True)
        total = len(rendered)
        if max_save_images > 0 and total > max_save_images:
            save_indices = np.linspace(0, total - 1, max_save_images, dtype=int)
        else:
            save_indices = range(total)
        for i in save_indices:
            img_np = (rendered[i].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(
                np.uint8
            )
            stem = Path(image_names[i]).stem
            Image.fromarray(img_np).save(os.path.join(images_dir, f"{stem}.png"))
        print(f"Saved {len(save_indices)} rendered images to {images_dir}")

    # Save context (input) images for reference
    if save_images and context_images is not None and context_image_names is not None:
        input_dir = os.path.join(output_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        for img_tensor, name in zip(context_images, context_image_names):
            img_np = (img_tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(
                np.uint8
            )
            stem = Path(name).stem
            Image.fromarray(img_np).save(os.path.join(input_dir, f"{stem}.png"))
        print(f"Saved {len(context_images)} input images to {input_dir}")

    # Save input view depth maps
    if save_depth and visualization_dump is not None:
        from src.visualization.vis_depth import viz_depth_tensor

        # Latent-resolution depth (used for Gaussian unprojection)
        if "depth" in visualization_dump:
            depth_dir = os.path.join(output_dir, "depth")
            os.makedirs(depth_dir, exist_ok=True)

            # [B, V, H, W, srf, s] -> [V, H, W]
            depth = visualization_dump["depth"][0, :, :, :, 0, 0].cpu().detach()

            for depth_i, name in zip(depth, context_image_names or []):
                stem = Path(name).stem
                depth_viz = viz_depth_tensor(1.0 / depth_i, return_numpy=True)
                Image.fromarray(depth_viz).save(os.path.join(depth_dir, f"{stem}.png"))

            print(f"Saved {len(depth)} latent-res depth maps to {depth_dir}")

        # Full-resolution depth (before latent downsampling)
        if "depth_fullres" in visualization_dump:
            depth_fullres_dir = os.path.join(output_dir, "depth_fullres")
            os.makedirs(depth_fullres_dir, exist_ok=True)

            # [B, V, H, W] -> [V, H, W]
            depth_fullres = visualization_dump["depth_fullres"][0].cpu().detach()

            for depth_i, name in zip(depth_fullres, context_image_names or []):
                stem = Path(name).stem
                depth_viz = viz_depth_tensor(1.0 / depth_i, return_numpy=True)
                Image.fromarray(depth_viz).save(
                    os.path.join(depth_fullres_dir, f"{stem}.png")
                )

            print(f"Saved {len(depth_fullres)} full-res depth maps to {depth_fullres_dir}")

    # Save rendered (target view) depth maps
    if save_depth and rendered_depth is not None:
        from src.visualization.vis_depth import viz_depth_tensor

        rendered_depth_dir = os.path.join(output_dir, "rendered_depth")
        os.makedirs(rendered_depth_dir, exist_ok=True)
        total = len(rendered_depth)
        if max_save_images > 0 and total > max_save_images:
            save_indices = np.linspace(0, total - 1, max_save_images, dtype=int)
        else:
            save_indices = range(total)
        for i in save_indices:
            depth_i = rendered_depth[i].cpu().detach()
            stem = Path(image_names[i]).stem
            depth_viz = viz_depth_tensor(1.0 / depth_i, return_numpy=True)
            Image.fromarray(depth_viz).save(os.path.join(rendered_depth_dir, f"{stem}.png"))
        print(f"Saved {len(save_indices)} rendered depth maps to {rendered_depth_dir}")

    # Save PLY
    if save_ply:
        try:
            from src.model.ply_export import export_ply

            ply_path = Path(output_dir) / "gaussians.ply"
            Vc = batch["context"]["extrinsics"].shape[1]
            mid_idx = Vc // 2
            export_ply(
                batch["context"]["extrinsics"][0, mid_idx],
                gaussians.means[0],
                gaussians.scales[0],
                gaussians.rotations[0],
                gaussians.harmonics[0],
                gaussians.opacities[0],
                ply_path,
                align_to_view=True,
            )
            print(f"Saved Gaussian PLY to {ply_path}")
        except Exception as e:
            print(f"Warning: Failed to save PLY: {e}")


# =============================================================================
# Smooth Video Rendering
# =============================================================================


@torch.no_grad()
def render_smooth_video(
    gaussians, decoder, all_c2w_np, context_c2w, intrinsic_np,
    near, far, image_shape, output_dir,
    render_chunk_size=10, fps=30, smooth_kernel=45,
    device="cuda",
):
    """Render a smooth video by smoothing all scene poses and rendering each frame.

    Args:
        gaussians: Gaussians dataclass from inference
        decoder: the gsplat decoder
        all_c2w_np: numpy [N, 4, 4] all scene camera-to-world poses in order
        context_c2w: torch [Vc, 4, 4] unnormalized context poses (for normalization pivot)
        intrinsic_np: numpy [3, 3] single normalized intrinsic matrix
        near, far: float near/far plane distances
        image_shape: (H, W) target image resolution
        output_dir: directory to save the video
        render_chunk_size: number of views to render at once
        fps: video frame rate
        smooth_kernel: Gaussian kernel size for trajectory smoothing
        device: torch device
    """
    import imageio

    N = len(all_c2w_np)
    h, w = image_shape
    print(f"Rendering smooth video with {N} frames (kernel={smooth_kernel})...")

    # 1. Smooth the trajectory
    poses_3x4 = all_c2w_np[:, :3, :]  # [N, 3, 4]
    smoothed_list = render_stabilization_path(poses_3x4, k_size=smooth_kernel)

    # Reconstruct [N, 4, 4] from smoothed [3, 4] matrices
    smoothed_c2w = np.zeros((N, 4, 4), dtype=np.float32)
    for i, pose in enumerate(smoothed_list):
        smoothed_c2w[i, :3, :] = pose
        smoothed_c2w[i, 3, 3] = 1.0

    # 2. Normalize poses using the same pivot as build_batch()
    smoothed_c2w_t = torch.tensor(smoothed_c2w, dtype=torch.float32)
    mid_idx = len(context_c2w) // 2
    smoothed_c2w_aligned = camera_normalization(
        context_c2w[mid_idx:mid_idx + 1], smoothed_c2w_t
    )

    # 3. Prepare intrinsics (single intrinsic for all frames)
    intrinsic_t = torch.tensor(intrinsic_np, dtype=torch.float32)

    # 4. Render in chunks
    all_frames = []
    for i in range(0, N, render_chunk_size):
        end = min(i + render_chunk_size, N)
        chunk_size = end - i
        extrinsics = smoothed_c2w_aligned[i:end].unsqueeze(0).to(device)  # [1, chunk, 4, 4]
        intrinsics = intrinsic_t.unsqueeze(0).unsqueeze(0).expand(1, chunk_size, -1, -1).to(device)  # [1, chunk, 3, 3]
        near_t = torch.full((1, chunk_size), near, device=device)
        far_t = torch.full((1, chunk_size), far, device=device)

        output = decoder.forward(
            gaussians, extrinsics, intrinsics, near_t, far_t,
            (h, w), depth_mode=None,
        )
        # output.color[0] is [chunk, 3, H, W]
        for img_tensor in output.color[0]:
            img_np = (img_tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            all_frames.append(img_np)

    # 5. Write video
    video_path = os.path.join(output_dir, "video.mp4")
    imageio.mimwrite(video_path, all_frames, fps=fps, quality=8)
    print(f"Saved smooth video ({N} frames) to {video_path}")


# =============================================================================
# Evaluation Metrics
# =============================================================================


@torch.no_grad()
def compute_metrics(rendered, target_images, target_names, output_dir, device="cuda:0",
                    chunk_size=4):
    """Compute PSNR, SSIM, LPIPS between rendered and ground truth target views.

    Args:
        rendered: [Vt, 3, H, W] rendered images tensor (on GPU)
        target_images: [Vt, 3, H, W] ground truth images tensor
        target_names: list of image names for per-view results
        output_dir: directory to save metrics.json
        device: device for LPIPS computation
        chunk_size: chunk size for LPIPS to avoid OOM
    Returns:
        dict with mean metrics
    """
    from src.evaluation.metrics import compute_psnr, compute_ssim, compute_lpips

    rendered = rendered.clamp(0, 1)
    target_images = target_images.clamp(0, 1).to(rendered.device)

    Vt = rendered.shape[0]

    # Compute PSNR and SSIM (lightweight, can do all at once)
    psnr_vals = compute_psnr(target_images, rendered)  # [Vt]
    ssim_vals = compute_ssim(target_images, rendered)  # [Vt]

    # Compute LPIPS in chunks (VGG network uses memory)
    lpips_vals = []
    for i in range(0, Vt, chunk_size):
        end = min(i + chunk_size, Vt)
        lpips_chunk = compute_lpips(target_images[i:end], rendered[i:end])
        lpips_vals.append(lpips_chunk)
    lpips_vals = torch.cat(lpips_vals, dim=0)  # [Vt]

    # Aggregate
    mean_psnr = psnr_vals.mean().item()
    mean_ssim = ssim_vals.mean().item()
    mean_lpips = lpips_vals.mean().item()

    # Build per-view results
    per_view = []
    for i in range(Vt):
        per_view.append({
            "name": target_names[i],
            "psnr": round(psnr_vals[i].item(), 4),
            "ssim": round(ssim_vals[i].item(), 4),
            "lpips": round(lpips_vals[i].item(), 4),
        })

    results = {
        "mean": {
            "psnr": round(mean_psnr, 3),
            "ssim": round(mean_ssim, 3),
            "lpips": round(mean_lpips, 3),
        },
        "per_view": per_view,
    }

    # Save to JSON
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Evaluation Metrics ({Vt} target views):")
    print(f"    PSNR:  {mean_psnr:.3f}")
    print(f"    SSIM:  {mean_ssim:.3f}")
    print(f"    LPIPS: {mean_lpips:.3f}")
    print(f"  Saved to {metrics_path}")

    return results


# =============================================================================
# Main
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="ReSplat inference on COLMAP-processed datasets"
    )

    # Scene specification (either --scene_path OR --data_dir + --scene_name/--scene_list)
    parser.add_argument(
        "--scene_path",
        type=str,
        default=None,
        help="Path to COLMAP scene directory (must contain sparse/0/ and images/)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Base directory containing scene subdirectories "
             "(e.g., datasets/dl3dv-evaluation/images_tar)",
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default=None,
        help="Scene name (hash). Used with --data_dir to construct scene_path",
    )
    parser.add_argument(
        "--scene_list",
        type=str,
        default=None,
        help="Path to text file with one scene name per line, or 'all' for all "
             "subdirs in --data_dir. Used with --data_dir for batch processing",
    )

    # Frame range (for long video scenes)
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="First frame index in the evaluation subset",
    )
    parser.add_argument(
        "--frame_distance",
        type=int,
        default=60,
        help="Number of frames from start_frame to include",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pretrained model .pth file (required unless --model_preset is used)",
    )

    # Model preset
    parser.add_argument(
        "--model_preset",
        type=str,
        default=None,
        choices=list(MODEL_PRESETS.keys()),
        help="Model preset that auto-sets checkpoint, overrides, num_context, "
             "num_refine, and max_resolution. Explicit CLI args override preset values.",
    )

    # Model config
    parser.add_argument(
        "--experiment",
        type=str,
        default="dl3dv",
        help="Hydra experiment config name (dl3dv or re10k), must match checkpoint",
    )
    parser.add_argument(
        "--num_refine",
        type=int,
        default=None,
        help="Number of refinement iterations (0=init only). Default: from preset, or 4",
    )

    # View selection
    parser.add_argument(
        "--num_context",
        type=int,
        default=None,
        help="Number of context (input) views.",
    )
    parser.add_argument(
        "--num_target",
        type=int,
        default=None,
        help="Number of target views to render. Default: all non-context views",
    )
    parser.add_argument(
        "--context_selection",
        type=str,
        default="fps",
        choices=["fps", "uniform"],
        help="Strategy for selecting context views: fps (farthest point sampling) or uniform",
    )
    parser.add_argument(
        "--target_selection",
        type=str,
        default="remaining",
        choices=["remaining", "all"],
        help="Strategy for selecting target views",
    )

    # Resolution
    parser.add_argument(
        "--max_resolution",
        type=int,
        default=None,
        help="Max resolution on the longer side. Default: from preset, or 960",
    )
    parser.add_argument(
        "--image_shape",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Explicit image shape [H W], overrides --max_resolution",
    )

    # Near/far planes
    parser.add_argument("--near", type=float, default=0.01, help="Near plane distance")
    parser.add_argument("--far", type=float, default=200.0, help="Far plane distance")

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/colmap_inference",
        help="Output directory for results",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        default=True,
        help="Save rendered images (default: True)",
    )
    parser.add_argument(
        "--no_save_images",
        action="store_true",
        default=False,
        help="Disable saving rendered images",
    )
    parser.add_argument(
        "--save_video", action="store_true", default=False, help="Save rendered video"
    )
    parser.add_argument(
        "--save_depth",
        action="store_true",
        default=False,
        help="Save depth maps (input view encoder depth + rendered target view depth)",
    )
    parser.add_argument(
        "--save_ply",
        action="store_true",
        default=False,
        help="Save Gaussians as PLY file",
    )
    parser.add_argument(
        "--render_chunk_size",
        type=int,
        default=10,
        help="Number of views to render at once (for memory management)",
    )
    parser.add_argument(
        "--smooth_video_kernel",
        type=int,
        default=45,
        help="Gaussian kernel size for camera trajectory smoothing in smooth video",
    )
    parser.add_argument(
        "--smooth_video_fps",
        type=int,
        default=30,
        help="FPS for smooth video output",
    )
    parser.add_argument(
        "--max_save_images",
        type=int,
        default=10,
        help="Max number of evenly-spaced frames to save as images from smooth video (0=none)",
    )

    # COLMAP paths
    parser.add_argument(
        "--sparse_dir",
        type=str,
        default=None,
        help="Relative path from scene to COLMAP sparse reconstruction. "
             "Default: 'sparse/0'",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Relative path from scene to images directory. "
             "Default: 'images_4'",
    )

    # Misc
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run on"
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Additional Hydra config overrides (e.g., model.encoder.monodepth_vit_type=vitl)",
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        default=False,
        help="Skip evaluation metrics (PSNR/SSIM/LPIPS) on target views",
    )

    return parser.parse_args()


def run_scene(args, scene_path, scene_name, output_dir,
              encoder=None, decoder=None, data_shim=None):
    """Run inference on a single COLMAP scene.

    If encoder/decoder/data_shim are None, builds the model (first scene).
    Returns (encoder, decoder, data_shim) so they can be reused across scenes.
    """
    save_images = args.save_images and not args.no_save_images

    # 1. Load COLMAP scene
    print(f"\n[1/6] Loading COLMAP scene from {scene_path}...")
    scene_data = load_colmap_scene(
        scene_path, args.sparse_dir, args.images_dir
    )
    num_total = len(scene_data["image_paths"])
    print(f"  Found {num_total} images")

    # Subset to frame range if specified
    if args.start_frame is not None:
        scene_data = subset_scene_data(
            scene_data, args.start_frame, args.frame_distance
        )
        num_total = len(scene_data["image_paths"])
        print(f"  Frame range [{args.start_frame}, {args.start_frame + args.frame_distance})"
              f" -> {num_total} images")

    # 2. Compute target resolution
    print(f"\n[2/6] Computing target resolution...")
    first_img = Image.open(scene_data["image_paths"][0])
    orig_w, orig_h = first_img.size
    target_h, target_w = compute_target_shape(
        orig_h, orig_w, args.max_resolution, args.image_shape
    )
    print(f"  Original: {orig_h}x{orig_w} -> Target: {target_h}x{target_w}")

    # 3. Select views
    print(f"\n[3/6] Selecting views...")
    num_context = args.num_context if args.num_context is not None else num_total
    num_context = min(num_context, num_total)

    context_indices = select_context_views(
        scene_data["c2w"], num_context, args.context_selection,
    )
    target_indices = select_target_views(
        num_total, context_indices, args.target_selection, args.num_target
    )
    print(f"  Context: {len(context_indices)} views (strategy: {args.context_selection})")
    print(f"  Target: {len(target_indices)} views (strategy: {args.target_selection})")

    if len(target_indices) == 0:
        print("  Warning: No target views selected. Using all views as targets.")
        target_indices = np.arange(num_total)

    # 4. Load and preprocess images
    print(f"\n[4/6] Loading and preprocessing images...")
    context_images = load_and_preprocess_images(
        [scene_data["image_paths"][i] for i in context_indices],
        target_h,
        target_w,
    )
    target_images = load_and_preprocess_images(
        [scene_data["image_paths"][i] for i in target_indices],
        target_h,
        target_w,
    )
    print(f"  Context images: {context_images.shape}")
    print(f"  Target images: {target_images.shape}")

    # 5. Build model (only on first scene)
    if encoder is None:
        print(f"\n[5/6] Building model (experiment={args.experiment})...")
        encoder, decoder, data_shim = build_model(
            experiment=args.experiment,
            checkpoint=args.checkpoint,
            num_refine=args.num_refine,
            image_shape=(target_h, target_w),
            overrides=args.overrides,
            device=args.device,
            no_strict_load=True,
        )
        print(f"  Model loaded on {args.device}")
    else:
        print(f"\n[5/6] Reusing model from previous scene")

    # 6. Build batch and run inference
    print(f"\n[6/6] Running inference...")
    context_c2w = torch.tensor(
        scene_data["c2w"][context_indices], dtype=torch.float32
    )
    target_c2w = torch.tensor(
        scene_data["c2w"][target_indices], dtype=torch.float32
    )
    context_K = torch.tensor(
        scene_data["intrinsics"][context_indices], dtype=torch.float32
    )
    target_K = torch.tensor(
        scene_data["intrinsics"][target_indices], dtype=torch.float32
    )

    batch = build_batch(
        context_images,
        target_images,
        context_c2w,
        target_c2w,
        context_K,
        target_K,
        args.near,
        args.far,
        scene_name,
        args.device,
    )
    batch = data_shim(batch)

    gaussians, rendered, rendered_depth, visualization_dump = run_inference(
        encoder, decoder, batch, args.num_refine, args.render_chunk_size,
        save_depth=args.save_depth,
    )

    # Save outputs
    print(f"\nSaving outputs to {output_dir}...")
    target_image_names = [scene_data["image_names"][i] for i in target_indices]
    context_image_names = [scene_data["image_names"][i] for i in context_indices]

    save_outputs(
        rendered,
        gaussians,
        batch,
        output_dir,
        target_image_names,
        save_images=save_images,
        save_ply=args.save_ply,
        save_depth=args.save_depth,
        context_image_names=context_image_names,
        context_images=context_images,
        visualization_dump=visualization_dump,
        rendered_depth=rendered_depth,
        max_save_images=args.max_save_images,
    )

    # Render smooth video from all scene poses
    if args.save_video:
        render_smooth_video(
            gaussians, decoder,
            scene_data["c2w"],  # all poses from start to end
            context_c2w,  # unnormalized context poses for normalization pivot
            scene_data["intrinsics"][0],  # single intrinsic
            args.near, args.far, (target_h, target_w), output_dir,
            render_chunk_size=args.render_chunk_size,
            fps=args.smooth_video_fps,
            smooth_kernel=args.smooth_video_kernel,
            device=args.device,
        )

    # Evaluate metrics on target views (rendered vs ground truth)
    if not args.no_eval:
        print(f"\nEvaluating metrics on target views...")
        metrics = compute_metrics(
            rendered, target_images, target_image_names, output_dir,
            device=args.device,
        )
        # Append config info to metrics file for reproducibility
        metrics["config"] = {
            "context_selection": args.context_selection,
            "num_context": len(context_indices),
            "num_target": len(target_indices),
            "num_refine": args.num_refine,
            "resolution": f"{target_h}x{target_w}",
            "model_preset": args.model_preset,
            "checkpoint": args.checkpoint,
        }
        if args.start_frame is not None:
            metrics["config"]["start_frame"] = args.start_frame
            metrics["config"]["frame_distance"] = args.frame_distance
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    print(f"\nDone! Results saved to {output_dir}")
    return encoder, decoder, data_shim


def write_aggregate_metrics(all_metrics, output_dir):
    """Write aggregate metrics across all scenes."""
    psnrs = [m["mean"]["psnr"] for m in all_metrics.values() if "mean" in m]
    ssims = [m["mean"]["ssim"] for m in all_metrics.values() if "mean" in m]
    lpipss = [m["mean"]["lpips"] for m in all_metrics.values() if "mean" in m]

    if not psnrs:
        return

    summary = {
        "num_scenes": len(psnrs),
        "mean_psnr": round(sum(psnrs) / len(psnrs), 3),
        "mean_ssim": round(sum(ssims) / len(ssims), 4),
        "mean_lpips": round(sum(lpipss) / len(lpipss), 4),
        "per_scene": {k: v["mean"] for k, v in all_metrics.items() if "mean" in v},
    }

    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "aggregate_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAggregate metrics ({len(psnrs)} scenes):")
    print(f"  PSNR: {summary['mean_psnr']}, SSIM: {summary['mean_ssim']}, "
          f"LPIPS: {summary['mean_lpips']}")


def main():
    args = parse_args()

    # Apply model preset defaults (explicit CLI args take precedence)
    if args.model_preset:
        preset = MODEL_PRESETS[args.model_preset]
        if args.checkpoint is None:
            args.checkpoint = preset["checkpoint"]
        if args.num_context is None:
            args.num_context = preset["num_context"]
        if args.num_refine is None:
            args.num_refine = preset["num_refine"]
        if args.max_resolution is None:
            args.max_resolution = preset["max_resolution"]
        # Prepend preset overrides; user's --overrides appended after so they win
        args.overrides = preset["overrides"] + args.overrides

    # Apply fallback defaults for anything still None
    if args.num_refine is None:
        args.num_refine = 4
    if args.max_resolution is None:
        args.max_resolution = 960

    # Validate checkpoint is set
    if args.checkpoint is None:
        raise ValueError(
            "No checkpoint specified. Use --checkpoint or --model_preset."
        )

    # Validate scene specification
    if args.scene_path is None and args.data_dir is None:
        raise ValueError(
            "Must specify either --scene_path or --data_dir (with --scene_name or --scene_list)."
        )
    if args.data_dir is not None and args.scene_name is None and args.scene_list is None:
        raise ValueError(
            "--data_dir requires --scene_name or --scene_list."
        )

    # Validate frame range
    if args.start_frame is not None and args.frame_distance is None:
        raise ValueError("--start_frame requires --frame_distance.")
    if args.frame_distance is not None and args.start_frame is None:
        args.start_frame = 0

    # Resolve sparse_dir/images_dir defaults based on mode
    if args.sparse_dir is None:
        args.sparse_dir = "sparse/0"
    if args.images_dir is None:
        args.images_dir = "images_4"

    print("=" * 60)
    print("ReSplat COLMAP Inference")
    print("=" * 60)
    if args.model_preset:
        print(f"  Preset: {args.model_preset}")
        preset = MODEL_PRESETS[args.model_preset]
        if preset["overrides"]:
            print(f"  Preset overrides: {preset['overrides']}")
    if args.start_frame is not None:
        print(f"  Frame range: [{args.start_frame}, {args.start_frame + args.frame_distance})")

    # Resolve scene list
    if args.scene_list is not None:
        if args.scene_list == "all":
            scene_names = sorted(
                d for d in os.listdir(args.data_dir)
                if os.path.isdir(os.path.join(args.data_dir, d))
            )
        else:
            with open(args.scene_list) as f:
                scene_names = [line.strip() for line in f if line.strip()]
    elif args.scene_name is not None:
        scene_names = [args.scene_name]
    else:
        scene_names = None  # single --scene_path mode

    encoder = decoder = data_shim = None

    if scene_names is not None:
        all_metrics = {}
        print(f"\nProcessing {len(scene_names)} scene(s) from {args.data_dir}")

        for i, sname in enumerate(scene_names):
            scene_path = os.path.join(args.data_dir, sname)
            scene_output_dir = os.path.join(args.output_dir, sname)

            print(f"\n{'=' * 60}")
            print(f"Scene {i + 1}/{len(scene_names)}: {sname}")
            print(f"{'=' * 60}")

            try:
                encoder, decoder, data_shim = run_scene(
                    args, scene_path, sname, scene_output_dir,
                    encoder, decoder, data_shim,
                )
            except Exception as e:
                print(f"\nError processing scene {sname}: {e}")
                continue
            finally:
                torch.cuda.empty_cache()

            # Collect per-scene metrics
            metrics_path = os.path.join(scene_output_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    all_metrics[sname] = json.load(f)

        # Write aggregate metrics
        if all_metrics and not args.no_eval:
            write_aggregate_metrics(all_metrics, args.output_dir)
    else:
        # Single scene mode (existing behavior)
        scene_name = os.path.basename(os.path.normpath(args.scene_path))
        run_scene(args, args.scene_path, scene_name, args.output_dir)


if __name__ == "__main__":
    main()
