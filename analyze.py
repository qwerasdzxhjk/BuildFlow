import os

# Set the CUDA device.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import collections
import csv

import cv2
import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from absl import app, flags
from ml_collections import config_flags
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import libs.autoencoder


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training config.", lock_config=False)
flags.mark_flags_as_required(["config"])

# TODO: Provide the analysis image directory via CLI.
flags.DEFINE_string("input_dir", "", "Images dir.")
# TODO: Provide the analysis mask directory via CLI.
flags.DEFINE_string("label_dir", "", "Masks dir.")
# TODO: Provide the analysis output directory via CLI.
flags.DEFINE_string("output_dir", "", "Output dir for saved data.")
flags.DEFINE_integer("sample_steps", 10, "Integration steps.")
flags.DEFINE_string("target_image", None, "Specific image stem (e.g., 'image_001').")
flags.DEFINE_integer("boundary_band_width", 5, "Buffer size.")


def find_paths_by_stem(folder_path, exts):
    stem_to_path = {}
    for ext in exts:
        for path in Path(folder_path).glob(f"**/*.{ext}"):
            stem_to_path[path.stem] = path
    return stem_to_path


def validate_analyze_config(config):
    required_paths = {
        "input_dir": "Please provide --input_dir before running analyze.py.",
        "label_dir": "Please provide --label_dir before running analyze.py.",
        "output_dir": "Please provide --output_dir before running analyze.py.",
    }

    for key, error_message in required_paths.items():
        value = getattr(config, key, "")
        if not value:
            raise ValueError(error_message)


class IntroductionExperimentAnalyzer:
    def __init__(self, device):
        self.device = device

    def _to_uint8_gray(self, img_tensor):
        img_np = (img_tensor.detach().cpu().float() * 0.5 + 0.5).clamp(0.0, 1.0).numpy()
        img_uint8 = (einops.rearrange(img_np, "c h w -> h w c") * 255).astype(np.uint8)
        return cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    def create_region_masks(self, gt_mask_tensor, buffer_size=5):
        gt_np = (gt_mask_tensor.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
        kernel_size = int(buffer_size * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        edge = cv2.morphologyEx(gt_np, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        mask_boundary = cv2.dilate(edge, kernel)
        mask_interior = cv2.bitwise_and(gt_np, cv2.bitwise_not(mask_boundary))
        gt_inv = cv2.bitwise_not(gt_np)
        mask_exterior = cv2.bitwise_and(gt_inv, cv2.bitwise_not(mask_boundary))

        return mask_interior, mask_exterior, mask_boundary

    def calc_masked_binarized_entropy(self, img_gray, mask=None):
        if mask is not None and cv2.countNonZero(mask) == 0:
            return 0.0

        _, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        hist = cv2.calcHist([img_bin], [0], mask, [256], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-8)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))

    def calc_laplacian_clarity_and_tv(self, img_tensor):
        img_np = (img_tensor.detach().cpu().float() * 0.5 + 0.5).clamp(0.0, 1.0).numpy()
        img_gray = 0.299 * img_np[0] + 0.587 * img_np[1] + 0.114 * img_np[2]

        gray_uint8 = (img_gray * 255).astype(np.uint8)
        laplacian_var = cv2.Laplacian(gray_uint8, cv2.CV_64F).var()

        diff_y = np.abs(img_gray[1:, :] - img_gray[:-1, :])
        diff_x = np.abs(img_gray[:, 1:] - img_gray[:, :-1])
        tv_value = np.mean(diff_y) + np.mean(diff_x)

        return float(laplacian_var), float(tv_value)

    def process_and_compare(self, p_dec, l_dec, times, gt_mask_tensor, buf_size=5):
        mask_interior, mask_exterior, mask_boundary = self.create_region_masks(
            gt_mask_tensor,
            buffer_size=buf_size,
        )

        metrics = {
            "time": times,
            "pixel": collections.defaultdict(list),
            "latent": collections.defaultdict(list),
        }

        for traj_name, states in [("pixel", p_dec), ("latent", l_dec)]:
            for t, img_t in zip(times, states):
                img_gray = self._to_uint8_gray(img_t)

                metrics[traj_name]["ent_in"].append(
                    self.calc_masked_binarized_entropy(img_gray, mask_interior)
                )
                metrics[traj_name]["ent_ex"].append(
                    self.calc_masked_binarized_entropy(img_gray, mask_exterior)
                )
                metrics[traj_name]["ent_bd"].append(
                    self.calc_masked_binarized_entropy(img_gray, mask_boundary)
                )

                clarity, tv = self.calc_laplacian_clarity_and_tv(img_t)
                metrics[traj_name]["clarity"].append(clarity)
                metrics[traj_name]["tv"].append(tv)

        return metrics


def save_metrics_to_csv_and_plots(metrics, out_dir, fname, buf_size):
    times = metrics["time"]
    p_m = metrics["pixel"]
    l_m = metrics["latent"]

    headers = [
        "time",
        "pixel_clarity", "pixel_tv", "pixel_ent_in", "pixel_ent_ex", "pixel_ent_bd",
        "latent_clarity", "latent_tv", "latent_ent_in", "latent_ent_ex", "latent_ent_bd",
    ]

    csv_path = os.path.join(out_dir, f"{fname}_metrics.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(len(times)):
            row = [
                times[i],
                p_m["clarity"][i], p_m["tv"][i], p_m["ent_in"][i], p_m["ent_ex"][i], p_m["ent_bd"][i],
                l_m["clarity"][i], l_m["tv"][i], l_m["ent_in"][i], l_m["ent_ex"][i], l_m["ent_bd"][i],
            ]
            writer.writerow(row)

    plt.style.use("seaborn-v0_8-whitegrid")

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(times, l_m["clarity"], "s--", color="#1f77b4", linewidth=2, label="Latent Linear (VAE)")
    ax1.plot(times, p_m["clarity"], "^-.", color="#2ca02c", linewidth=2, label="Pixel Linear")
    ax1.set_title("Laplacian Sharpness (Structural Clarity)")
    ax1.set_xlabel("Interpolation Time $t$")
    ax1.set_ylabel("Variance of Laplacian")
    ax1.legend()

    ax2.plot(times, l_m["tv"], "s--", color="#1f77b4", linewidth=2, label="Latent Linear (VAE)")
    ax2.plot(times, p_m["tv"], "^-.", color="#2ca02c", linewidth=2, label="Pixel Linear")
    ax2.set_title("Image Total Variation (Texture Complexity)")
    ax2.set_xlabel("Interpolation Time $t$")
    ax2.set_ylabel("Total Variation")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{fname}_quality_metrics.png"), dpi=300)
    plt.close(fig1)

    fig2, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(16, 5))

    regions = [
        (ax3, "ent_in", "Interior Bulk Entropy"),
        (ax4, "ent_ex", "Exterior Background Entropy"),
        (ax5, "ent_bd", f"Boundary Interface Entropy ({buf_size}px)"),
    ]

    for ax, key, title in regions:
        ax.plot(times, l_m[key], "s--", color="#1f77b4", linewidth=2, label="Latent Linear (VAE)")
        ax.plot(times, p_m[key], "^-.", color="#2ca02c", linewidth=2, label="Pixel Linear")
        ax.set_title(title)
        ax.set_xlabel("Interpolation Time $t$")
        ax.set_ylabel("Shannon Entropy")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{fname}_entropy_metrics.png"), dpi=300)
    plt.close(fig2)


def save_intermediate_images(p_dec, l_dec, times, out_dir):
    """Save each interpolation frame as an individual image file."""
    img_out_dir = os.path.join(out_dir, "interpolations")
    os.makedirs(img_out_dir, exist_ok=True)

    for i, t in enumerate(times):
        for prefix, states in [("pixel", p_dec), ("latent", l_dec)]:
            img_tensor = states[i]
            img_np = (img_tensor.float() * 0.5 + 0.5).clamp(0.0, 1.0).numpy()
            img_uint8 = (einops.rearrange(img_np, "c h w -> h w c") * 255).astype(np.uint8)

            save_path = os.path.join(img_out_dir, f"{prefix}_t_{t:.2f}.png")
            Image.fromarray(img_uint8).save(save_path)


class SimpleImageDataset(Dataset):
    def __init__(self, image_dir, label_dir, target_image=None):
        super().__init__()
        image_map = find_paths_by_stem(image_dir, ["tif", "jpg", "png"])
        label_map = find_paths_by_stem(label_dir, ["tif", "jpg", "png"])
        stems = [target_image] if target_image in image_map else sorted(image_map.keys())
        self.samples = [(image_map[s], label_map[s]) for s in stems if s in label_map]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ipath, lpath = self.samples[index]
        try:
            img = (np.array(Image.open(ipath).convert("RGB")) / 127.5 - 1.0).astype(np.float32)
            img_tensor = torch.from_numpy(einops.rearrange(img, "h w c -> c h w"))
            mask = np.array(Image.open(lpath))
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)[None, ...])
            return {"filename": ipath.stem, "image": img_tensor, "mask": mask_tensor}
        except Exception:
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None


def run_intro_experiment(config):
    validate_analyze_config(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.output_dir, exist_ok=True)

    dataset = SimpleImageDataset(config.input_dir, config.label_dir, config.target_image)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    autoencoder = libs.autoencoder.get_model(**config.autoencoder).to(device).eval()
    analyzer = IntroductionExperimentAnalyzer(device)

    steps = config.sample_steps
    buf_size = config.boundary_band_width
    times = [i / max(steps, 1) for i in range(steps + 1)]

    for batch in tqdm(dataloader, desc="Extracting metrics, plots, and images"):
        img = batch["image"].to(device)
        mask = batch["mask"].to(device)
        fname = batch["filename"][0]

        mask_as_img = mask.repeat(1, 3, 1, 1) * 2.0 - 1.0

        p_dec = [(1.0 - t) * img + t * mask_as_img for t in times]
        p_dec = [state[0].detach().cpu() for state in p_dec]

        with torch.no_grad(), torch.cuda.amp.autocast():
            z_src = autoencoder.encode(img)
            z_tgt = autoencoder.encode(mask_as_img)
            l_dec = []
            for t in times:
                z_t = (1.0 - t) * z_src + t * z_tgt
                dec_t = autoencoder.decode(z_t)
                l_dec.append(dec_t[0].detach().cpu())

        out_dir = os.path.join(config.output_dir, fname)
        os.makedirs(out_dir, exist_ok=True)

        metrics = analyzer.process_and_compare(p_dec, l_dec, times, mask[0], buf_size)
        save_metrics_to_csv_and_plots(metrics, out_dir, fname, buf_size)
        save_intermediate_images(p_dec, l_dec, times, out_dir)

        print(f"[{fname}] Completed. CSV, plots, and interpolation frames were saved to {out_dir}")


def main(argv):
    config = FLAGS.config
    with config.unlocked():
        for f in ["input_dir", "label_dir", "output_dir", "sample_steps", "target_image", "boundary_band_width"]:
            if hasattr(FLAGS, f):
                config[f] = getattr(FLAGS, f)
    run_intro_experiment(config)


if __name__ == "__main__":
    app.run(main)
