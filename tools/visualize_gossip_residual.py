import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


PASCAL_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]

CITY_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize class-wise GossipFP disagreement residual norms."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to ckpt.pth or ckpt_best.pth.")
    parser.add_argument("--dataset", choices=["pascal", "city"], default="pascal")
    parser.add_argument("--output", default="residual_heatmap.png", help="Output PNG path.")
    parser.add_argument("--csv", default=None, help="Optional CSV path. Defaults to output stem + .csv.")
    parser.add_argument("--topk", type=int, default=8, help="Number of high-residual classes to annotate.")
    return parser.parse_args()


def unwrap_gossip_state(checkpoint):
    if "gossip_state" in checkpoint:
        return checkpoint["gossip_state"]
    return checkpoint


def tensor_from_state(state, name):
    if name not in state:
        raise KeyError(f"'{name}' was not found in checkpoint gossip_state.")
    value = state[name]
    if not torch.is_tensor(value):
        value = torch.tensor(value)
    return value.detach().float().cpu()


def color_from_value(value):
    value = max(0.0, min(1.0, float(value)))
    # blue -> pale -> red
    if value < 0.5:
        t = value / 0.5
        r = int(64 + 120 * t)
        g = int(118 + 85 * t)
        b = int(196 + 35 * t)
    else:
        t = (value - 0.5) / 0.5
        r = int(184 + 55 * t)
        g = int(203 - 126 * t)
        b = int(231 - 124 * t)
    return r, g, b


def draw_heatmap(names, norms, risks, output_path, topk):
    width = 1160
    row_h = 34
    margin_l = 190
    margin_r = 50
    margin_t = 88
    margin_b = 54
    height = margin_t + margin_b + row_h * len(names)
    bar_w = width - margin_l - margin_r

    max_norm = max(float(norms.max()), 1e-6)
    norm_values = (norms / max_norm).clamp(0, 1).tolist()
    risk_values = risks.clamp(0, 1).tolist()

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    title = "GossipFP Class Disagreement Residual ||b_c||_2"
    draw.text((margin_l, 24), title, fill=(20, 28, 38), font=font)
    draw.text((margin_l, 52), "Longer/hotter bars indicate stronger semantic graph disagreement.", fill=(70, 78, 90), font=font)

    order = torch.argsort(norms, descending=True).tolist()
    top = set(order[: max(0, topk)])

    for row, idx in enumerate(order):
        y = margin_t + row * row_h
        name = names[idx]
        norm = float(norms[idx])
        norm_v = norm_values[idx]
        risk = float(risks[idx]) if idx < len(risks) else 0.0
        fill = color_from_value(norm_v)
        bar_len = int(bar_w * norm_v)

        label_color = (16, 24, 39) if idx in top else (75, 85, 99)
        draw.text((24, y + 9), f"{idx:02d} {name}", fill=label_color, font=font)
        draw.rectangle((margin_l, y + 6, margin_l + bar_w, y + row_h - 7), outline=(226, 232, 240), fill=(248, 250, 252))
        draw.rectangle((margin_l, y + 6, margin_l + bar_len, y + row_h - 7), fill=fill)
        draw.text((margin_l + bar_w + 12, y + 9), f"{norm:.3f} / risk {risk:.3f}", fill=(31, 41, 55), font=font)

    image.save(output_path)


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = unwrap_gossip_state(checkpoint)
    disagreement = tensor_from_state(state, "last_disagreement")
    risks = tensor_from_state(state, "last_risk") if "last_risk" in state else torch.zeros(disagreement.shape[0])
    norms = disagreement.norm(dim=1)

    names = PASCAL_CLASSES if args.dataset == "pascal" else CITY_CLASSES
    if len(names) != disagreement.shape[0]:
        names = [f"class_{idx}" for idx in range(disagreement.shape[0])]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "residual_norm", "risk"])
        for idx, name in enumerate(names):
            writer.writerow([idx, name, float(norms[idx]), float(risks[idx])])

    draw_heatmap(names, norms, risks, output, args.topk)
    print(f"Saved residual heatmap to {output}")
    print(f"Saved residual values to {csv_path}")


if __name__ == "__main__":
    main()
