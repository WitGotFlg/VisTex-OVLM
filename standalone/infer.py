#!/usr/bin/env python3
"""
Standalone VisTex-OVLM inference.
No project source code dependencies. Requires: torch, torchvision, transformers, numpy, PIL.

Usage:
    python infer.py --ref assets/images/ref1.png --img assets/images/img1.png --text spoon
"""
import os
import sys
import argparse
import copy

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules import (
    Backbone, LanguageBackbone, VLDyHead, ConvMapModule,
    AnchorGenerator, ATSSPostProcessor, to_image_list,
)

# ═══════════════════════════════════════════════
# Model definition (attribute names match checkpoint keys)
# ═══════════════════════════════════════════════

class RPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = VLDyHead(num_convs=8, skip_stages={0})
        self.anchor_generator = AnchorGenerator()


class VisTex(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.language_backbone = LanguageBackbone()
        self.rpn = RPN()
        self.fcto768 = ConvMapModule()


# ═══════════════════════════════════════════════
# Image preprocessing
# ═══════════════════════════════════════════════

PIXEL_MEAN = torch.tensor([103.530, 116.280, 123.675]).view(3, 1, 1)
PIXEL_STD = torch.tensor([57.375, 57.120, 58.395]).view(3, 1, 1)


def _rgb_to_bgr_normalize(tensor):
    """Convert RGB [0,1] tensor to BGR [0,255] normalized."""
    return (tensor[[2, 1, 0]] * 255.0 - PIXEL_MEAN) / PIXEL_STD


def preprocess_image(path, min_size=800, max_size=800):
    """Load and preprocess query image. Returns (tensor[3,H,W], orig_size, (H,W))."""
    img = Image.open(path).convert("RGB")
    ow, oh = img.size
    scale = min_size / min(ow, oh)
    if max(ow, oh) * scale > max_size:
        scale = max_size / max(ow, oh)
    nw, nh = int(round(ow * scale)), int(round(oh * scale))
    img = img.resize((nw, nh), Image.Resampling.BILINEAR)
    return _rgb_to_bgr_normalize(TF.to_tensor(img)), (ow, oh), (nh, nw)


def preprocess_ref(path, size=800):
    """Load reference patch, resize to size×size. Returns [1,3,size,size]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return _rgb_to_bgr_normalize(TF.to_tensor(img)).unsqueeze(0)


# ═══════════════════════════════════════════════
# Caption / positive map
# ═══════════════════════════════════════════════

def build_positive_map(tokenized):
    """Parse tokenized caption to build {label_id: [token_positions]} dict.
    Splits phrases on '.' (token 1012) and [SEP] (token 102)."""
    ids = tokenized["input_ids"][0]
    pmap, phrase, positions = {}, 0, []
    for idx in range(1, len(ids)):
        t = ids[idx].item()
        if t == 0:
            break
        if t in (1012, 102):  # '.' or [SEP]
            if positions:
                phrase += 1
                pmap[phrase] = positions
                positions = []
        else:
            positions.append(idx)
        if phrase >= 79:
            break
    return pmap


def mstb_inject(visual_768, positive_map, label, lang_tensor):
    """Place 768-d visual embedding at position after label's last token."""
    next_pos = positive_map[label][-1] + 1
    lang_tensor[:, next_pos, :] = visual_768


# ═══════════════════════════════════════════════
# Weight loading
# ═══════════════════════════════════════════════

def load_model(weight_path, device):
    model = VisTex()
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    cleaned = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[7:]
        cleaned[k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    real_missing = [k for k in missing if "relative_position_index" not in k]
    if real_missing:
        print(f"Note: {len(real_missing)} missing keys (may include buffers)")
    return model.to(device).eval()


# ═══════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════

def visualize_and_save(img_path, predictions, pred_size, output_path, threshold=0.3):
    img = Image.open(img_path).convert("RGB")
    ow, oh = img.size
    pw, ph = pred_size
    sx, sy = ow / pw, oh / ph
    draw = ImageDraw.Draw(img)

    boxes = predictions.bbox.cpu().numpy()
    scores = predictions.get_field("scores").cpu().numpy()
    count = 0
    for i in range(len(boxes)):
        if scores[i] < threshold:
            continue
        x1, y1, x2, y2 = boxes[i]
        x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
        draw.text((x1, max(y1 - 12, 0)), f"{scores[i]:.2f}", fill="lime")
        count += 1

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    img.save(output_path)
    print(f"Saved {count} detections (threshold={threshold}) to {output_path}")


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="VisTex-OVLM standalone inference")
    parser.add_argument("--ref", required=True, help="Reference patch image")
    parser.add_argument("--img", required=True, help="Query image")
    parser.add_argument("--text", required=True, help="Object text description")
    parser.add_argument("--weight", default="vistex/model_best.pth")
    parser.add_argument("--output", default="output/standalone_results")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model = load_model(args.weight, device)

    # Preprocess images
    img_tensor, orig_size, (nh, nw) = preprocess_image(args.img)
    ref_tensor = preprocess_ref(args.ref).to(device)
    batch, image_sizes = to_image_list([img_tensor], 32)
    batch = batch.to(device)

    # Tokenize caption
    caption = args.text.strip() + ". "
    tokenized = tokenizer(caption, max_length=256, padding="max_length",
                          return_tensors="pt", truncation=True)
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    positive_map = build_positive_map(tokenized)

    with torch.no_grad():
        # 1. Visual features for query
        visual_features = model.backbone(batch)

        # 2. Language features
        lang_dict = model.language_backbone(tokenized)

        # 3. Ref → backbone → fcto768 → 768-d visual embedding
        ref_features = model.backbone(ref_tensor)
        visual_768 = model.fcto768(ref_features)

        # 4. MSTB injection into embedded and hidden
        label = 1
        pm_copy = copy.deepcopy(positive_map)
        mstb_inject(visual_768, pm_copy, label, lang_dict["embedded"])
        mstb_inject(visual_768, positive_map, label, lang_dict["hidden"])

        # 5. VLDyHead tower + prediction heads
        logits, bbox_reg, centerness, dot_logits = model.rpn.head(
            visual_features, lang_dict, lang_dict["hidden"])

        # 6. Anchors + postprocess
        anchors = model.rpn.anchor_generator(image_sizes, visual_features)
        postprocessor = ATSSPostProcessor()
        results = postprocessor(bbox_reg, centerness, anchors, logits,
                                dot_logits, positive_map)

    predictions = results[0]
    n_total = len(predictions)
    n_above = (predictions.get_field("scores") >= args.threshold).sum().item()
    print(f"Detections: {n_total} total, {n_above} above threshold {args.threshold}")

    # Save visualization
    os.makedirs(args.output, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.img))[0]
    out_path = os.path.join(args.output, f"{name}_result.jpg")
    visualize_and_save(args.img, predictions, (nw, nh), out_path, args.threshold)


if __name__ == "__main__":
    main()