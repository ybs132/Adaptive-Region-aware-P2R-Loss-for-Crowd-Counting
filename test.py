# test.py
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torchvision import transforms
from models import build_model
from yacs.config import CfgNode as CN


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="/datasets/ShanghaiTech",
        help="ShanghaiTech root directory (contains part_A, part_B)",
    )
    parser.add_argument(
        "--part",
        type=str,
        default="A",
        choices=["A", "B", "both"],
        help="Which part to evaluate: A, B, or both",
    )

    return parser.parse_args()


def get_split_paths(root, part):
    if part == "A":
        part_dir = os.path.join(root, "part_A")
    else:
        part_dir = os.path.join(root, "part_B")

    img_dir = os.path.join(part_dir, "test_data", "images")
    gt_dir  = os.path.join(part_dir, "test_data", "new-anno")

    return part_dir, img_dir, gt_dir


MODEL = None
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model_once():
    global MODEL
    if MODEL is not None:
        return MODEL

    cfg = CN()
    cfg.NAME = "VGG16BN"
    cfg.FACTOR = 1
    cfg.LOSS = "P2R"

    student, teacher = build_model(cfg)

    # ----- choice test model ---- #
    ckpt = torch.load("./exp/sha-L5/output/ckpt_epoch_best.pth", map_location="cpu")
    # ---------------------------- #

    student.load_state_dict(ckpt["student"])

    student.cuda()
    student.eval()
    MODEL = student
    return MODEL


def predict_count(img_path):
    model = load_model_once()

    img = Image.open(img_path).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0).cuda()   # (1, 3, H, W)

    with torch.no_grad():
        den = model(x)                      # (1, 1, h, w) density like
        pred_count = (den > 0).sum().item()

    return pred_count


def eval_split(img_dir, gt_dir):
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if len(img_paths) == 0:
        print(f"[WARN] No images found in {img_dir}")
        return None

    abs_errors = []
    sq_errors = []

    for img_path in tqdm(img_paths, desc=f"Eval {img_dir}"):
        base = os.path.splitext(os.path.basename(img_path))[0]
        gt_name = f"GT_{base}.npy"
        gt_path = os.path.join(gt_dir, gt_name)

        if not os.path.exists(gt_path):
            print(f"[WARN] GT not found for {base}: {gt_path}")
            continue

        gt_points = np.load(gt_path)          
        gt_count = gt_points.shape[0]  

        pred_count = predict_count(img_path)

        err = abs(pred_count - gt_count)
        abs_errors.append(err)
        sq_errors.append(err ** 2)

    if len(abs_errors) == 0:
        print(f"[ERROR] No valid samples found in {img_dir}")
        return None

    abs_errors = np.array(abs_errors)
    sq_errors = np.array(sq_errors)

    mae = abs_errors.mean()
    mse = np.sqrt(sq_errors.mean())  # RMSE

    return mae, mse


def main():
    args = parse_args()

    parts = []
    if args.part == "both":
        parts = ["A", "B"]
    else:
        parts = [args.part]

    for p in parts:
        print(f"\n========== Evaluating ShanghaiTech Part {p} ==========")
        part_dir, img_dir, gt_dir = get_split_paths(args.root, p)

        print(f"Part {p} image dir: {img_dir}")
        print(f"Part {p} gt dir   : {gt_dir}")

        result = eval_split(img_dir, gt_dir)
        if result is None:
            continue

        mae, mse = result
        print(f"[Part {p}] MAE: {mae:.3f}, MSE (RMSE): {mse:.3f}")


if __name__ == "__main__":
    main()