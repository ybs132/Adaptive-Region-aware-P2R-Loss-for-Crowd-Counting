import os
import glob
import numpy as np
import scipy.io as sio

def convert_split(split_dir):
    img_dir = os.path.join(split_dir, "images")
    gt_dir  = os.path.join(split_dir, "ground-truth")
    out_dir = os.path.join(split_dir, "new-anno")

    os.makedirs(out_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    print(f"[{split_dir}] Found {len(img_paths)} images.")

    for img_path in img_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]  # IMG_1

        mat_path = os.path.join(gt_dir, f"GT_{base}.mat")
        if not os.path.exists(mat_path):
            print(f"[WARN] GT file not found for {base}: {mat_path}")
            continue

        mat = sio.loadmat(mat_path)

        # mat["image_info"][0][0][0][0][0] → (N, 2)
        try:
            points = mat["image_info"][0][0][0][0][0]
        except:
            points = mat["image_info"][0][0][0][0]

        points = np.asarray(points, dtype=np.float32)

        np.save(os.path.join(out_dir, f"GT_{base}.npy"), points)

    print(f"[DONE] Converted → {out_dir}\n")


def convert_shanghai_base(base_dir):
    for part in ["part_A", "part_B"]:
        p = os.path.join(base_dir, part)
        if not os.path.exists(p):
            print(f"[SKIP] Not found: {p}")
            continue

        print(f"===== Converting {part} =====")
        convert_split(os.path.join(p, "train_data"))
        convert_split(os.path.join(p, "test_data"))


if __name__ == "__main__":
    BASE = "/datasets/ShanghaiTech"   

    convert_shanghai_base(BASE)