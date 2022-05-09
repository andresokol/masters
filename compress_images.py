import os
import typing as tp

import cv2
import tqdm

FILE_ROOT = "/home/andresokol/code/mastersdata"
FFHQ_DIR = f"{FILE_ROOT}/ffhq-dataset/images1024x1024"
COMPRESSED_ROOT = f"{FILE_ROOT}/compressed"


def image_paths() -> tp.List[tp.Tuple[str, str]]:
    paths = []
    for dirname in os.listdir(FFHQ_DIR):
        for filename in os.listdir(f"{FFHQ_DIR}/{dirname}"):
            if filename[:5].isdigit() and filename[5:] == ".png":
                paths.append((dirname, filename[:5]))
    return paths


def main():
    # img_dir, img_name = "01000", "01000"
    for img_dir, img_name in tqdm.tqdm(image_paths()):
        target_dir = f"{COMPRESSED_ROOT}/{img_dir}"
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        image = cv2.imread(f"{FFHQ_DIR}/{img_dir}/{img_name}.png")
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(f"{target_dir}/{img_name}.jpg", image)


if __name__ == "__main__":
    main()
