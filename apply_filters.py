import os
import typing as tp

import cv2
import numpy as np
import scipy.ndimage
import time
import tqdm
from PIL import Image


KERNELS = [
    np.array(
        [
            [-1, 2, -1],
            [-1, 2, -1],
            [-1, 2, -1],
        ]
    ),
    np.array(
        [
            [2, -1, -1],
            [-1, 2, -1],
            [-1, -1, 2],
        ]
    ),
    np.array(
        [
            [-1, -1, -1],
            [2, 2, 2],
            [-1, -1, -1],
        ]
    ),
    np.array(
        [
            [-1, -1, 2],
            [-1, 2, -1],
            [2, -1, -1],
        ]
    ),
]

FILE_ROOT = "/home/andresokol/code/mastersdata"
FFHQ_DIR = f"{FILE_ROOT}/ffhq-dataset/images1024x1024"
PREPARED_ROOT = f"{FILE_ROOT}/prepared"


def get_direction_weights(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    channel = cv2.filter2D(image, -1, kernel).astype(float)
    # channel = cv2.boxFilter(channel, -1, (3, 3))
    channel = scipy.ndimage.maximum_filter(channel, size=5)
    return channel


# def directed_sobel(image: np.ndarray, axis: int) -> np.ndarray:
#     channel = scipy.ndimage.sobel(image, axis=axis)
#     channel = np.abs(channel - 128) * 2
#     # channel = scipy.ndimage.maximum_filter(channel, size=3)
#     return channel


# def image_gradient(img: np.ndarray) -> np.ndarray:
#     hor_grad = (img[1:, :] - img[:-1, :])[:, :-1]
#     ver_grad = (img[:, 1:] - img[:, :-1])[:-1:, :]
#     magnitude = np.sqrt(hor_grad ** 2 + ver_grad ** 2)
#     return magnitude


def extract_structure_data(filepath: str) -> np.ndarray:
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    direction_weights = np.dstack(
        (
            get_direction_weights(image, KERNELS[0]),
            np.maximum(
                get_direction_weights(image, KERNELS[1]),
                get_direction_weights(image, KERNELS[3]),
            ),
            get_direction_weights(image, KERNELS[2]),
        )
    )
    directions = np.argmax(direction_weights, axis=2)
    # downsampled = cv2.resize(directions.astype('uint8'), (64, 64))
    # directions = cv2.resize(downsampled, image.shape[:2])
    # directions = scipy.ndimage.grey_opening(directions, size=3)
    # directions = scipy.ndimage.grey_closing(directions, size=3)
    directions = scipy.ndimage.median_filter(directions, size=3)

    blue_channel = directions * 127
    red_channel = 255 - blue_channel

    image = cv2.merge(
        (
            red_channel.astype("uint8"),
            np.zeros(shape=red_channel.shape, dtype="uint8"),
            blue_channel.astype("uint8"),
        )
    )

    # image = cv2.boxFilter(image, -1, (16, 16))

    # downsampled = cv2.resize(image, (64, 64))
    # image = cv2.resize(downsampled, image.shape[:2])

    return image


def apply_threshold(image: np.ndarray, threshold: float) -> np.ndarray:
    normalized = image.astype(float) / 255.0
    return (normalized > threshold).astype(int) * 255


def process(img_dir, img_name) -> np.ndarray:
    image = extract_structure_data(f"{FFHQ_DIR}/{img_dir}/{img_name}.png")

    mask = cv2.imread(f"{PREPARED_ROOT}/{img_dir}/{img_name}_mask.png")

    hair_mask, _, _ = cv2.split(mask)
    hair_mask = cv2.resize(hair_mask, image.shape[:2])
    hair_mask = apply_threshold(hair_mask, 0.4)

    image = np.dstack((image, hair_mask.astype("uint8")))

    return image


def apply_on_original(img_dir: str, img_name: str, image: np.ndarray) -> np.ndarray:
    bg = cv2.cvtColor(
        cv2.imread(f"{FFHQ_DIR}/{img_dir}/{img_name}.png"),
        cv2.COLOR_RGB2RGBA,
    )

    bg_pil = Image.fromarray(bg)
    image_pil = Image.fromarray(image)
    composite = Image.alpha_composite(bg_pil, image_pil)
    return np.asarray(composite)  # type: ignore


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
        target_dir = f"{PREPARED_ROOT}/{img_dir}"
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        image = process(img_dir, img_name)
        cv2.imwrite(
            f"{target_dir}/{img_name}_structure_masked.png",
            image,
        )
        structure_overlayed = apply_on_original(img_dir, img_name, image)
        cv2.imwrite(
            f"{target_dir}/{img_name}_structure_overlayed.png",
            structure_overlayed,
        )

    # cv2.imshow("w", structure_overlayed)
    # while cv2.waitKey(5) & 0xFF != 27:
    #     time.sleep(0.5)


if __name__ == "__main__":
    main()
