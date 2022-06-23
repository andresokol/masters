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

SOURCE_DIR = "/home/andresokol/data/compressed"
MASK_ROOT = "/home/andresokol/data/masks_v2"
ORIENTATION_ROOT = "/home/andresokol/data/orientation_v2"


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


def read_original(img_dir: str, img_name: str) -> np.ndarray:
    image = cv2.imread(f"{SOURCE_DIR}/{img_dir}/{img_name}.jpg")
    image = cv2.resize(image, (512, 512))
    return image


def read_mask(img_dir: str, img_name: str) -> np.ndarray:
    image = cv2.imread(f"{MASK_ROOT}/{img_dir}/{img_name}.png")
    image = cv2.resize(image, (512, 512))
    return image


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def extract_structure_data(original: np.ndarray) -> np.ndarray:
    image = np.copy(original)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     direction_weights = np.dstack(
#         (
#             get_direction_weights(image, KERNELS[0]),
#             np.maximum(
#                 get_direction_weights(image, KERNELS[1]),
#                 get_direction_weights(image, KERNELS[3]),
#             ),
#             get_direction_weights(image, KERNELS[2]),
#         )
#     )
#     directions = np.argmax(direction_weights, axis=2)

#     blue_channel = directions * 127
#     red_channel = 255 - blue_channel

    axis_0 = scipy.ndimage.convolve1d(image.astype(float), [-1, 1, 0], axis=0)
    axis_0 = np.abs(axis_0)
    
    axis_1 = scipy.ndimage.convolve1d(image.astype(float), [-1, 1, 0], axis=1)
    axis_1 = np.abs(axis_1)

    composite = np.dstack([
        axis_1,
        np.zeros_like(axis_0),
        axis_0,
    ])
    
    composite = adjust_gamma(composite.astype('uint8'), gamma=4)

    
#     image = cv2.merge(
#         (
#             red_channel.astype("uint8"),
#             np.zeros(shape=red_channel.shape, dtype="uint8"),
#             blue_channel.astype("uint8"),
#         )
#     )

    # image = cv2.boxFilter(image, -1, (16, 16))

    # downsampled = cv2.resize(image, (64, 64))

    return composite


def apply_threshold(image: np.ndarray, threshold: float) -> np.ndarray:
    normalized = image.astype(float) / 255.0
    return (normalized > threshold).astype(int) * 255


def process(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = extract_structure_data(image)

    hair_mask, _, _ = cv2.split(mask)
    # hair_mask = cv2.resize(hair_mask, image.shape[:2])
    hair_mask = apply_threshold(hair_mask, 0.4)

    image = np.dstack((image, hair_mask.astype("uint8")))

    return image


def apply_on_original(bg: np.ndarray, image: np.ndarray) -> np.ndarray:
    bg_pil = Image.fromarray(bg).convert("RGBA")
    image_pil = Image.fromarray(image)
    composite = Image.alpha_composite(bg_pil, image_pil)
    return np.asarray(composite)  # type: ignore


def image_paths() -> tp.List[tp.Tuple[str, str]]:
    paths = []
    for dirname in os.listdir(SOURCE_DIR):
        for filename in os.listdir(f"{SOURCE_DIR}/{dirname}"):
            if not os.path.exists(f"{MASK_ROOT}/{dirname}/{filename[:-len('.jpg')]}.png"):
                continue
            if filename.endswith(".jpg"):
                paths.append((dirname, filename[:-len(".jpg")]))
    print(f"Found {len(paths)} images")
    return paths


def main():
    # img_dir, img_name = "01000", "01000"
    for img_dir, img_name in tqdm.tqdm(image_paths()):
        target_dir = f"{ORIENTATION_ROOT}/{img_dir}"
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        if os.path.exists(f"{target_dir}/{img_name}.png"):
            continue

        original = read_original(img_dir, img_name)
        mask = read_mask(img_dir, img_name)

        image = process(original, mask)

        structure_overlayed = apply_on_original(original, image)
        cv2.imwrite(
            f"{target_dir}/{img_name}.png",
            structure_overlayed,
        )

    # cv2.imshow("w", structure_overlayed)
    # while cv2.waitKey(5) & 0xFF != 27:
    #     time.sleep(0.5)


if __name__ == "__main__":
    main()
