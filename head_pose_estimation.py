# https://towardsdatascience.com/head-pose-estimation-using-python-d165d3541600
import os
import random
import typing as tp

import cv2
import mediapipe as mp
import numpy as np
import tqdm
from PIL import Image

# from face_geometry import PCF, get_metric_landmarks
from vector import Vector

FFHQ_DIR = "/home/andresokol/code/mastersdata/ffhq-dataset/images1024x1024"
RENDERED_DIR = "/home/andresokol/code/mastersdata/rendered"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True,
)


def get_landmarks(image: np.ndarray) -> tp.Optional[np.ndarray]:
    image.flags.writeable = False
    landmarks = face_mesh.process(image).multi_face_landmarks  # type: ignore
    if not landmarks:
        raise Exception("no face found")
    landmark = landmarks[0].landmark
    image.flags.writeable = True
    return np.array([[x.x, x.y, x.z] for x in landmark])


def draw_pt(image: np.ndarray, pt, color=(255, 0, 0)) -> np.ndarray:
    img_h, img_w, _ = image.shape

    return cv2.circle(
        image,
        (int(img_w * pt[0]), int(img_h * pt[1])),
        radius=1,
        color=color,
        thickness=1,
    )


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def apply_overlay(image: np.ndarray, overlay_img: Image, box: list) -> np.ndarray:
    # cv2.line(image, box[0], box[1], (0, 0, 255), 1)
    # cv2.line(image, box[1], box[2], (0, 0, 255), 1)
    # cv2.line(image, box[2], box[3], (0, 0, 255), 1)
    # cv2.line(image, box[3], box[0], (0, 0, 255), 1)
    #
    img_h, img_w, _ = image.shape
    overlay_h, overlay_w = overlay_img.size

    overlay = overlay_img.copy()
    # overlay.putalpha(1)
    overlay: Image = overlay.transform(
        size=(img_w, img_h),
        method=Image.Transform.PERSPECTIVE,  # type: ignore
        data=find_coeffs(
            box,
            [(0, 0), (0, overlay_h), (overlay_w, overlay_h), (overlay_w, 0)],
        ),
    )

    img_t = Image.fromarray(image).convert("RGBA")
    img_t = Image.alpha_composite(img_t, overlay)
    img_t.resize((512, 512))

    return np.asarray(img_t)


def get_face_box(image: np.ndarray) -> tp.List[tp.Tuple[int, int]]:
    landmarks = get_landmarks(image)
    # if landmarks is None:
    #     return image
    assert landmarks is not None, "no face in picture"

    stable = [
        33,
        263,
        # 61, 291,
        199,
    ]
    # for i in stable:
    #     image = draw_pt(image, landmarks[i], color=(0, 255, 255))

    left_eye = landmarks[33]
    right_eye = landmarks[263]
    chin_pt = landmarks[199]

    img_h, img_w, _ = image.shape

    left = Vector(img_w * left_eye[0], img_h * left_eye[1], img_w * left_eye[2])
    right = Vector(img_w * right_eye[0], img_h * right_eye[1], img_w * right_eye[2])

    medium = 0.5 * (right + left)
    chin = Vector(img_w * chin_pt[0], img_h * chin_pt[1], img_w * chin_pt[2])

    normal_1 = medium - left
    normal_1.normalize()
    chin_down = medium - chin
    chin_down.normalize()
    # chin_down_outwards = normal_1.cross_product(chin_down)
    # chin_down_outwards.normalize()
    # chin_down_outwards = 50 * chin_down_outwards

    # cv2.line(image, (int(medium.x), int(medium.y)),
    #          (int((medium + chin_down_outwards).x), int((medium + chin_down_outwards).y)),
    #          (200, 200, 0), 2)

    eye_distance = (left - right).length()
    box_halfsize = 3 * eye_distance

    to_left = box_halfsize * normal_1
    to_top = box_halfsize * -1 * chin_down

    box_3d = [
        medium + to_left - to_top,
        medium + to_left + to_top,
        medium - to_left + to_top,
        medium - to_left - to_top,
    ]

    box_screen = [(int(v.x), int(v.y)) for v in box_3d]

    return box_screen


def online():
    overlay_img = Image.open("result/strands00001_base.png")

    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        _, image = capture.read()
        image: np.ndarray = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image = process_image(image, overlay_img)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Head Pose Estimation", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    capture.release()


def image_paths() -> tp.List[tp.Tuple[str, str]]:
    paths = []
    for dirname in os.listdir(FFHQ_DIR):
        for filename in os.listdir(f"{FFHQ_DIR}/{dirname}"):
            if filename[:5].isdigit() and filename[5:] == ".png":
                paths.append((dirname, filename[:5]))
    return paths


def offline():
    strands = [f"strands{i:0>5}" for i in range(1, 59)]
    for img_dir, img_name in tqdm.tqdm(image_paths()):
        strand = random.choice(strands)

        image: np.ndarray = cv2.imread(f"{FFHQ_DIR}/{img_dir}/{img_name}.png")
        #     cv2.COLOR_RGB2BGR,
        # )
        overlay_base = Image.open(f"result/{strand}_base.png")
        overlay_structure = Image.open(f"result/{strand}_structure.png")

        target_dir = f"{RENDERED_DIR}/{img_dir}"
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        if os.path.exists(f"{target_dir}/{img_name}_base.png") and os.path.exists(
                f"{target_dir}/{img_name}_structure.png"):
            continue

        try:
            box = get_face_box(image)
            cv2.imwrite(
                f"{target_dir}/{img_name}_base.png",
                apply_overlay(image, overlay_base, box),
            )
            cv2.imwrite(
                f"{target_dir}/{img_name}_structure.png",
                apply_overlay(image, overlay_structure, box),
            )
        except Exception as exc:
            print("ERR!", exc, img_dir, img_name + ".png")


if __name__ == "__main__":
    offline()
