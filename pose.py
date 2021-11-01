import cv2
import mediapipe as mp
import numpy as np
from plot import *
from tqdm import tqdm

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def poses_for_dataset(dataloader):
    assert dataloader.batch_size is None

    result_list = []
    annotated_images = []
    for image, label in tqdm(dataloader):
        results, annotated = estimate_poses(image, label)
        result_list.append(results)
        annotated_images.append(annotated)
    return result_list, annotated_images


def estimate_poses(image, label, plot=False):
    BG_COLOR = (192, 192, 192)  # gray
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
        # for i in range(images.shape[0]):
            image = image.numpy().transpose((1, 2, 0))
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                if plot:
                    plot_image(image, label.item(), "Landmark not available")
                return None, None
            # print(
            #    f'Nose coordinates: ('
            #    f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
            #    f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
            # )
            annotated_image = image.copy()
            # Draw segmentation on the image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            # condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            # bg_image = np.zeros(image.shape, dtype=np.uint8)
            # bg_image[:] = BG_COLOR
            # annotated_image = np.where(condition, annotated_image, bg_image)
            # Draw pose landmarks on the image.
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # cv2.imwrite('/tmp/annotated_image' + str(i) + '.png', annotated_image)
            # Plot pose world landmarks.
            if plot:
                plot_image(annotated_image)
            # mp_drawing.plot_landmarks(
            #    results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    return results, annotated_image
