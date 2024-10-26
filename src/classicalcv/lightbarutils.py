from typing import Tuple

import cv2
import numpy as np

from .coordsmath import get_shorter_edge_midpoints
from .cvutils import get_bb_of_largest_cc
from .panelclasses import LightBar, LightBarPair


def split_image_in_half(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    # split image evenly into a left in right half, returning (left, right)
    middle = img.shape[1] // 2
    return img[:, :middle], img[:, middle:], middle


def get_light_bar(mask: np.ndarray) -> LightBar:
    """
    Find the minAreaRect of the largest connected component in the mask,
    find the midpoints of the top and bottom edge of the minarearect,
    and set the points of the lightbar accordingly
    :param img:
    :return:
    """
    rect = get_bb_of_largest_cc(mask)
    upper, lower = get_shorter_edge_midpoints(rect)
    return LightBar(upper, lower, rect)


def get_light_bar_pair(mask: np.ndarray) -> LightBarPair:
    left, right, middle = split_image_in_half(mask)
    leftlb = get_light_bar(left)
    rightlb = get_light_bar(right)
    rightlb.upper = (rightlb.upper[0] + middle, rightlb.upper[1])
    rightlb.lower = (rightlb.lower[0] + middle, rightlb.lower[1])
    return LightBarPair(leftlb, rightlb)


def unwarp_img_from_lbs(img: np.ndarray, light_bar_pair: LightBarPair) -> np.ndarray:
    """Given and image and lightbar pair, use warpPerspective to move the lightbar points to the corners of a 100x100 image array"""
    input_pts = np.array(
        [light_bar_pair.left.upper, light_bar_pair.right.upper, light_bar_pair.right.lower, light_bar_pair.left.lower],
        dtype=np.float32)
    output_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(input_pts, output_pts)
    outputimg = np.zeros((100, 100) if len(img.shape) == 2 else (100, 100, 3), dtype=np.uint8)
    cv2.warpPerspective(img, matrix, (100, 100), outputimg)

    return outputimg[:, 20:-20] if len(img.shape) == 2 else outputimg[:, 20:-20, :]
