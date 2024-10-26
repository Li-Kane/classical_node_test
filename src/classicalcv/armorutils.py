import cv2
import numpy as np

from .colorutils import classify_color, threshold_by_color
from .lightbarutils import unwarp_img_from_lbs, get_light_bar_pair
from .panelclasses import ArmorPanel


def get_robot_type_mask(img: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    color = classify_color(ycrcb)
    mask = threshold_by_color(ycrcb, color)
    lbpair = get_light_bar_pair(mask)
    unwarped = unwarp_img_from_lbs(img, lbpair)
    shrinked = unwarped
    shrinked = cv2.cvtColor(shrinked, cv2.COLOR_BGR2GRAY)
    _, masked = cv2.threshold(shrinked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return masked


def classify_panel(img: np.ndarray) -> ArmorPanel:
    pass


def find_pose(panel: ArmorPanel) -> np.ndarray:
    pass


def guess_pitch_yaw(pose: np.ndarray) -> np.ndarray:
    pass
