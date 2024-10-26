import cv2
import numpy as np

from .panelclasses import PanelColor


def classify_color(ycrcb: np.ndarray) -> PanelColor:
    spread_factor = 28
    middle = 128
    np.seterr(divide='ignore')
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    red_factor = np.log(np.count_nonzero(cr < middle - spread_factor)) - np.log(
        np.count_nonzero(cr > middle + spread_factor))

    blue_factor = np.log(np.count_nonzero(cb < middle - spread_factor)) - np.log(
        np.count_nonzero(cb > middle + spread_factor))
    np.seterr(divide='ignore')
    # TODO: Handle
    if np.sign(red_factor) == np.sign(blue_factor):
        raise RuntimeError('Purple Armor detected')
    if red_factor < blue_factor:
        return PanelColor.RED
    else:
        return PanelColor.BLUE


def threshold_by_color(ycrcb: np.ndarray, color: PanelColor) -> np.ndarray:
    channel = None
    match color:
        case PanelColor.RED:
            channel = 1
        case PanelColor.BLUE:
            channel = 2

    return cv2.threshold(ycrcb[:, :, channel], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
