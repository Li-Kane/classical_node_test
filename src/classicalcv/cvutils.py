import cv2
import numpy as np


def get_bb_of_largest_cc(mask: np.ndarray):
    # get the minAreaRect of the largest connected component in the mask
    # get connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    # mask out largest
    largest_cc = np.zeros_like(mask)
    largest_cc[labels == np.argmax(stats[1:, 4]) + 1] = 255
    # get contours
    contours, _ = cv2.findContours(largest_cc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # get bounding box
    return cv2.minAreaRect(contours[0])
