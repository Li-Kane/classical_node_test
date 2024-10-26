import numpy as np


def get_shorter_edge_midpoints(rect):
    (cx, cy), (width, height), angle = rect
    angle = np.radians(angle)

    if width < height:
        shorter = width
        longer = height
    else:
        shorter = height
        longer = width
    # get the midpoints of the shorter edge

    vecs = np.array([[-longer / 2, 0], [longer / 2, 0]])
    if shorter == width:
        vecs = np.roll(vecs, 1, axis=1)

    rotmatr = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    vecs = rotmatr @ vecs.T
    vecs += np.array([[cx], [cy]])
    return tuple(map(tuple, vecs.T))
