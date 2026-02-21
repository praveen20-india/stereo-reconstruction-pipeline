"""
Essential Matrix and Camera Pose Recovery
==========================================

Provides functions for computing the essential matrix from the fundamental
matrix and intrinsic parameters, decomposing it into rotation and translation
components, and selecting the geometrically valid camera pose through the
cheirality (positive depth) constraint.
"""

import numpy as np
from .triangulation import linear_triangulation


def compute_essential_matrix(F, K1, K2):
    """Compute the essential matrix from the fundamental matrix and intrinsics.

    The essential matrix relates corresponding points in normalized image
    coordinates: x2^T E x1 = 0, where E = K2^T F K1.

    Parameters
    ----------
    F : ndarray, shape (3, 3)
        Fundamental matrix.
    K1 : ndarray, shape (3, 3)
        Intrinsic matrix of camera 1.
    K2 : ndarray, shape (3, 3)
        Intrinsic matrix of camera 2.

    Returns
    -------
    E : ndarray, shape (3, 3)
        Essential matrix with enforced singular value constraint [s, s, 0].
    """
    E = K2.T @ F @ K1

    # Enforce the constraint that two singular values must be equal
    U, S, Vt = np.linalg.svd(E)
    s = (S[0] + S[1]) / 2.0
    E = U @ np.diag([s, s, 0.0]) @ Vt

    return E


def decompose_essential_matrix(E):
    """Decompose the essential matrix into four possible (R, t) solutions.

    The SVD of E yields two possible rotations and two possible translation
    directions, giving four candidate camera poses.

    Parameters
    ----------
    E : ndarray, shape (3, 3)
        Essential matrix.

    Returns
    -------
    solutions : list of tuple(ndarray, ndarray)
        Four (R, t) pairs where R is (3, 3) and t is (3, 1).
    """
    U, _, Vt = np.linalg.svd(E)

    # Ensure proper rotation (det = +1)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=float)

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2:3]  # shape (3, 1)

    solutions = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t),
    ]

    return solutions


def recover_camera_pose(E, K1, K2, pts1, pts2):
    """Select the correct camera pose using the cheirality constraint.

    Among the four possible (R, t) decompositions of E, the physically
    valid solution is the one where triangulated 3D points lie in front
    of both cameras (positive depth).

    Parameters
    ----------
    E : ndarray, shape (3, 3)
        Essential matrix.
    K1 : ndarray, shape (3, 3)
        Intrinsic matrix of camera 1.
    K2 : ndarray, shape (3, 3)
        Intrinsic matrix of camera 2.
    pts1 : ndarray, shape (2, N)
        Points in image 1.
    pts2 : ndarray, shape (2, N)
        Corresponding points in image 2.

    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix of camera 2 relative to camera 1.
    t : ndarray, shape (3, 1)
        Translation vector of camera 2 relative to camera 1.
    """
    solutions = decompose_essential_matrix(E)

    # Camera 1 is at the origin: P1 = K1 [I | 0]
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])

    best_solution = None
    max_positive_depth = 0

    for R, t in solutions:
        P2 = K2 @ np.hstack([R, t])

        points_3d = linear_triangulation(P1, P2, pts1, pts2)

        # Check depth in camera 1 (z > 0)
        depth_cam1 = points_3d[2, :]

        # Transform to camera 2 frame and check depth
        pts_cam2 = R @ points_3d[:3, :] + t
        depth_cam2 = pts_cam2[2, :]

        positive_count = np.sum((depth_cam1 > 0) & (depth_cam2 > 0))

        if positive_count > max_positive_depth:
            max_positive_depth = positive_count
            best_solution = (R, t)

    if best_solution is None:
        raise ValueError("Failed to recover valid camera pose from essential matrix.")

    return best_solution
