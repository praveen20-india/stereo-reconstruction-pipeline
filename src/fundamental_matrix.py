"""
Fundamental Matrix Estimation
=============================

Implements the normalized 8-point algorithm for estimating the fundamental
matrix from point correspondences, including Hartley normalization,
rank-2 enforcement via SVD, and the Sampson distance error metric.
"""

import numpy as np


def normalize_points(pts):
    """Apply isotropic normalization to 2D point coordinates.

    Translates points so their centroid is at the origin, then scales
    them so the average distance from the origin is sqrt(2). This
    conditioning dramatically improves numerical stability of the
    8-point algorithm.

    Parameters
    ----------
    pts : ndarray, shape (2, N)
        Raw 2D point coordinates.

    Returns
    -------
    T : ndarray, shape (3, 3)
        The normalization transformation matrix.
    pts_normalized : ndarray, shape (3, N)
        Homogeneous normalized coordinates.
    """
    assert pts.shape[0] == 2, "Input must be a 2xN array of point coordinates."

    centroid = np.mean(pts, axis=1, keepdims=True)
    pts_centered = pts - centroid

    rms_distance = np.sqrt(np.mean(np.sum(pts_centered ** 2, axis=0)))
    scale = np.sqrt(2) / (rms_distance + 1e-12)

    T = np.array([
        [scale, 0, -scale * centroid[0, 0]],
        [0, scale, -scale * centroid[1, 0]],
        [0, 0, 1]
    ], dtype=float)

    pts_homogeneous = np.vstack([pts, np.ones((1, pts.shape[1]))])
    pts_normalized = T @ pts_homogeneous

    return T, pts_normalized


def enforce_rank2(F):
    """Enforce the rank-2 constraint on a fundamental matrix.

    The fundamental matrix must be singular (rank 2) since epipolar lines
    must converge at the epipole. This is enforced by zeroing the smallest
    singular value after SVD decomposition.

    Parameters
    ----------
    F : ndarray, shape (3, 3)
        Unconstrained fundamental matrix estimate.

    Returns
    -------
    F_rank2 : ndarray, shape (3, 3)
        Rank-2 fundamental matrix.
    """
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0.0
    return U @ np.diag(S) @ Vt


def eight_point_algorithm(pts1, pts2, enforce_rank=True, normalize=True):
    """Estimate the fundamental matrix using the normalized 8-point algorithm.

    Constructs a linear system from point correspondences using the
    epipolar constraint x2^T F x1 = 0, then solves via SVD. Optionally
    applies Hartley normalization and rank-2 enforcement.

    Parameters
    ----------
    pts1 : ndarray, shape (2, N)
        Point coordinates in the first image, N >= 8.
    pts2 : ndarray, shape (2, N)
        Corresponding point coordinates in the second image.
    enforce_rank : bool, optional
        If True, enforce the rank-2 singularity constraint. Default True.
    normalize : bool, optional
        If True, apply Hartley normalization before estimation. Default True.

    Returns
    -------
    F : ndarray, shape (3, 3)
        Estimated fundamental matrix, normalized to unit Frobenius norm.
    """
    assert pts1.shape[1] >= 8, "At least 8 point correspondences are required."
    assert pts1.shape == pts2.shape, "Point arrays must have identical shape."

    if normalize:
        T1, x1 = normalize_points(pts1)
        T2, x2 = normalize_points(pts2)
    else:
        x1 = np.vstack([pts1, np.ones((1, pts1.shape[1]))])
        x2 = np.vstack([pts2, np.ones((1, pts2.shape[1]))])
        T1 = np.eye(3)
        T2 = np.eye(3)

    # Build the design matrix from the epipolar constraint
    n = x1.shape[1]
    A = np.zeros((n, 9))
    for i in range(n):
        u1, v1, w1 = x1[:, i]
        u2, v2, w2 = x2[:, i]
        A[i] = [
            u2 * u1, u2 * v1, u2 * w1,
            v2 * u1, v2 * v1, v2 * w1,
            w2 * u1, w2 * v1, w2 * w1,
        ]

    # Solve the homogeneous system via SVD
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    F_normalized = Vt[-1].reshape(3, 3)

    if enforce_rank:
        F_normalized = enforce_rank2(F_normalized)

    # Denormalize
    F = T2.T @ F_normalized @ T1

    # Scale to unit Frobenius norm
    F = F / (np.linalg.norm(F) + 1e-12)

    return F


def compute_analytical_fundamental(K1, K2, R, t):
    """Compute the fundamental matrix analytically from known camera parameters.

    Uses the relation F = K2^{-T} R^T [t]_x K1^{-1} where [t]_x is the
    skew-symmetric matrix of the translation vector.

    Parameters
    ----------
    K1 : ndarray, shape (3, 3)
        Intrinsic matrix of camera 1.
    K2 : ndarray, shape (3, 3)
        Intrinsic matrix of camera 2.
    R : ndarray, shape (3, 3)
        Rotation matrix from camera 1 to camera 2.
    t : ndarray, shape (3,)
        Translation vector from camera 1 to camera 2.

    Returns
    -------
    F : ndarray, shape (3, 3)
        Analytical fundamental matrix.
    """
    tx, ty, tz = t[0], t[1], t[2]
    t_cross = np.array([
        [0, -tz, ty],
        [tz, 0, -tx],
        [-ty, tx, 0]
    ])

    F = np.linalg.inv(K2).T @ R.T @ t_cross @ np.linalg.inv(K1)
    F = F / (F[2, 2] + 1e-12)

    return F


def sampson_distance(F, pts1, pts2):
    """Compute the Sampson distance for point correspondences.

    The Sampson distance provides a first-order approximation to the
    geometric reprojection error for each correspondence under the
    epipolar constraint.

    Parameters
    ----------
    F : ndarray, shape (3, 3)
        Fundamental matrix.
    pts1 : ndarray, shape (2, N)
        Points in the first image.
    pts2 : ndarray, shape (2, N)
        Corresponding points in the second image.

    Returns
    -------
    distances : ndarray, shape (N,)
        Sampson distance for each correspondence.
    """
    x1 = np.vstack([pts1, np.ones((1, pts1.shape[1]))])
    x2 = np.vstack([pts2, np.ones((1, pts2.shape[1]))])

    Fx1 = F @ x1
    Ftx2 = F.T @ x2

    numerator = np.sum(x2 * Fx1, axis=0) ** 2
    denominator = Fx1[0] ** 2 + Fx1[1] ** 2 + Ftx2[0] ** 2 + Ftx2[1] ** 2 + 1e-18

    return numerator / denominator
