"""
Triangulation and Point Projection
====================================

Implements linear triangulation via the Direct Linear Transform (DLT)
method, projection of 3D world points onto 2D image planes, and
reprojection error computation for reconstruction quality assessment.
"""

import numpy as np


def project_points(points_3d, P):
    """Project 3D world points onto a 2D image plane.

    Applies the camera projection matrix P to convert 3D homogeneous
    world coordinates into 2D inhomogeneous image coordinates.

    Parameters
    ----------
    points_3d : ndarray, shape (3, N) or (4, N)
        3D points in world coordinates. If (3, N), a homogeneous row
        of ones is appended automatically.
    P : ndarray, shape (3, 4)
        Camera projection matrix.

    Returns
    -------
    points_2d : ndarray, shape (2, N)
        Projected 2D image coordinates.
    """
    if points_3d.shape[0] == 3:
        points_3d = np.vstack([points_3d, np.ones((1, points_3d.shape[1]))])

    projected = P @ points_3d

    z = np.where(np.abs(projected[2]) < 1e-12, 1e-12, projected[2])
    points_2d = np.vstack([projected[0] / z, projected[1] / z])

    return points_2d


def linear_triangulation(P1, P2, pts1, pts2):
    """Triangulate 3D points from two-view correspondences using DLT.

    For each point correspondence, constructs a 4x4 linear system from
    the projection equations of both views and solves via SVD to recover
    the 3D position.

    Parameters
    ----------
    P1 : ndarray, shape (3, 4)
        Projection matrix of camera 1.
    P2 : ndarray, shape (3, 4)
        Projection matrix of camera 2.
    pts1 : ndarray, shape (2, N)
        Points in image 1.
    pts2 : ndarray, shape (2, N)
        Corresponding points in image 2.

    Returns
    -------
    points_3d : ndarray, shape (3, N)
        Triangulated 3D points in inhomogeneous world coordinates.
    """
    n = pts1.shape[1]
    points_3d = np.zeros((3, n))

    for i in range(n):
        x1, y1 = pts1[0, i], pts1[1, i]
        x2, y2 = pts2[0, i], pts2[1, i]

        A = np.array([
            x1 * P1[2, :] - P1[0, :],
            y1 * P1[2, :] - P1[1, :],
            x2 * P2[2, :] - P2[0, :],
            y2 * P2[2, :] - P2[1, :],
        ])

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        points_3d[:, i] = X[:3] / (X[3] + 1e-12)

    return points_3d


def compute_reprojection_error(points_3d, pts_observed, P):
    """Compute the reprojection error between observed and projected points.

    Measures the Euclidean distance in pixels between each observed 2D
    point and the reprojection of its corresponding triangulated 3D point.

    Parameters
    ----------
    points_3d : ndarray, shape (3, N)
        Triangulated 3D points.
    pts_observed : ndarray, shape (2, N)
        Observed 2D point coordinates.
    P : ndarray, shape (3, 4)
        Projection matrix used for reprojection.

    Returns
    -------
    errors : ndarray, shape (N,)
        Per-point reprojection error in pixels.
    mean_error : float
        Mean reprojection error across all points.
    """
    pts_reprojected = project_points(points_3d, P)

    errors = np.sqrt(np.sum((pts_observed - pts_reprojected) ** 2, axis=0))
    mean_error = float(np.mean(errors))

    return errors, mean_error
