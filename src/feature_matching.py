"""
Feature Matching and Synthetic Correspondence Generation
=========================================================

Provides utilities for generating synthetic point correspondences
from known camera configurations, as well as noise injection for
robustness evaluation of geometric estimation algorithms.
"""

import numpy as np
from .triangulation import project_points


def build_camera_intrinsics(au, av, u0, v0):
    """Construct a camera intrinsic calibration matrix.

    Parameters
    ----------
    au : float
        Focal length in horizontal pixels.
    av : float
        Focal length in vertical pixels.
    u0 : float
        Principal point horizontal coordinate.
    v0 : float
        Principal point vertical coordinate.

    Returns
    -------
    K : ndarray, shape (3, 3)
        Upper-triangular intrinsic matrix.
    """
    return np.array([
        [au, 0, u0],
        [0, av, v0],
        [0, 0, 1]
    ], dtype=float)


def build_rotation_matrix(angles):
    """Construct a 3D rotation matrix from Euler angles (X-Y-Z convention).

    Parameters
    ----------
    angles : tuple of float
        Rotation angles (rx, ry, rz) in radians around X, Y, Z axes.

    Returns
    -------
    R : ndarray, shape (3, 3)
        Combined rotation matrix R = Rx @ Ry @ Rz.
    """
    rx, ry, rz = angles

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])

    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    return Rx @ Ry @ Rz


def build_projection_matrix(K, R, t):
    """Construct the 3x4 camera projection matrix P = K [R^T | -R^T t].

    Parameters
    ----------
    K : ndarray, shape (3, 3)
        Camera intrinsic matrix.
    R : ndarray, shape (3, 3)
        World-to-camera rotation matrix.
    t : ndarray, shape (3,) or (3, 1)
        Camera position in world coordinates.

    Returns
    -------
    P : ndarray, shape (3, 4)
        Camera projection matrix.
    """
    t = np.asarray(t).reshape(3, 1)
    return K @ np.hstack([R.T, -R.T @ t])


def generate_3d_scene_points():
    """Generate a set of 3D scene points for evaluation.

    Creates a structured grid of 20 points distributed across the
    scene volume to provide diverse viewing geometry for testing.

    Returns
    -------
    points : ndarray, shape (3, 20)
        3D point coordinates in world frame.
    """
    return np.array([
        [100, 300, 500, 700, 900, 100, 300, 500, 700, 900,
         100, 300, 500, 700, 900, 100, 300, 500, 700, 900],
        [-400, -400, -400, -400, -400, -40, -40, -40, -40, -40,
         40, 40, 40, 40, 40, 400, 400, 400, 400, 400],
        [2000, 3000, 4000, 2000, 3000, 4000, 2000, 3000, 4000, 2000,
         3000, 4000, 2000, 3000, 4000, 2000, 3000, 4000, 2000, 3000]
    ], dtype=float)


def generate_synthetic_correspondences(points_3d, P1, P2):
    """Project 3D points through two cameras to create correspondences.

    Parameters
    ----------
    points_3d : ndarray, shape (3, N)
        3D scene points.
    P1 : ndarray, shape (3, 4)
        Projection matrix of camera 1.
    P2 : ndarray, shape (3, 4)
        Projection matrix of camera 2.

    Returns
    -------
    pts1 : ndarray, shape (2, N)
        Projected points in image 1.
    pts2 : ndarray, shape (2, N)
        Corresponding projected points in image 2.
    """
    pts1 = project_points(points_3d, P1)
    pts2 = project_points(points_3d, P2)
    return pts1, pts2


def add_gaussian_noise(points, sigma=1.0):
    """Add independent Gaussian noise to 2D point coordinates.

    Parameters
    ----------
    points : ndarray, shape (2, N)
        Clean point coordinates.
    sigma : float
        Standard deviation of the Gaussian noise in pixels.

    Returns
    -------
    noisy_points : ndarray, shape (2, N)
        Perturbed point coordinates.
    """
    noise = np.random.normal(0, sigma, points.shape)
    return points + noise
