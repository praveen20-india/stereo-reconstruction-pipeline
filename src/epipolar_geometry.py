"""
Epipolar Geometry Computations
===============================

Functions for computing epipolar lines, extracting epipoles through
multiple independent methods, and evaluating point-to-epipolar-line
distances as a quality metric for fundamental matrix estimation.
"""

import numpy as np


def compute_epipolar_lines(pts1, pts2, F):
    """Compute epipolar lines in both images from point correspondences and F.

    For a point x1 in image 1, the corresponding epipolar line in image 2
    is l2 = F x1. Similarly, l1 = F^T x2 gives the epipolar line in image 1
    corresponding to point x2.

    Parameters
    ----------
    pts1 : ndarray, shape (2, N) or (3, N)
        Points in image 1. If (2, N), homogeneous coordinates are appended.
    pts2 : ndarray, shape (2, N) or (3, N)
        Corresponding points in image 2.
    F : ndarray, shape (3, 3)
        Fundamental matrix.

    Returns
    -------
    lines1 : ndarray, shape (3, N)
        Epipolar lines in image 1 (coefficients a, b, c for ax + by + c = 0).
    lines2 : ndarray, shape (3, N)
        Epipolar lines in image 2.
    coeffs1 : ndarray, shape (2, N)
        Slope-intercept coefficients [m; q] for lines in image 1.
    coeffs2 : ndarray, shape (2, N)
        Slope-intercept coefficients [m; q] for lines in image 2.
    """
    if pts1.shape[0] == 2:
        pts1 = np.vstack([pts1, np.ones((1, pts1.shape[1]))])
    if pts2.shape[0] == 2:
        pts2 = np.vstack([pts2, np.ones((1, pts2.shape[1]))])

    lines2 = F @ pts1
    lines1 = F.T @ pts2

    eps = 1e-12
    b1 = np.where(np.abs(lines1[1]) < eps, np.sign(lines1[1]) * eps + eps, lines1[1])
    b2 = np.where(np.abs(lines2[1]) < eps, np.sign(lines2[1]) * eps + eps, lines2[1])

    coeffs1 = np.vstack([-lines1[0] / b1, -lines1[2] / b1])
    coeffs2 = np.vstack([-lines2[0] / b2, -lines2[2] / b2])

    return lines1, lines2, coeffs1, coeffs2


def compute_epipoles_svd(F):
    """Extract epipoles from the null spaces of the fundamental matrix.

    The left epipole (in image 1) lies in the right null space of F,
    and the right epipole (in image 2) lies in the left null space of F.

    Parameters
    ----------
    F : ndarray, shape (3, 3)
        Fundamental matrix.

    Returns
    -------
    epipole1 : ndarray, shape (2,)
        Epipole in image 1 (inhomogeneous coordinates).
    epipole2 : ndarray, shape (2,)
        Epipole in image 2 (inhomogeneous coordinates).
    """
    U, _, Vt = np.linalg.svd(F)

    e1_homogeneous = Vt[-1, :]
    epipole1 = e1_homogeneous[:2] / (e1_homogeneous[2] + 1e-12)

    e2_homogeneous = U[:, -1]
    epipole2 = e2_homogeneous[:2] / (e2_homogeneous[2] + 1e-12)

    return epipole1, epipole2


def compute_epipoles_from_lines(pts1, pts2, F):
    """Compute epipoles as the intersection of epipolar lines.

    All epipolar lines in an image pass through the corresponding epipole.
    The intersection is found by solving a minimal linear system formed
    from two epipolar lines.

    Parameters
    ----------
    pts1 : ndarray, shape (2, N)
        Points in image 1 (at least 2 correspondences needed).
    pts2 : ndarray, shape (2, N)
        Corresponding points in image 2.
    F : ndarray, shape (3, 3)
        Fundamental matrix.

    Returns
    -------
    epipole1 : ndarray, shape (2,)
        Epipole in image 1.
    epipole2 : ndarray, shape (2,)
        Epipole in image 2.
    """
    lines1, lines2, _, _ = compute_epipolar_lines(pts1, pts2, F)

    A1 = lines1[:, :2].T
    _, _, Vt1 = np.linalg.svd(A1)
    epipole1 = Vt1[-1, :2] / (Vt1[-1, 2] + 1e-12)

    A2 = lines2[:, :2].T
    _, _, Vt2 = np.linalg.svd(A2)
    epipole2 = Vt2[-1, :2] / (Vt2[-1, 2] + 1e-12)

    return epipole1, epipole2


def compute_epipoles_from_cameras(P1, P2):
    """Compute epipoles by projecting camera centers into the other view.

    The epipole in image 1 is the projection of camera 2's center through P1,
    and vice versa.

    Parameters
    ----------
    P1 : ndarray, shape (3, 4)
        Projection matrix of camera 1.
    P2 : ndarray, shape (3, 4)
        Projection matrix of camera 2.

    Returns
    -------
    epipole1 : ndarray, shape (2,)
        Epipole in image 1.
    epipole2 : ndarray, shape (2,)
        Epipole in image 2.
    """
    _, _, Vt1 = np.linalg.svd(P1)
    C1 = Vt1[-1, :]

    _, _, Vt2 = np.linalg.svd(P2)
    C2 = Vt2[-1, :]

    ep1_homogeneous = P1 @ C2
    epipole1 = ep1_homogeneous[:2] / (ep1_homogeneous[2] + 1e-12)

    ep2_homogeneous = P2 @ C1
    epipole2 = ep2_homogeneous[:2] / (ep2_homogeneous[2] + 1e-12)

    return epipole1, epipole2


def compute_point_line_distance(lines, points):
    """Compute the algebraic distance from points to their corresponding lines.

    Uses the standard formula d = |ax + by + c| / sqrt(a^2 + b^2) for
    the distance from point (x, y) to line ax + by + c = 0.

    Parameters
    ----------
    lines : ndarray, shape (3, N)
        Line coefficients (a, b, c) for each correspondence.
    points : ndarray, shape (2, N) or (3, N)
        Point coordinates. If (2, N), homogeneous ones are appended.

    Returns
    -------
    statistics : tuple (sum, mean, std)
        Summary statistics of the distance distribution.
    distances : ndarray, shape (N,)
        Per-correspondence absolute distances.
    """
    if points.shape[0] == 2:
        points = np.vstack([points, np.ones((1, points.shape[1]))])

    numerator = np.abs(np.sum(lines * points, axis=0))
    denominator = np.sqrt(lines[0] ** 2 + lines[1] ** 2 + 1e-18)
    distances = numerator / denominator

    return (float(np.sum(distances)), float(np.mean(distances)), float(np.std(distances))), distances


def compute_epipolar_distances(pts1, pts2, F):
    """Evaluate the quality of F by measuring point-to-epipolar-line distances.

    For a perfect fundamental matrix, every point should lie exactly on its
    corresponding epipolar line, yielding zero distance.

    Parameters
    ----------
    pts1 : ndarray, shape (2, N)
        Points in image 1.
    pts2 : ndarray, shape (2, N)
        Corresponding points in image 2.
    F : ndarray, shape (3, 3)
        Fundamental matrix.

    Returns
    -------
    d1 : ndarray, shape (N,)
        Distances from image 1 points to their epipolar lines.
    d2 : ndarray, shape (N,)
        Distances from image 2 points to their epipolar lines.
    """
    lines1, lines2, _, _ = compute_epipolar_lines(pts1, pts2, F)
    _, d1 = compute_point_line_distance(lines1, pts1)
    _, d2 = compute_point_line_distance(lines2, pts2)
    return d1, d2
