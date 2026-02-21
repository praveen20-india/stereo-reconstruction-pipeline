"""
Visualization Utilities
========================

Publication-quality plotting functions for multi-view geometry results,
including projected point displays, epipolar line overlays, epipole
markers, normalization sensitivity scatter plots, 3D scene visualization,
and reprojection error analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def save_figure(fig, filepath):
    """Save a matplotlib figure to disk with tight bounding box.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    filepath : str
        Output file path (PNG format recommended).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_projected_points(pts1, pts2, image_size, save_path=None):
    """Display projected point correspondences on both image planes.

    Parameters
    ----------
    pts1 : ndarray, shape (2, N)
        Points in image 1.
    pts2 : ndarray, shape (2, N)
        Corresponding points in image 2.
    image_size : tuple (W, H)
        Image dimensions in pixels.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    W, H = image_size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pts, title in [(ax1, pts1, "Camera 1"), (ax2, pts2, "Camera 2")]:
        ax.add_patch(plt.Rectangle((0, 0), W, H, fill=False, edgecolor="gray", linewidth=1.5))
        ax.scatter(pts[0], pts[1], s=40, marker="o", alpha=0.7, zorder=5, edgecolors="k", linewidths=0.5)
        ax.set_xlim(-0.15 * W, 1.15 * W)
        ax.set_ylim(1.15 * H, -0.15 * H)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("u (pixels)")
        ax.set_ylabel("v (pixels)")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Point Correspondences on Image Planes", fontsize=15, fontweight="bold")
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_epipolar_lines(pts1, pts2, coeffs1, coeffs2, epipole1, epipole2,
                        image_size, margins, save_path=None):
    """Draw epipolar lines and epipoles on both image planes.

    Parameters
    ----------
    pts1, pts2 : ndarray, shape (2, N)
        Point correspondences.
    coeffs1, coeffs2 : ndarray, shape (2, N)
        Slope-intercept coefficients [m; q] for epipolar lines.
    epipole1, epipole2 : ndarray, shape (2,)
        Epipole locations.
    image_size : tuple (W, H)
        Image dimensions.
    margins : tuple ((x0_1, x1_1), (x0_2, x1_2))
        X-range for drawing lines in each image.
    save_path : str, optional
        If provided, save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    W, H = image_size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pts, coeffs, epipole, margin, title in [
        (ax1, pts1, coeffs1, epipole1, margins[0], "Image 1 - Epipolar Lines"),
        (ax2, pts2, coeffs2, epipole2, margins[1], "Image 2 - Epipolar Lines"),
    ]:
        ax.add_patch(plt.Rectangle((0, 0), W, H, fill=False, edgecolor="gray", linewidth=1.5))
        ax.scatter(pts[0], pts[1], s=40, marker="o", c="green", alpha=0.7, zorder=5,
                   edgecolors="k", linewidths=0.5, label="Points")

        x_start, x_end = margin
        m_vals, q_vals = coeffs[0], coeffs[1]
        x_range = np.array([x_start, x_end])

        for i in range(m_vals.size):
            y_vals = x_range * m_vals[i] + q_vals[i]
            ax.plot(x_range, y_vals, color="steelblue", alpha=0.6, linewidth=0.8)

        ax.scatter([epipole[0]], [epipole[1]], s=100, marker="*", c="red",
                   zorder=10, label="Epipole", edgecolors="k")

        ep_margin = 50
        ax.set_xlim(min(0, epipole[0] - ep_margin), max(W, epipole[0] + ep_margin))
        ax.set_ylim(min(0, epipole[1] - ep_margin), max(H, epipole[1] + ep_margin))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("u (pixels)")
        ax.set_ylabel("v (pixels)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Epipolar Geometry Visualization", fontsize=15, fontweight="bold")
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_inlier_matches(pts1_clean, pts2_clean, pts1_noisy, pts2_noisy,
                        image_size, save_path=None):
    """Visualize clean vs. noisy point correspondences side by side.

    Parameters
    ----------
    pts1_clean, pts2_clean : ndarray, shape (2, N)
        Noise-free projected points.
    pts1_noisy, pts2_noisy : ndarray, shape (2, N)
        Noise-perturbed projected points.
    image_size : tuple (W, H)
        Image dimensions.
    save_path : str, optional
        If provided, save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    W, H = image_size
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    datasets = [
        (axes[0, 0], pts1_clean, "Camera 1 - Clean"),
        (axes[0, 1], pts2_clean, "Camera 2 - Clean"),
        (axes[1, 0], pts1_noisy, "Camera 1 - With Noise"),
        (axes[1, 1], pts2_noisy, "Camera 2 - With Noise"),
    ]

    for ax, pts, title in datasets:
        ax.add_patch(plt.Rectangle((0, 0), W, H, fill=False, edgecolor="gray", linewidth=1.5))
        color = "tab:blue" if "Clean" in title else "tab:orange"
        ax.scatter(pts[0], pts[1], s=40, c=color, alpha=0.7, edgecolors="k", linewidths=0.5)
        ax.set_xlim(-0.15 * W, 1.15 * W)
        ax.set_ylim(1.15 * H, -0.15 * H)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("u (pixels)")
        ax.set_ylabel("v (pixels)")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Clean vs. Noisy Point Correspondences", fontsize=15, fontweight="bold")
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_epipole_scatter(ep1_unnorm, ep2_unnorm, ep1_norm, ep2_norm,
                         num_iterations, save_path=None):
    """Scatter plot comparing epipole estimates with and without normalization.

    Parameters
    ----------
    ep1_unnorm, ep2_unnorm : ndarray, shape (N, 2)
        Epipole estimates without Hartley normalization.
    ep1_norm, ep2_norm : ndarray, shape (N, 2)
        Epipole estimates with normalization.
    num_iterations : int
        Number of noise trials.
    save_path : str, optional
        If provided, save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, ep_unnorm, ep_norm, title in [
        (ax1, ep1_unnorm, ep1_norm, "Epipole in Image 1"),
        (ax2, ep2_unnorm, ep2_norm, "Epipole in Image 2"),
    ]:
        ax.scatter(ep_unnorm[:, 0], ep_unnorm[:, 1], alpha=0.5, s=30,
                   label="Without normalization", color="red")
        ax.scatter(ep_norm[:, 0], ep_norm[:, 1], alpha=0.5, s=30,
                   label="With normalization", color="blue")
        ax.set_xlabel("u coordinate")
        ax.set_ylabel("v coordinate")
        ax.set_title(f"{title} ({num_iterations} trials)", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Effect of Hartley Normalization on Epipole Stability",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_3d_points(points_3d, save_path=None):
    """Visualize a 3D point cloud in world coordinates.

    Parameters
    ----------
    points_3d : ndarray, shape (3, N)
        3D point coordinates.
    save_path : str, optional
        If provided, save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points_3d[0], points_3d[1], points_3d[2],
               s=50, c=points_3d[2], cmap="viridis", edgecolors="k", linewidths=0.5)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("3D Scene Point Cloud", fontsize=14, fontweight="bold")

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_reprojection_errors(errors_cam1, errors_cam2, save_path=None):
    """Bar and distribution plot of reprojection errors for both cameras.

    Parameters
    ----------
    errors_cam1 : ndarray, shape (N,)
        Per-point reprojection errors from camera 1.
    errors_cam2 : ndarray, shape (N,)
        Per-point reprojection errors from camera 2.
    save_path : str, optional
        If provided, save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    n = len(errors_cam1)
    indices = np.arange(n)

    ax1.bar(indices - 0.15, errors_cam1, 0.3, label="Camera 1", color="steelblue", alpha=0.8)
    ax1.bar(indices + 0.15, errors_cam2, 0.3, label="Camera 2", color="coral", alpha=0.8)
    ax1.set_xlabel("Point Index")
    ax1.set_ylabel("Reprojection Error (pixels)")
    ax1.set_title("Per-Point Reprojection Errors", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2, axis="y")

    all_errors = np.concatenate([errors_cam1, errors_cam2])
    ax2.hist(all_errors, bins=15, color="mediumpurple", alpha=0.75, edgecolor="k")
    ax2.axvline(np.mean(all_errors), color="red", linestyle="--", linewidth=1.5,
                label=f"Mean: {np.mean(all_errors):.4f} px")
    ax2.set_xlabel("Reprojection Error (pixels)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Error Distribution", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    fig.suptitle("Reprojection Error Analysis", fontsize=15, fontweight="bold")
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig
