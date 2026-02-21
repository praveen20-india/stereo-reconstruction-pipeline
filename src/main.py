#!/usr/bin/env python3
"""
Multi-View Geometry Pipeline
==============================

End-to-end pipeline for fundamental matrix estimation, epipolar geometry
analysis, essential matrix decomposition, triangulation, and robustness
evaluation under noise. Generates all result visualizations.

Usage
-----
    python -m src.main

All result images are saved to ``report/result_images/``.
"""

import os
import sys
import numpy as np

from .feature_matching import (
    build_camera_intrinsics,
    build_rotation_matrix,
    build_projection_matrix,
    generate_3d_scene_points,
    generate_synthetic_correspondences,
    add_gaussian_noise,
)
from .fundamental_matrix import (
    eight_point_algorithm,
    compute_analytical_fundamental,
    sampson_distance,
)
from .epipolar_geometry import (
    compute_epipolar_lines,
    compute_epipoles_svd,
    compute_epipoles_from_lines,
    compute_epipoles_from_cameras,
    compute_epipolar_distances,
)
from .essential_matrix import (
    compute_essential_matrix,
    recover_camera_pose,
)
from .triangulation import (
    linear_triangulation,
    compute_reprojection_error,
)
from .visualization import (
    plot_projected_points,
    plot_epipolar_lines,
    plot_inlier_matches,
    plot_epipole_scatter,
    plot_3d_points,
    plot_reprojection_errors,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "result_images")
IMAGE_SIZE = (256, 256)
NOISE_SIGMA = 1.0
NUM_NOISE_TRIALS = 100
RANDOM_SEED = 42


def setup_cameras():
    """Configure a stereo camera pair with known intrinsics and extrinsics.

    Returns
    -------
    K1, K2 : ndarray, shape (3, 3)
        Intrinsic matrices.
    P1, P2 : ndarray, shape (3, 4)
        Projection matrices.
    R : ndarray, shape (3, 3)
        Relative rotation (camera 2 w.r.t. camera 1 frame).
    t : ndarray, shape (3,)
        Relative translation vector.
    """
    K1 = build_camera_intrinsics(au=100, av=120, u0=128, v0=128)
    K2 = build_camera_intrinsics(au=90, av=110, u0=128, v0=128)

    R1 = np.eye(3)
    t1 = np.zeros(3)

    rotation_angles = (0.1, np.pi / 4, 0.2)
    t2 = np.array([-1000, 190, 230])
    R2 = build_rotation_matrix(rotation_angles)

    P1 = build_projection_matrix(K1, R1, t1)
    P2 = build_projection_matrix(K2, R2, t2)

    return K1, K2, P1, P2, R2, t2


def run_fundamental_estimation(pts1, pts2, K1, K2, R, t):
    """Estimate and validate the fundamental matrix.

    Returns
    -------
    F_computed : ndarray
        Estimated fundamental matrix.
    F_analytical : ndarray
        Ground-truth analytical fundamental matrix.
    """
    print("=" * 60)
    print("FUNDAMENTAL MATRIX ESTIMATION")
    print("=" * 60)

    F_analytical = compute_analytical_fundamental(K1, K2, R, t)
    print(f"Analytical F (rank {np.linalg.matrix_rank(F_analytical)}):")
    print(F_analytical)

    F_computed = eight_point_algorithm(pts1, pts2, enforce_rank=True, normalize=True)
    print(f"\nComputed F (rank {np.linalg.matrix_rank(F_computed)}):")
    print(F_computed)

    frobenius_error = np.linalg.norm(F_analytical - F_computed, "fro")
    print(f"\nFrobenius norm |F_analytical - F_computed|: {frobenius_error:.6f}")
    print(f"Condition number of computed F: {np.linalg.cond(F_computed):.2f}")

    return F_computed, F_analytical


def run_epipole_analysis(pts1, pts2, F, P1, P2):
    """Compute epipoles using three independent methods and compare.

    Returns
    -------
    epipole1, epipole2 : ndarray, shape (2,)
        Epipoles from the SVD method (primary).
    """
    print("\n" + "=" * 60)
    print("EPIPOLE COMPUTATION (THREE METHODS)")
    print("=" * 60)

    ep1_svd, ep2_svd = compute_epipoles_svd(F)
    ep1_lines, ep2_lines = compute_epipoles_from_lines(pts1, pts2, F)
    ep1_cam, ep2_cam = compute_epipoles_from_cameras(P1, P2)

    methods = [
        ("SVD of F", ep1_svd, ep2_svd),
        ("Epipolar line intersection", ep1_lines, ep2_lines),
        ("Camera center projection", ep1_cam, ep2_cam),
    ]

    for name, e1, e2 in methods:
        print(f"\n  {name}:")
        print(f"    Epipole 1: ({e1[0]:.4f}, {e1[1]:.4f})")
        print(f"    Epipole 2: ({e2[0]:.4f}, {e2[1]:.4f})")

    return ep1_svd, ep2_svd


def run_noise_sensitivity_analysis(pts1_clean, pts2_clean):
    """Evaluate normalization impact across repeated noise trials.

    Returns
    -------
    ep1_unnorm, ep2_unnorm : ndarray, shape (N, 2)
        Epipole estimates without normalization.
    ep1_norm, ep2_norm : ndarray, shape (N, 2)
        Epipole estimates with normalization.
    """
    print("\n" + "=" * 60)
    print(f"NOISE SENSITIVITY ANALYSIS ({NUM_NOISE_TRIALS} trials)")
    print("=" * 60)

    ep1_unnorm, ep2_unnorm = [], []
    ep1_norm, ep2_norm = [], []

    for i in range(NUM_NOISE_TRIALS):
        pts1_noisy = add_gaussian_noise(pts1_clean, sigma=NOISE_SIGMA)
        pts2_noisy = add_gaussian_noise(pts2_clean, sigma=NOISE_SIGMA)

        F_no_norm = eight_point_algorithm(pts1_noisy, pts2_noisy,
                                          enforce_rank=True, normalize=False)
        e1, e2 = compute_epipoles_svd(F_no_norm)
        ep1_unnorm.append(e1)
        ep2_unnorm.append(e2)

        F_norm = eight_point_algorithm(pts1_noisy, pts2_noisy,
                                       enforce_rank=True, normalize=True)
        e1, e2 = compute_epipoles_svd(F_norm)
        ep1_norm.append(e1)
        ep2_norm.append(e2)

        if (i + 1) % 25 == 0:
            print(f"  Completed {i + 1}/{NUM_NOISE_TRIALS} trials")

    return (np.array(ep1_unnorm), np.array(ep2_unnorm),
            np.array(ep1_norm), np.array(ep2_norm))


def run_triangulation_analysis(P1, P2, pts1, pts2, points_3d_gt):
    """Triangulate points and evaluate reprojection error.

    Returns
    -------
    points_3d_est : ndarray, shape (3, N)
        Triangulated 3D points.
    errors1, errors2 : ndarray
        Per-point reprojection errors.
    """
    print("\n" + "=" * 60)
    print("TRIANGULATION AND REPROJECTION ERROR")
    print("=" * 60)

    points_3d_est = linear_triangulation(P1, P2, pts1, pts2)

    reconstruction_error = np.mean(np.sqrt(
        np.sum((points_3d_gt - points_3d_est) ** 2, axis=0)
    ))
    print(f"Mean 3D reconstruction error: {reconstruction_error:.6f}")

    errors1, mean_err1 = compute_reprojection_error(points_3d_est, pts1, P1)
    errors2, mean_err2 = compute_reprojection_error(points_3d_est, pts2, P2)
    print(f"Mean reprojection error (Camera 1): {mean_err1:.6f} pixels")
    print(f"Mean reprojection error (Camera 2): {mean_err2:.6f} pixels")

    return points_3d_est, errors1, errors2


def main():
    """Execute the complete multi-view geometry pipeline."""
    np.random.seed(RANDOM_SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Camera setup ---
    K1, K2, P1, P2, R, t = setup_cameras()
    points_3d = generate_3d_scene_points()

    # --- Generate correspondences ---
    pts1_clean, pts2_clean = generate_synthetic_correspondences(points_3d, P1, P2)
    pts1_noisy = add_gaussian_noise(pts1_clean, sigma=NOISE_SIGMA)
    pts2_noisy = add_gaussian_noise(pts2_clean, sigma=NOISE_SIGMA)

    # --- Fundamental matrix estimation ---
    F_computed, F_analytical = run_fundamental_estimation(
        pts1_noisy, pts2_noisy, K1, K2, R, t
    )

    # --- Epipole analysis ---
    ep1, ep2 = run_epipole_analysis(pts1_clean, pts2_clean, F_computed, P1, P2)

    # --- Epipolar geometry ---
    _, _, coeffs1, coeffs2 = compute_epipolar_lines(pts1_clean, pts2_clean, F_computed)

    # --- Triangulation ---
    points_3d_est, errors1, errors2 = run_triangulation_analysis(
        P1, P2, pts1_clean, pts2_clean, points_3d
    )

    # --- Essential matrix and pose recovery ---
    print("\n" + "=" * 60)
    print("ESSENTIAL MATRIX AND POSE RECOVERY")
    print("=" * 60)
    E = compute_essential_matrix(F_computed, K1, K2)
    print(f"Essential matrix:\n{E}")
    R_recovered, t_recovered = recover_camera_pose(E, K1, K2, pts1_clean, pts2_clean)
    print(f"\nRecovered rotation:\n{R_recovered}")
    print(f"Recovered translation direction:\n{t_recovered.flatten()}")

    # --- Noise sensitivity analysis ---
    ep1_unnorm, ep2_unnorm, ep1_norm, ep2_norm = run_noise_sensitivity_analysis(
        pts1_clean, pts2_clean
    )

    # --- Sampson distance ---
    sampson_errors = sampson_distance(F_computed, pts1_clean, pts2_clean)
    print(f"\nMean Sampson distance: {np.mean(sampson_errors):.8f}")

    # --- Epipolar line distances ---
    d1, d2 = compute_epipolar_distances(pts1_clean, pts2_clean, F_computed)
    print(f"Mean point-to-epipolar-line distance (Image 1): {np.mean(d1):.6f}")
    print(f"Mean point-to-epipolar-line distance (Image 2): {np.mean(d2):.6f}")

    # ===================================================================
    # Generate and save all result visualizations
    # ===================================================================
    print("\n" + "=" * 60)
    print("SAVING RESULT VISUALIZATIONS")
    print("=" * 60)

    plot_projected_points(
        pts1_clean, pts2_clean, IMAGE_SIZE,
        save_path=os.path.join(RESULTS_DIR, "matched_features.png"),
    )
    print("  Saved: matched_features.png")

    margins = ((-400, 300), (1, 400))
    plot_epipolar_lines(
        pts1_clean, pts2_clean, coeffs1, coeffs2, ep1, ep2,
        IMAGE_SIZE, margins,
        save_path=os.path.join(RESULTS_DIR, "epipolar_lines.png"),
    )
    print("  Saved: epipolar_lines.png")

    plot_inlier_matches(
        pts1_clean, pts2_clean, pts1_noisy, pts2_noisy,
        IMAGE_SIZE,
        save_path=os.path.join(RESULTS_DIR, "inlier_matches.png"),
    )
    print("  Saved: inlier_matches.png")

    plot_3d_points(
        points_3d_est,
        save_path=os.path.join(RESULTS_DIR, "3d_points_plot.png"),
    )
    print("  Saved: 3d_points_plot.png")

    plot_reprojection_errors(
        errors1, errors2,
        save_path=os.path.join(RESULTS_DIR, "reprojection_error_plot.png"),
    )
    print("  Saved: reprojection_error_plot.png")

    plot_epipole_scatter(
        ep1_unnorm, ep2_unnorm, ep1_norm, ep2_norm,
        NUM_NOISE_TRIALS,
        save_path=os.path.join(RESULTS_DIR, "normalization_comparison.png"),
    )
    print("  Saved: normalization_comparison.png")

    # --- Write evaluation summary ---
    summary_path = os.path.join(os.path.dirname(__file__), "..", "results", "evaluation_summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        f.write("Multi-View Geometry - Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of point correspondences: {pts1_clean.shape[1]}\n")
        f.write(f"Noise level (sigma): {NOISE_SIGMA} pixels\n")
        f.write(f"Normalization sensitivity trials: {NUM_NOISE_TRIALS}\n\n")
        f.write("Fundamental Matrix Estimation\n")
        f.write("-" * 40 + "\n")
        f.write(f"Frobenius error |F_analytical - F_computed|: "
                f"{np.linalg.norm(F_analytical - F_computed, 'fro'):.6f}\n")
        f.write(f"Condition number of F: {np.linalg.cond(F_computed):.2f}\n")
        f.write(f"Rank of computed F: {np.linalg.matrix_rank(F_computed)}\n\n")
        f.write("Epipole Locations (SVD method)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Epipole 1: ({ep1[0]:.4f}, {ep1[1]:.4f})\n")
        f.write(f"Epipole 2: ({ep2[0]:.4f}, {ep2[1]:.4f})\n\n")
        f.write("Triangulation Quality\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean reprojection error (Camera 1): {np.mean(errors1):.6f} pixels\n")
        f.write(f"Mean reprojection error (Camera 2): {np.mean(errors2):.6f} pixels\n")
        f.write(f"Mean 3D reconstruction error: {np.mean(np.sqrt(np.sum((points_3d - points_3d_est)**2, axis=0))):.6f}\n\n")
        f.write("Sampson Distance\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean: {np.mean(sampson_errors):.8f}\n")
        f.write(f"Max:  {np.max(sampson_errors):.8f}\n\n")
        f.write("Point-to-Epipolar-Line Distance\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean (Image 1): {np.mean(d1):.6f}\n")
        f.write(f"Mean (Image 2): {np.mean(d2):.6f}\n")

    print(f"  Saved: evaluation_summary.txt")
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
