# Epipolar Geometry and Stereo Reconstruction from Image Correspondences

## 1. Introduction

Recovering three-dimensional structure from two-dimensional images is a central challenge in computer vision. When two cameras observe the same scene from different viewpoints, the geometric relationship between the resulting images is governed by **epipolar geometry** — a projective framework that constrains where corresponding points can appear across views.

This project implements a complete pipeline for estimating epipolar geometry from point correspondences, recovering camera pose, and triangulating 3D scene structure. The implementation covers the **normalized 8-point algorithm** for fundamental matrix estimation, **essential matrix decomposition** for pose recovery, and **linear triangulation** via the Direct Linear Transform. A systematic evaluation quantifies the impact of Hartley normalization on estimation robustness under measurement noise.

The pipeline is validated on synthetic data generated from a known two-camera configuration, enabling direct comparison between estimated and ground-truth geometric quantities.

---

## 2. Camera Geometry and Epipolar Constraints

### 2.1 The Pinhole Camera Model

A pinhole camera projects a 3D world point **X** = (X, Y, Z, 1)^T to a 2D image point **x** = (u, v, 1)^T through the projection matrix:

```
x ~ P X = K [R | t] X
```

where **K** is the 3x3 upper-triangular intrinsic matrix encoding focal length and principal point, and **[R | t]** represents the extrinsic camera pose (rotation and translation from world to camera coordinates).

### 2.2 Epipolar Geometry

Consider two cameras with projection matrices **P1** and **P2** observing a 3D point **X**. The projections **x1** and **x2** are related through the **epipolar constraint**:

```
x2^T F x1 = 0
```

where **F** is the **fundamental matrix**, a 3x3 rank-2 matrix with 7 degrees of freedom. Geometrically, this constraint states that corresponding points, the 3D scene point, and both camera centers are coplanar (the **epipolar plane**).

### 2.3 Epipolar Lines and Epipoles

The fundamental matrix maps a point in one image to its corresponding **epipolar line** in the other image:

- **l2 = F x1** — the epipolar line in image 2 corresponding to point **x1**
- **l1 = F^T x2** — the epipolar line in image 1 corresponding to point **x2**

All epipolar lines in an image converge at the **epipole**, which is the projection of the other camera's center. The epipoles are characterized by:

```
F e1 = 0    (e1 is the right null space of F)
F^T e2 = 0  (e2 is the left null space of F)
```

---

## 3. Fundamental Matrix Estimation

### 3.1 The 8-Point Algorithm

The epipolar constraint for N corresponding point pairs yields a homogeneous linear system:

```
A f = 0
```

where **f** is the 9-vector obtained by stacking the entries of **F**, and each row of the Nx9 matrix **A** is constructed from one correspondence (u1, v1, u2, v2):

```
[u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1]
```

With N >= 8 correspondences, the solution is the right singular vector of **A** corresponding to its smallest singular value.

### 3.2 Hartley Normalization

Direct application of the 8-point algorithm to raw pixel coordinates leads to severe ill-conditioning of the design matrix **A**. The condition number can reach values exceeding 10^6, making the SVD solution extremely sensitive to noise.

Hartley normalization transforms the points in each image independently so that:
- The centroid is translated to the origin
- The mean distance from the origin is sqrt(2)

The normalization transformation is:

```
T = [ s  0  -s*mu_x ]
    [ 0  s  -s*mu_y ]
    [ 0  0     1    ]
```

where `s = sqrt(2) / rms_distance` and (mu_x, mu_y) is the centroid. After estimation on normalized coordinates, the fundamental matrix is denormalized:

```
F = T2^T F_normalized T1
```

### 3.3 Rank-2 Enforcement

The fundamental matrix must satisfy det(**F**) = 0 (rank 2) to ensure all epipolar lines in an image converge at a single epipole. The SVD-based estimate from the 8-point algorithm generally has rank 3 due to noise.

Rank-2 enforcement is achieved by decomposing the estimate via SVD:

```
F = U diag(s1, s2, s3) V^T
```

and setting the smallest singular value to zero:

```
F_rank2 = U diag(s1, s2, 0) V^T
```

This is the closest rank-2 matrix to the original estimate in the Frobenius norm sense.

---

## 4. Robust Estimation (RANSAC)

In real-world scenarios with outlier correspondences, the standard 8-point algorithm is insufficient. The **RANSAC** (Random Sample Consensus) framework addresses this by:

1. Randomly selecting a minimal sample of 8 correspondences
2. Estimating **F** from the sample
3. Computing the Sampson distance for all correspondences
4. Counting inliers (correspondences with distance below a threshold)
5. Repeating for K iterations and keeping the estimate with the most inliers
6. Refining **F** using all inliers

The **Sampson distance** provides an efficient first-order approximation to the geometric error:

```
d_sampson = (x2^T F x1)^2 / (f1^2 + f2^2 + f1'^2 + f2'^2)
```

where f_i and f_i' are the first two components of **Fx1** and **F^T x2** respectively.

The current implementation uses synthetic correspondences without outliers, so RANSAC is not required. However, the Sampson distance metric is implemented and evaluated as a quality measure.

---

## 5. Essential Matrix and Camera Pose Recovery

### 5.1 Essential Matrix

When the camera intrinsics **K1** and **K2** are known, the **essential matrix** relates corresponding points in normalized image coordinates:

```
E = K2^T F K1
```

The essential matrix has exactly 5 degrees of freedom (3 for rotation, 2 for translation direction) and its two non-zero singular values are equal.

### 5.2 SVD Decomposition

The essential matrix admits four possible (R, t) decompositions through SVD:

```
E = U diag(s, s, 0) V^T
```

Using the orthogonal matrix:

```
W = [ 0  -1  0 ]
    [ 1   0  0 ]
    [ 0   0  1 ]
```

the four candidate solutions are:

```
R1 = U W V^T,     t = +u3
R1 = U W V^T,     t = -u3
R2 = U W^T V^T,   t = +u3
R2 = U W^T V^T,   t = -u3
```

where **u3** is the third column of **U**.

### 5.3 Cheirality Constraint

Only one of the four solutions places all triangulated points in front of both cameras (positive depth). The correct solution is selected by triangulating a set of correspondences with each candidate pose and verifying that the reconstructed points have positive z-coordinates in both camera frames.

---

## 6. Triangulation and 3D Reconstruction

### 6.1 Direct Linear Transform (DLT)

Given projection matrices **P1**, **P2** and corresponding points **x1**, **x2**, the 3D point **X** is recovered by solving the overdetermined system:

```
[ x1 * p3^T - p1^T ]
[ y1 * p3^T - p2^T ]     X = 0
[ x2 * p3'^T - p1'^T ]
[ y2 * p3'^T - p2'^T ]
```

where **pi^T** denotes the i-th row of the projection matrix. The solution is the right singular vector corresponding to the smallest singular value of this 4x4 matrix.

### 6.2 Reprojection Error

The quality of triangulation is evaluated by projecting the recovered 3D points back onto the image planes and measuring the Euclidean distance to the observed 2D points:

```
e_reproj = ||x_observed - P * X_triangulated||_2
```

Low reprojection error (sub-pixel) indicates accurate 3D reconstruction and a well-estimated projection model.

---

## 7. Experimental Evaluation

### 7.1 Experimental Setup

The evaluation uses a synthetic two-camera system with the following configuration:

| Parameter     | Camera 1     | Camera 2     |
|---------------|-------------|-------------|
| Focal length  | (100, 120)  | (90, 110)   |
| Principal pt  | (128, 128)  | (128, 128)  |
| Rotation      | Identity    | Rx(0.1) Ry(π/4) Rz(0.2) |
| Translation   | Origin      | (-1000, 190, 230) |
| Image size    | 256 x 256   | 256 x 256   |

A set of 20 3D scene points distributed across the viewing volume at depths ranging from 2000 to 4000 units provides the test correspondences.

### 7.2 Fundamental Matrix Accuracy

The estimated fundamental matrix is compared against the analytical ground truth using the Frobenius norm. The rank-2 constraint is verified, and the condition number is reported. With Hartley normalization and rank-2 enforcement, the estimated **F** closely matches the analytical solution.

### 7.3 Epipole Consistency

Epipoles are computed using three independent methods:

1. **SVD null space** of **F** and **F^T**
2. **Intersection of epipolar lines** (least-squares via SVD)
3. **Camera center projection** through the opposite projection matrix

Agreement across all three methods validates the correctness of the fundamental matrix estimate.

### 7.4 Triangulation Quality

Triangulated 3D points are evaluated against ground-truth positions and through reprojection error. With clean correspondences, the DLT method achieves sub-pixel reprojection error, confirming the accuracy of the reconstruction pipeline.

---

## 8. Error Analysis

### 8.1 Normalization Sensitivity

The impact of Hartley normalization is evaluated through a Monte Carlo experiment with 100 noise trials (Gaussian noise, sigma = 1.0 pixel). For each trial:

- The 8-point algorithm is run **with** and **without** normalization
- Epipoles are extracted and recorded

The scatter plot of epipole locations reveals:
- **Without normalization**: extreme scatter, with estimates dispersed across hundreds of pixels
- **With normalization**: tight clustering around the ground-truth location

This confirms that normalization is not merely beneficial but essential for reliable estimation.

### 8.2 Sampson Distance

The Sampson distance is evaluated for all correspondences under the estimated fundamental matrix. For noise-free correspondences, the Sampson distance approaches machine precision, validating the algebraic accuracy of the estimate.

### 8.3 Point-to-Epipolar-Line Distance

The algebraic distance from each point to its corresponding epipolar line is computed in both images. This metric directly evaluates how well the epipolar constraint is satisfied and provides a per-correspondence quality assessment.

---

## 9. Discussion

### 9.1 The Necessity of Normalization

The experimental results demonstrate that Hartley normalization transforms the 8-point algorithm from an unreliable method into a robust estimation procedure. The underlying cause is the large dynamic range of pixel coordinates, which leads to a design matrix with vastly different column magnitudes and poor conditioning.

### 9.2 Rank Constraint and Geometric Consistency

Enforcing rank 2 ensures that all epipolar lines in an image are concurrent at the epipole. Without this constraint, the estimated geometry is internally inconsistent — different subsets of correspondences would imply different epipole locations.

### 9.3 Degenerate Configurations

Several geometric configurations lead to degenerate or ambiguous fundamental matrix estimation:

- **Coplanar points**: When all scene points lie on a plane, the fundamental matrix is not uniquely determined by point correspondences alone
- **Pure rotation**: When the camera undergoes only rotation with no translation, the epipolar geometry degenerates (the epipole moves to infinity)
- **Coincident cameras**: Zero baseline produces an undefined fundamental matrix

### 9.4 Practical Considerations

In real-world applications, the pipeline must contend with:
- Outlier correspondences from mismatched features (addressed by RANSAC)
- Lens distortion effects on the epipolar geometry
- Numerical precision limits in SVD computation
- Scene structures that approach degenerate configurations

---

## 10. Conclusion

This project demonstrates a complete multi-view geometry pipeline from point correspondences to 3D reconstruction. The key findings are:

1. The **normalized 8-point algorithm** with rank-2 enforcement provides reliable fundamental matrix estimation when correspondences are clean
2. **Hartley normalization** is essential — estimation without it is catastrophically sensitive to noise
3. **Three independent methods** for epipole computation yield consistent results, validating the estimation pipeline
4. **Linear triangulation** via DLT achieves sub-pixel reprojection error with well-estimated camera geometry
5. The **essential matrix decomposition** with cheirality checking successfully recovers the correct camera pose from four candidates

The implementation provides a modular, well-documented codebase suitable for extension to real-world feature matching with RANSAC, multi-view bundle adjustment, and dense reconstruction.

---

## Result Visualizations

### Point Correspondences
![Point correspondences projected onto both image planes](result_images/matched_features.png)

### Epipolar Lines
![Epipolar lines and epipole locations in both views](result_images/epipolar_lines.png)

### Clean vs. Noisy Correspondences
![Side-by-side comparison of clean and noise-perturbed correspondences](result_images/inlier_matches.png)

### 3D Point Cloud
![Triangulated 3D scene points](result_images/3d_points_plot.png)

### Reprojection Error
![Per-point reprojection error and distribution analysis](result_images/reprojection_error_plot.png)

### Normalization Comparison
![Epipole estimate scatter: with vs. without Hartley normalization](result_images/normalization_comparison.png)
