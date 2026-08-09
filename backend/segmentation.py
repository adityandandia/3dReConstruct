import shutil
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def _estimate_eps(xyz: np.ndarray, k: int = 25) -> float:
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=k + 1)
    kth_dists = dists[:, -1]
    return float(np.percentile(kth_dists, 95))


def _remove_ground_plane(xyz: np.ndarray, iterations: int = 300,
                          distance_threshold_frac: float = 0.01,
                          verticality_thresh: float = 0.85) -> np.ndarray:
    """
    Single dominant-plane RANSAC. Only removes it if the fitted normal
    is near-vertical (i.e. plane is near-horizontal -> a floor),
    to avoid eating a wall or the object itself.
    Returns a boolean mask: True = kept (not ground).
    """
    n = len(xyz)
    if n < 50:
        return np.ones(n, dtype=bool)

    extent = np.percentile(np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1), 90)
    thresh = distance_threshold_frac * extent

    best_inliers = np.array([], dtype=int)
    for _ in range(iterations):
        idx = np.random.choice(n, 3, replace=False)
        p0, p1, p2 = xyz[idx[0]], xyz[idx[1]], xyz[idx[2]]
        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-8:
            continue
        normal = normal / norm_len
        dists = np.abs((xyz - p0) @ normal)
        inliers = np.where(dists < thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_normal = normal

    if len(best_inliers) < 0.05 * n:
        return np.ones(n, dtype=bool)  # no dominant plane found

    # near-vertical normal = check alignment against each coordinate axis
    max_axis_alignment = max(abs(best_normal[0]), abs(best_normal[1]), abs(best_normal[2]))
    if max_axis_alignment < verticality_thresh:
        return np.ones(n, dtype=bool)  # plane exists but isn't floor-like — keep everything

    mask = np.ones(n, dtype=bool)
    mask[best_inliers] = False
    return mask


def segment_object(raw_ply_path: Path, camera_centers: np.ndarray, output_path: Path,
                    min_cluster_frac: float = 0.01, keep_top_n: int = 1):
    """
    Isolates the evidence object's Gaussians via ground-plane removal +
    DBSCAN, selecting cluster(s) by centroid proximity to the median
    camera position. Falls back to the unsegmented cloud on any failure
    or empty result, rather than blocking the pipeline.
    """
    from backend.tasks import _read_ply, _write_ply  # local import: avoids circular import

    print(f"\n[SEGMENTATION]: DBSCAN + ground-plane isolation on {raw_ply_path}")

    try:
        header_lines, data, properties, num_vertices = _read_ply(raw_ply_path)
        x_i, y_i, z_i = properties.index("x"), properties.index("y"), properties.index("z")
        xyz = data[:, [x_i, y_i, z_i]]

        ground_mask = _remove_ground_plane(xyz)
        data_ng = data[ground_mask]
        xyz_ng = xyz[ground_mask]
        print(f"  -> Ground-plane removal: kept {len(data_ng)} / {num_vertices}")

        if len(xyz_ng) < 50:
            raise RuntimeError("Too few points remain after ground-plane removal")

        eps = _estimate_eps(xyz_ng, k=25)
        db = DBSCAN(eps=eps, min_samples=25, algorithm="ball_tree", n_jobs=-1).fit(xyz_ng)
        labels = db.labels_

        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(unique_labels) == 0:
            raise RuntimeError("DBSCAN found no clusters")

        total = len(xyz_ng)
        size_ok = counts >= (min_cluster_frac * total)
        candidate_labels = unique_labels[size_ok]
        if len(candidate_labels) == 0:
            candidate_labels = unique_labels  # fall back: consider all clusters

        median_cam = np.median(camera_centers, axis=0) if len(camera_centers) else np.median(xyz_ng, axis=0)

        scored = []
        for lbl in candidate_labels:
            centroid = xyz_ng[labels == lbl].mean(axis=0)
            dist = np.linalg.norm(centroid - median_cam)
            scored.append((dist, lbl))
        scored.sort(key=lambda x: x[0])
        chosen_labels = {lbl for _, lbl in scored[:keep_top_n]}

        final_mask = np.array([lbl in chosen_labels for lbl in labels])
        segmented_data = data_ng[final_mask]

        if len(segmented_data) == 0:
            raise RuntimeError("Segmentation produced 0 points")

        _write_ply(output_path, header_lines, segmented_data, len(properties))

        # write-then-reread verification
        _, verify_data, _, verify_n = _read_ply(output_path)
        if verify_n != len(segmented_data):
            raise RuntimeError(f"Verification mismatch: wrote {len(segmented_data)}, read back {verify_n}")

        print(f"  -> Segmentation complete: {len(segmented_data)} / {num_vertices} Gaussians kept.")

    except Exception as e:
        print(f"[WARNING] Segmentation failed ({e}); falling back to unsegmented cloud.")
        shutil.copy(raw_ply_path, output_path)
