import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

def get_reliable_colmap_points(points3d_path: Path, min_observations: int = 3) -> np.ndarray:
    """
    Parses COLMAP's points3D.txt to find points corroborated by multiple cameras.
    
    COLMAP points3D format:
    POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    """
    reliable_xyz = []
    
    with open(points3d_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
                
            parts = line.strip().split()
            if len(parts) < 8:
                continue
                
            # Track elements start at index 8 and come in pairs (IMAGE_ID, POINT2D_IDX)
            track_length = (len(parts) - 8) // 2
            
            if track_length >= min_observations:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                reliable_xyz.append([x, y, z])
                
    return np.array(reliable_xyz)


def filter_gaussians_by_reliability(
    gaussian_xyz: np.ndarray, 
    reliable_colmap_xyz: np.ndarray, 
    distance_threshold: float
) -> np.ndarray:
    """
    Compares the generated Gaussian point cloud against the reliable COLMAP 
    points using a KD-Tree. Returns a boolean mask of defensible points.
    """
    if len(reliable_colmap_xyz) == 0:
        print("[WARNING] No reliable COLMAP points found. Masking everything.")
        return np.zeros(len(gaussian_xyz), dtype=bool)
        
    tree = cKDTree(reliable_colmap_xyz)
    
    # Query the distance to the nearest reliable COLMAP point for every Gaussian
    distances, _ = tree.query(gaussian_xyz, k=1)
    
    # Return mask: True if Gaussian is within the threshold radius of a reliable point
    return distances <= distance_threshold


from scipy.spatial.transform import Rotation

def get_camera_centers(images_txt_path: Path) -> np.ndarray:
    """
    Parses COLMAP images.txt and returns world-space camera centers,
    computed as center = -R^T t (COLMAP stores world-to-camera R,t).
    """
    centers = []
    with open(images_txt_path, "r") as f:
        lines = [line for line in f if not line.startswith("#")]
    for i in range(0, len(lines), 2):  # image lines alternate w/ POINTS2D lines
        parts = lines[i].strip().split()
        if len(parts) < 9:
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        t = np.array([tx, ty, tz])
        centers.append(-R.T @ t)
    return np.array(centers)
