import open3d as o3d
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

print(f"[CLEANUP] Loading {input_path}")
pcd = o3d.io.read_point_cloud(input_path)
print(f"[CLEANUP] Points before: {len(pcd.points)}")

# Step 1: Statistical outlier removal
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f"[CLEANUP] After SOR: {len(pcd.points)}")

# Step 2: Radius outlier removal
pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=0.05)
print(f"[CLEANUP] After radius filter: {len(pcd.points)}")

# Step 3: Voxel downsample (optional — uncomment if needed)
# pcd = pcd.voxel_down_sample(voxel_size=0.01)
# print(f"[CLEANUP] After voxel downsample: {len(pcd.points)}")

o3d.io.write_point_cloud(output_path, pcd)
print(f"[CLEANUP] Saved to {output_path}")
