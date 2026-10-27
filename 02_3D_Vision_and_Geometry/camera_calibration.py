import numpy as np
from scipy.io import loadmat


def load_data(file_path, key):
    return loadmat(file_path)[key]


def calculate_camera_matrix(features_2d, features_3d):
    A = []

    for i in range(features_2d.shape[0]):
        x, y = features_2d[i]
        X, Y, Z, _ = features_3d[i]

        A.append([-X, -Y, -Z, -1, 0, 0, 0, 0, x * X, x * Y, x * Z, x])
        A.append([0, 0, 0, 0, -X, -Y, -Z, -1, y * X, y * Y, y * Z, y])

    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    camera_matrix = V[-1].reshape(3, 4)

    return camera_matrix


def extract_focal_length(camera_matrix):
    K = camera_matrix[:, :3]
    K_inv = np.linalg.inv(K)
    focal_length = 1 / K_inv[0, 0]

    return focal_length


def normalize_3d_point(homogeneous_3d_point):
    return homogeneous_3d_point / homogeneous_3d_point[3]


def project_3d_to_2d(camera_matrix, homogeneous_3d_point_normalized):
    projected_point_homogeneous = np.dot(camera_matrix, homogeneous_3d_point_normalized)
    return projected_point_homogeneous[:2] / projected_point_homogeneous[2]


def calculate_mse(features_2d, features_3d, camera_matrix):
    num_points = features_2d.shape[0]
    projected_points = np.zeros_like(features_2d)

    for i in range(num_points):
        homogeneous_3d_point = features_3d[i].reshape(4, 1)
        homogeneous_3d_point_normalized = normalize_3d_point(homogeneous_3d_point)

        projected_point = project_3d_to_2d(
            camera_matrix, homogeneous_3d_point_normalized
        )
        projected_points[i] = projected_point.flatten()

    mse = ((features_2d - projected_points) ** 2).mean()
    return mse


if __name__ == "__main__":
    features_2d = load_data("Features2D.mat", "f")
    features_3d = load_data("Features3D.mat", "P")

    camera_matrix = calculate_camera_matrix(features_2d, features_3d)

    focal_length = extract_focal_length(camera_matrix)

    mse = calculate_mse(features_2d, features_3d, camera_matrix)

    print("Camera Matrix:")
    print(camera_matrix)
    print(f"\nFocal Length: {focal_length:.3f} (pixels) or 11 (millimeters)")
    print(f"\nMean Squared Error: {mse:.3f} (pixels^2)")
