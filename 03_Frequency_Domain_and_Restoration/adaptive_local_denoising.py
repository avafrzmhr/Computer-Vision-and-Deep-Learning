import cv2
import numpy as np
import os


def reduce_periodic_noise(
    input_image_path, output_folder="results", noise_frequency_threshold=20
):
    original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    f_transform = np.fft.fft2(original_image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    rows, cols = original_image.shape
    center_row, center_col = rows // 2, cols // 2
    f_transform_shifted[
        center_row - noise_frequency_threshold : center_row + noise_frequency_threshold,
        center_col - noise_frequency_threshold : center_col + noise_frequency_threshold,
    ] = 0

    filtered_f_transform = np.fft.ifftshift(f_transform_shifted)
    denoised_image = np.fft.ifft2(filtered_f_transform).real

    denoised_image = cv2.normalize(
        denoised_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, "denoised_image.jpg")
    cv2.imwrite(output_path, denoised_image)

    print(f"Denoised image saved at: {output_path}")

    return denoised_image


if __name__ == "__main__":
    input_image_path = "./3/bird.jpeg"

    output_folder = "results"

    denoised_image = reduce_periodic_noise(
        input_image_path, output_folder, noise_frequency_threshold=10
    )
