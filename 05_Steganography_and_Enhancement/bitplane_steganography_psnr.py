import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from math import log10


def hide_image_in_plane(main_image, secret_image, plane_bit):
    plane_mask = 1 << plane_bit
    main_image = main_image.astype(np.uint8)
    secret_image = secret_image.astype(np.uint8)

    if (
        secret_image.shape[0] > main_image.shape[0]
        or secret_image.shape[1] > main_image.shape[1]
    ):
        print("Error: Secret image is larger than the main image.")
        return None

    for i in range(secret_image.shape[0]):
        for j in range(secret_image.shape[1]):
            main_image[i, j] &= (~plane_mask & 0xFF)  # Apply mask limit
            main_image[i, j] |= ((secret_image[i, j] >> plane_bit) & 1) * plane_mask

    return main_image


def main():
    main_image_path = "1/main_image.jpg"
    secret_image_path = "1/secret_image.jpg"

    main_image = cv2.imread(main_image_path, cv2.IMREAD_GRAYSCALE)
    secret_image = cv2.imread(secret_image_path, cv2.IMREAD_GRAYSCALE)

    if main_image is None:
        print("Error: Unable to load the main image.")
        return
    if secret_image is None:
        print("Error: Unable to load the secret image.")
        return

    output_dir = "stego_images"
    os.makedirs(output_dir, exist_ok=True)

    stego_images = []
    for plane_bit in range(8):
        stego_image = hide_image_in_plane(main_image.copy(), secret_image, plane_bit)
        stego_images.append(stego_image)

    for plane_bit, stego_image in enumerate(stego_images):
        stego_image_name = os.path.join(
            output_dir, f"stego_image_plane_{plane_bit}.jpg"
        )
        cv2.imwrite(stego_image_name, stego_image)

        # Calculate PSNR
        mse = np.mean((main_image - stego_image) ** 2)
        if mse == 0:
            psnr = 100
        else:
            psnr = 20 * log10(255 / np.sqrt(mse))

        # Calculate SSIM
        ssim_value = ssim(main_image, stego_image)

        print(f"Bit Plane {plane_bit}: PSNR = {psnr:.2f}, SSIM = {ssim_value:.2f}")

    print(
        "Steganography completed for all planes. Stego images saved in the 'stego_images' folder."
    )


if __name__ == "__main__":
    main()
