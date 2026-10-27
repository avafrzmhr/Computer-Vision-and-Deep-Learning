import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def resize_image(image_path, target_size):

    img = Image.open(image_path).convert("L")
    img_resized = img.resize(target_size, Image.Resampling.LANCZOS) 
    return np.array(img_resized)



def fourier_transform(image_array, save_folder, image_name):

    fourier_transform = np.fft.fft2(image_array)
    fft_shifted = np.fft.fftshift(fourier_transform)

    magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)
    phase_spectrum = np.angle(fft_shifted)

    magnitude_folder = os.path.join(save_folder, "magnitude")
    phase_folder = os.path.join(save_folder, "phase")
    os.makedirs(magnitude_folder, exist_ok=True)
    os.makedirs(phase_folder, exist_ok=True)

    magnitude_path = os.path.join(magnitude_folder, image_name + "_magnitude.png")
    phase_path = os.path.join(phase_folder, image_name + "_phase.png")

    plt.imsave(magnitude_path, magnitude_spectrum, cmap="gray")
    plt.imsave(phase_path, phase_spectrum, cmap="gray")

    return magnitude_spectrum, phase_spectrum


def save_reconstructed_image(reconstructed_image, save_folder, image_name):

    reconstructed_folder = os.path.join(save_folder, "reconstructed")
    os.makedirs(reconstructed_folder, exist_ok=True)

    reconstructed_path = os.path.join(
        reconstructed_folder, image_name + "_reconstructed.png"
    )

    plt.imsave(reconstructed_path, reconstructed_image, cmap="gray")



def reconstruct_image(magnitude_spectrum, phase_spectrum):

    new_fourier_transform = np.exp(magnitude_spectrum) * np.exp(1j * phase_spectrum)
    reconstructed_image = np.abs(np.fft.ifft2(np.fft.ifftshift(new_fourier_transform)))

    return reconstructed_image


if __name__ == "__main__":
    path_to_image_a = "./1/input.jpeg"
    path_to_image_b = "./1/input2.jpeg"

    save_folder_a = "./output_a"
    save_folder_b = "./output_b"

    try:
        img_a = Image.open(path_to_image_a)
        img_b = Image.open(path_to_image_b)

        target_size = (
            min(img_a.size[1], img_b.size[1]),  
            min(img_a.size[0], img_b.size[0])   
        )

        image_a_resized = resize_image(path_to_image_a, target_size)
        image_b_resized = resize_image(path_to_image_b, target_size)

        magnitude_spectrum_a, phase_spectrum_a = fourier_transform(
            image_a_resized, save_folder_a, "image_a"
        )

        magnitude_spectrum_b, phase_spectrum_b = fourier_transform(
            image_b_resized, save_folder_b, "image_b"
        )

        reconstructed_image_a = reconstruct_image(magnitude_spectrum_a, phase_spectrum_b)
        reconstructed_image_b = reconstruct_image(magnitude_spectrum_b, phase_spectrum_a)

        save_reconstructed_image(reconstructed_image_a, save_folder_a, "image_a")
        save_reconstructed_image(reconstructed_image_b, save_folder_b, "image_b")

        print("Images processed and saved successfully!")
    except Exception as e:
        print(f"Error occurred: {e}")
