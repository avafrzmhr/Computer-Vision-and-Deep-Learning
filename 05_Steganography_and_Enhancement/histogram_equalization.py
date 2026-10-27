import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def load_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {file_path}")
    return image.astype(np.uint8)



def save_image(image, folder, name):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, name)
    cv2.imwrite(file_path, image)
    print(f"Image saved: {file_path}")



def histogram_equalization(input_image):
    equalized_image = cv2.equalizeHist(input_image)
    return equalized_image


def smoothed_edited_image(input_image, kernel_size=(5, 5), sigma=1.0):
    smoothed_image = cv2.GaussianBlur(input_image, kernel_size, sigma)
    return smoothed_image

def calculate_r(input_image, smoothed_image, epsilon=1e-6):
    """
    r(x, y) = f(x, y) / (g(x, y) + epsilon)
    """
    r_image = input_image / (smoothed_image + epsilon)  
    r_image = cv2.normalize(r_image, None, 0, 255, cv2.NORM_MINMAX) 
    return r_image.astype(np.uint8)



def main():
    input_image_path = "./Girl.jpg"
    img = load_image(input_image_path)

    equ = histogram_equalization(img)

    save_folder = "./output/histogram_equalization"
    save_image(equ, save_folder, "equalized_image.jpg")

    smoothed_result = smoothed_edited_image(img)

    save_folder = "./output/smoothed_edited_image"
    save_image(smoothed_result, save_folder, "smoothed_edited_image.jpg")
    
    r_image = calculate_r(img, smoothed_result, epsilon=1e-6)

    save_folder = "./output/r_image"
    save_image(r_image, save_folder, "r_image.jpg")

 

if __name__ == "__main__":
    main()
