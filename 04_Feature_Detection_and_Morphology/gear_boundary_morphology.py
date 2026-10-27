import cv2
import numpy as np
import os


def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


def preprocess_image(img):
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    return opening


def find_contours(binary_img):
    contours, _ = cv2.findContours(
        binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def create_mask(img, contours):
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, -1, (255), thickness=2)
    return mask


def save_result(img, result_folder, result_filename):
    result_path = os.path.join(result_folder, result_filename)

    os.makedirs(result_folder, exist_ok=True)

    cv2.imwrite(result_path, img)
    print(f"Result saved at: {result_path}")


def main():
    image_path = "./1/charckdande.png"
    result_folder = "results"
    result_filename = "gear_teeth_boundary_extraction.png"

    img = load_image(image_path)

    binary_img = preprocess_image(img)

    contours = find_contours(binary_img)

    mask = create_mask(img, contours)

    save_result(mask, result_folder, result_filename)


if __name__ == "__main__":
    main()