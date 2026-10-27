import cv2
import os


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image at {image_path}")
    return image


def save_image(output_path, image):
    cv2.imwrite(output_path, image)


def apply_gaussian_high_pass_filter(image, sigma=2.0):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (0, 0), sigma)
    high_pass = gray_image - blurred
    result = cv2.addWeighted(gray_image, 1.5, high_pass, -0.4, 0)
    return result


def enhance_image(input_image_path, output_folder, sigma=2.0):
    input_image = load_image(input_image_path)

    enhanced_image = apply_gaussian_high_pass_filter(input_image, sigma=sigma)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, "enhanced_image.jpg")
    save_image(output_path, enhanced_image)


if __name__ == "__main__":
    input_image_path ="./2/xray.png"
    output_folder = "results"
    sigma_value = 1.0

    try:
        enhance_image(input_image_path, output_folder, sigma=sigma_value)
        print("Image enhanced and saved successfully.")
    except FileNotFoundError as e:
        print(e)
