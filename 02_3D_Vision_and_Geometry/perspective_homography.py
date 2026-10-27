import cv2
import numpy as np


def load_image(file_path):
    return cv2.imread(file_path)


def transform_perspective(image, corners_src, corners_dst):
    height, width = image.shape[:2]
    matrix = cv2.getPerspectiveTransform(corners_src, corners_dst)
    result = cv2.warpPerspective(image, matrix, (width, height))
    return result


def save_and_display_image(image, file_name="front_view.jpg"):
    cv2.imwrite(file_name, image)
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    image_path = "./button_view.jpeg"
    output_path = "front_view.jpg"

    image = load_image(image_path)

    corners_src = np.float32([[193, 0], [191, 150], [80, 181], [0, 59]])

    corners_dst = np.float32(
        [
            [0, 0],
            [image.shape[1], 0],
            [image.shape[1], image.shape[0]],
            [0, image.shape[0]],
        ]
    )

    front_view_img = transform_perspective(image, corners_src, corners_dst)

    save_and_display_image(front_view_img, output_path)


if __name__ == "__main__":
    main()
