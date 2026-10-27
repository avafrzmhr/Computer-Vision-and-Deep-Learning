import cv2
import numpy as np

image_path = "./2/charchdande2.png"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (9, 9), 2)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

edges_closed = cv2.Canny(closed, 50, 150)

circles_closed = cv2.HoughCircles(edges_closed, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                  param1=50, param2=30, minRadius=5, maxRadius=30)

circle_data_closed = []
if circles_closed is not None:
    circles_closed = np.uint16(np.around(circles_closed))
    for i in circles_closed[0, :]:
        circle_data_closed.append((i[0], i[1], i[2])) 

        cv2.circle(image, (i[0], i[1]), i[2], (255, 255, 0), 2) 
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)     

output_path = "./2/detected_circles_colored.jpg"
cv2.imwrite(output_path, image)

print("Detected circles (x, y, radius):", circle_data_closed)
print("Processed image saved at:", output_path)
