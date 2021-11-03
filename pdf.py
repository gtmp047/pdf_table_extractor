import cv2
import numpy as np
import pytesseract
from imutils import contours
import imutils
import matplotlib.pyplot as plt





# Load image, enlarge, convert to grayscale, Otsu's threshold
file_path = 'pdf_image_data/1.jpg'
image = cv2.imread(file_path)
image_contour = cv2.imread(file_path, 0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


# Morph close to combine adjacent contours into a single contour
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (85, 5))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

# Find contours, sort from top-to-bottom
# Iterate through contours, extract row ROI, OCR, and parse data
cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
(cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")


table_image = cv2.imread(file_path)
tab_kernel = np.ones((5, 5), np.uint8)
tab_data_list = []
for cnt in cnts:
    x, y, w, h = cv2.boundingRect(cnt)
    table_image = image[y:y+h, x:x+w]
    table_image_contour = image_contour[y:y+h, x:x+w]


    # для каждого прямоугольника ищем табличные данные
    ret, thresh_value = cv2.threshold(table_image_contour, 180, 255, cv2.THRESH_BINARY_INV)
    dilated_value = cv2.dilate(thresh_value, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilated_value, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for tab_cnt in contours:
        x, y, w, h = cv2.boundingRect(tab_cnt)
        # bounding the images
        # if y < 150:
        table_image = cv2.rectangle(table_image, (x, y), (x + w, y + h), (0, 0, 255), 1)

    plt.imshow(table_image)
    plt.show()
    cv2.namedWindow('detecttable', cv2.WINDOW_NORMAL)
    break

