import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from imutils import contours
import imutils

MIN_AREA = 1000


def get_cell_blocks(file_path):
    # Load image, enlarge, convert to grayscale, Otsu's threshold
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph close to combine adjacent contours into a single contour
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (85, 5))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)

    # Find contours, sort from top-to-bottom
    # Iterate through contours, extract row ROI, OCR, and parse data
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

    return cnts


def get_detailed_cell_info(file_path):
    table_image_contour = cv2.imread(file_path, 0)

    ret, thresh_value = cv2.threshold(table_image_contour, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    dilated_value = cv2.dilate(thresh_value, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(
        dilated_value, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


if __name__ == '__main__':
    file_path = 'pdf_image_data/1.jpg'
    cnts = get_detailed_cell_info(file_path)

    image = cv2.imread(file_path)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > MIN_AREA:
            table_image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)

    plt.imshow(table_image)
    plt.show()
    cv2.namedWindow('detecttable', cv2.WINDOW_NORMAL)
