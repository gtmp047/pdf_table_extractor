import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from imutils import contours

from classes import Cell, Table

# Adding custom options
TESSERACT_CONF = r'--oem 3 --psm 6'

TRESHOLD = 0.9
MIN_AREA = 1500



def detect_table(src_img):
    if len(src_img.shape) == 2:
        gray_img = src_img
    elif len(src_img.shape) == 3:
        gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    thresh_img = cv2.adaptiveThreshold(~gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    h_img = thresh_img.copy()
    v_img = thresh_img.copy()
    scale = 70
    h_size = int(h_img.shape[1] / scale)

    h_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
    h_erode_img = cv2.erode(h_img, h_structure, 1)

    h_dilate_img = cv2.dilate(h_erode_img, h_structure, 1)
    v_size = int(v_img.shape[0] / scale)

    v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
    v_erode_img = cv2.erode(v_img, v_structure, 1)
    v_dilate_img = cv2.dilate(v_erode_img, v_structure, 1)

    mask_img = h_dilate_img + v_dilate_img
    joints_img = cv2.bitwise_and(h_dilate_img, v_dilate_img)

    return mask_img, joints_img


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


def get_detailed_cell_info(image):
    grey_table_image_contour =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, thresh_value = cv2.threshold(grey_table_image_contour, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    dilated_value = cv2.dilate(thresh_value, kernel, iterations=1)
    cnts, hierarchy = cv2.findContours(dilated_value, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_area_cnts = []
    for c in cnts:
        _, _, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) > MIN_AREA and h >= 25 and w >= 25:
            min_area_cnts.append(c)

    return min_area_cnts


def check_region_is_table(region, dots):
    x, y, w, h = cv2.boundingRect(region)
    dots_region = dots[y:y + h, x:x + w]
    if not cv2.countNonZero(dots_region):
        return False
    return True






def get_image_with_vert_and_hor(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    blank_image = ~np.zeros(image.shape, np.uint8)


    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=4)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(blank_image, [c], -1, 0, -1)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=4)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(blank_image, [c], -1, 0, -1)

    return blank_image


def extract_text_from_region(region):
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=4)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, 0, -1)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=4)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, 0, -1)

    return pytesseract.image_to_string(thresh, lang='rus', config=TESSERACT_CONF)


if __name__ == '__main__':
    file_path = 'pdf_image_data/2.jpg'
    main_image = cv2.imread(file_path)
    # ver_and_hor_image = get_image_with_vert_and_hor(main_image)

    detailed_data = get_detailed_cell_info(main_image)
    detailed_data.reverse()
    region_blocks = get_cell_blocks(file_path)
    _, dots = detect_table(main_image)

    # image = cv2.imread(file_path)
    # for detailed_data_cell in detailed_data:
    #     x, y, w, h = cv2.boundingRect(detailed_data_cell)
    #     table_image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
    #
    # plt.imshow(table_image)
    # plt.show()
    # cv2.namedWindow('detecttable', cv2.WINDOW_NORMAL)



    for region_block in region_blocks:
        x, y, w, h = cv2.boundingRect(region_block)
        y -= 5
        h += 5
        text = pytesseract.image_to_string(main_image[y:y + h, x:x + w], lang='rus', config=TESSERACT_CONF)

        if not check_region_is_table(region_block, dots):
            # определяем просто как текст
            continue
        else:
            # запрещенные слова игнорим
            text = pytesseract.image_to_string(main_image[y:y + h, x:x + w], lang='rus', config=TESSERACT_CONF)
            if 'страница:' in text.lower() and text.lower().startswith('страница:'):
                continue

            table_struct = Table()

            # выделяем блоки в таблице
            detailed_data_cell_in_region = []
            for detailed_data_cell in detailed_data:
                detailed_data_cell_rect = cv2.boundingRect(detailed_data_cell)
                region_block_rect = cv2.boundingRect(region_block)

                simplisity = Cell.interception_perc(Cell(*region_block_rect), Cell(*detailed_data_cell_rect))
                if simplisity >= TRESHOLD and region_block_rect != detailed_data_cell_rect:
                    x, y, w, h = cv2.boundingRect(detailed_data_cell)
                    text = pytesseract.image_to_string(main_image[y:y + h, x:x + w], lang='rus', config=TESSERACT_CONF)
                    detailed_data_cell_in_region.append(detailed_data_cell)
                    table_struct.add_value(Cell(*detailed_data_cell_rect))

                    # image = cv2.imread(file_path)
                    # x, y, w, h = cv2.boundingRect(detailed_data_cell)
                    # table_image = image[y:y + h, x:x + w]
                    # text2 = pytesseract.image_to_string(main_image[y:y + h, x:x + w], lang='rus', config=TESSERACT_CONF)
                    # plt.imshow(table_image)
                    # plt.show()
                    # cv2.namedWindow('detecttable', cv2.WINDOW_NORMAL)

            # x, y, w, h = cv2.boundingRect(region_block)
            # table_image = table_image[y:y + h, x:x + w]
            # plt.imshow(table_image)
            # plt.show()
            # cv2.namedWindow('detecttable', cv2.WINDOW_NORMAL)

            # image = cv2.imread(file_path)
            # table_image = None
            # for c in detailed_data_cell_in_region:
            #     x, y, w, h = cv2.boundingRect(c)
            #     table_image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
            #
            # x, y, w, h = cv2.boundingRect(region_block)
            # table_image = table_image[y:y + h, x:x + w]
            # plt.imshow(table_image)
            # plt.show()
            # cv2.namedWindow('detecttable', cv2.WINDOW_NORMAL)
