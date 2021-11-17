import shutil
from os import listdir, walk, unlink
from os.path import isfile, join
import random
import datetime
import json

import cv2
import numpy as np
import pytesseract
from imutils import contours
from pdf2image import convert_from_path

from classes import Cell, Table

# Adding custom options
TESSERACT_CONF = r'--oem 3 --psm 6'

options = {
    'FALSE': 'table_content/not_ok.png',
    'TRUE': 'table_content/ok.png',
}

SIMPLISITY_TRESHOLD = 0.9
MIN_AREA = 1500


def _get_cur_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _compare_images(target_image, template):
    w, h = template.shape[:-1]

    try:
        res = cv2.matchTemplate(target_image, template, cv2.TM_CCOEFF_NORMED)
    except:
        return
    threshold = .7
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        start_point = pt
        end_point = (pt[0] + h, pt[1] + w)
        middle_point_rand = (int((start_point[0] + end_point[0]) / 2 + random.randint(-10, 10)),
                             int((start_point[1] + end_point[1]) / 2 + random.randint(-10, 10)))

        cv2.rectangle(target_image,
                      (middle_point_rand[0] - 5, middle_point_rand[1] - 5),
                      (middle_point_rand[0] + 5, middle_point_rand[1] + 5), (0, 0, 255), 2)

        cv2.rectangle(target_image,
                      start_point,
                      end_point, (0, 255, 0), 2)

        cv2.imwrite(f'good/{_get_cur_time()}_.png', target_image)
        return middle_point_rand
    return

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


def get_cell_blocks(image):
    # Load image, enlarge, convert to grayscale, Otsu's threshold
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
    grey_table_image_contour = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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


def delete_dir_content(path='temp'):
    for root, dirs, files in walk(path):
        for f in files:
            unlink(join(root, f))
        for d in dirs:
            shutil.rmtree(join(root, d))


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

def _save_data(name, text_arr, table_arr):
    data_json = {
        'text': text_arr,
        'tables': []
    }

    for table in table_arr:
        table_rows = []
        for row in table.rows:
            temp_row =[str(i) for i in row]
            table_rows.append(temp_row)
        data_json['tables'].append(table_rows)

    with open(f'out/{name}.json', 'w') as f:
        json.dump(data_json , f, ensure_ascii=False)


def extract_text(image):
    return pytesseract.image_to_string(image, lang='rus',
                                       config=TESSERACT_CONF).rstrip().replace('\n', ' ')


if __name__ == '__main__':
    extracted_fact = dict()
    total_text_arr = []
    total_table_arr = []
    pdf_name = '110391501'
    numbers_to_extruct = ['огрн', 'инн']
    pdf = convert_from_path(f'pdf/{pdf_name}.pdf')

    for i in range(len(pdf)):
        # Save pages as images in the pdf
        pdf[i].save(f'temp/{pdf_name}_page_' + str(i) + '.jpg', 'JPEG')

    files = [join('temp', f) for f in listdir('temp') if
             isfile(join('temp', f))]
    files.sort()

    for image_path in files:
        main_image = image = cv2.imread(image_path)

        detailed_data = get_detailed_cell_info(main_image)
        detailed_data.reverse()
        region_blocks = get_cell_blocks(main_image)
        _, dots = detect_table(main_image)

        for region_block in region_blocks:
            x, y, w, h = cv2.boundingRect(region_block)
            y -= 5
            h += 5
            text = extract_text(main_image[y:y + h, x:x + w])

            extructed_num = [i for i in numbers_to_extruct if i in text.lower()]

            if not check_region_is_table(region_block, dots):
                # определяем просто как текст
                total_text_arr.append(text)
            else:
                # запрещенные слова игнорим
                text = extract_text(main_image[y:y + h, x:x + w])
                if 'страница:' in text.lower() and text.lower().startswith('страница:'):
                    continue

                table_struct = Table()

                # выделяем блоки в таблице
                detailed_data_cell_in_region = []
                for detailed_data_cell in detailed_data:
                    detailed_data_cell_rect = cv2.boundingRect(detailed_data_cell)
                    region_block_rect = cv2.boundingRect(region_block)

                    simplisity = Cell.interception_perc(Cell(*region_block_rect), Cell(*detailed_data_cell_rect))
                    if simplisity >= SIMPLISITY_TRESHOLD and region_block_rect != detailed_data_cell_rect:
                        x, y, w, h = cv2.boundingRect(detailed_data_cell)

                        # иногда

                        text = extract_text(main_image[y:y + h, x:x + w])
                        if extructed_num and not text.isdigit():
                            continue

                        # Проверка на присутствие галочек
                        cur_options = []
                        for option_name, path in options.items():
                            # if ('коллегиальный' in text or 'единоличный' in text) and option_name
                            rule = _compare_images(main_image[y:y + h, x:x + w], cv2.imread(path))
                            if rule:
                                cur_options.append(option_name)
                        if cur_options:
                            text = ';'.join(cur_options)

                        detailed_data_cell_in_region.append(detailed_data_cell)
                        table_struct.add_value(Cell(x, y, w, h, text))

                total_table_arr.append(table_struct)

            if extructed_num:
                text = ''.join([i.text for i in table_struct.rows[0]])
                if extructed_num[0] != 'инн':
                    extracted_fact[extructed_num[0]] = text
                else:
                    extracted_fact['инн'] = text[:10]
                    extracted_fact['кпп'] = text[10:]
                numbers_to_extruct.remove(extructed_num[0])
                del total_table_arr[-1]

    delete_dir_content()
    _save_data(pdf_name,total_text_arr, total_table_arr)

