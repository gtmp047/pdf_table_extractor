import datetime
import json
import random
import re
import shutil
from os import listdir, walk, unlink
from os.path import isfile, join

import cv2
import fitz
import numpy as np
import pytesseract
from imutils import contours
from pdf2image import convert_from_path

from classes import Cell, Table

# Adding custom options
TESSERACT_CONF = r'--oem 3 --psm 6'

BOOLEAN_STRING_SET = {'НЕТ', 'ДА'}
options = {
    'НЕТ': 'table_content/not_ok.png',
    'ДА': 'table_content/ok.png',
}

SIMPLISITY_TRESHOLD = 0.9
MIN_AREA = 6000
OK_BUTTON_AREA = 89 * 87


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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 20))
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


def int_mean(value):
    return int(np.mean(value))


def check_image_contain_all_white_pixels(region, dest_image, checker):
    if len(dest_image.shape) == 2:
        gray_img = dest_image
    elif len(dest_image.shape) == 3:
        gray_img = cv2.cvtColor(dest_image, cv2.COLOR_BGR2GRAY)

    x, y, w, h = cv2.boundingRect(region)
    gray_img_region = gray_img[y:y + h, x:x + w]
    if not checker(gray_img_region):
        return True
    return False


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


def prepare_and_save_data(name, text_arr, table_arr, extracted_fact, raw_text):
    extracted_fact.update({
        'Наименование отчета': text_arr[1],
        'Получатель': text_arr[0],
        'Наименование организации': text_arr[2],
        'Юридический адрес': text_arr[3],
    })

    # Получение списка деятельности
    activity_list = []
    activity_table = [table for table in table_arr if table[0][0].startswith('1') and table][0]
    for row in activity_table:
        if row[0].text != '1' and row[1].text:
            activity_list.append(row[1].text)
    extracted_fact.update({
        'Основные виды деятельности в отчетном периоде': activity_list
    })

    # получение предпринимательской деятельности
    enterprise_activity_list = []
    enterprise_activity_table = [table for table in table_arr if table[0][0].startswith('2')][0]
    for row in enterprise_activity_table:
        if row[-1].text in BOOLEAN_STRING_SET:
            enterprise_activity_list.append(f'{row[-2].text} : {row[-1].text}')
    extracted_fact.update({
        'Предпринимательская деятельность': enterprise_activity_list
    })

    # получение источников формирования имущества
    sources_of_property_list = []
    sources_of_property_tables = [table for table in table_arr if table[0][0].startswith('3')]
    for table in sources_of_property_tables:
        for row in table:
            if row[-1].text in BOOLEAN_STRING_SET:
                sources_of_property_list.append(f'{row[-2].text} : {row[-1].text}')
        extracted_fact.update({
            'Источники формирования имущества': sources_of_property_list
        })

    # ищем забагованную 4.2 которая лежит в Управление деятельностью:
    table_initiator_index, row_initiator_index = \
        [(table_index, row_index) for table_index, table in enumerate(table_arr)
         for row_index, row in enumerate(table.rows)
         if row[0].text == '4.2'][0]

    if table_initiator_index:
        new_table = Table()
        new_table.rows = table_arr[table_initiator_index][row_initiator_index:]
        del table_arr[table_initiator_index].rows[row_initiator_index:]
        table_arr.append(new_table)


    # получение списка агентов управления деятельностью:
    agent_list = []
    agent_tables = [table for table in table_arr if table[0][0].startswith('4')]
    for table in agent_tables:
        temp_agent = {}

        if table[0][0].text != '4':
            #  попали на нормальную таблицу
            temp_agent[table[0][1].text] = table[0][2].text
            for row in table:
                for i, item in enumerate(row):
                    if ';' in item.text:
                        temp_agent['коллегиальный'] = item.text.split(';')[0]
                        temp_agent['единоличный'] = item.text.split(';')[1]
                    if 'Периодичность проведения заседаний' in item.text:
                        temp_agent['Периодичность проведения заседаний'] = row[i + 1].text
                    if 'Проведено заседаний' in item.text:
                        temp_agent['Проведено заседаний'] = row[i + 1].text

        else:
            # попали на первую таблицу где забили на форматирование. Ищем в тупую
            for row in table:
                for item in row:
                    if 'Высший орган управления' in item.text:
                        temp_str = item.text
                        temp_agent['Вид управляющего органа'] = 'Высший орган управления'
                        temp_str = temp_str.replace('Высший орган управления', '')
                        temp_str = temp_str.replace('(сведения о персональном составе указываются в листе А)', '')
                        temp_agent['Наименование органа'] = temp_str.strip()

                    if 'Периодичность проведения заседаний' in item.text:
                        temp_str = item.text
                        temp_str = temp_str.replace('Периодичность проведения заседаний в соответствии с', '')
                        temp_str = temp_str.replace('учредительными документами', '')
                        temp_agent['Периодичность проведения заседаний'] = temp_str.strip()

                    if 'Проведено заседаний' in item.text:
                        temp_str = item.text
                        temp_str = temp_str.replace('Проведено заседаний', '')
                        temp_agent['Проведено заседаний'] = temp_str.strip()

        agent_list.append(temp_agent)
    extracted_fact.update({
        'Управление деятельностью': agent_list
    })

    com_name_index = \
        [i for i, v in enumerate(text_arr) if v.startswith('Сведения о персональном составе руководящих органов')][0]
    if com_name_index:
        com_name = text_arr[com_name_index + 1].replace('(полное наименование руководящего органа)', '')
        extracted_fact.update({
            'Наименование органа': com_name.strip()
        })

    # подпись
    index_of_sign = [i for i, v in enumerate(text_arr) if
                     v.startswith('Лицо, имеющее право без доверенности действовать')][0]
    if index_of_sign:
        sign_text = text_arr[index_of_sign + 1].replace(
            '(подпись)', '').replace('(дата)', '').replace('(фамилия, имя, отчество, занимаемая должность)', '')
        fio, other = sign_text.split(',')
        sign_date = re.search(r'\d{2}.\d{2}.\d{4}', other).group()
        sign_position = other.replace(sign_date, '')

        extracted_fact.update({
            'ФИО подписывающего лица': fio.strip(),
            'Должность': sign_position.strip(),
            'Дата подписи': sign_date
        })

    # определение списка руководящего состава
    ruler_list = []
    ruler_tables = [table for table in table_arr if table[0][0].startswith('Фамилия')]
    for table in ruler_tables:
        temp_ruller = {}
        for row in table:
            temp_ruller.update({row[-2].text : row[-1].text})
        ruler_list.append(temp_ruller)
    extracted_fact.update({
        'Состав руководящих органов': ruler_list
    })

    with open(f'out/{name}.json', 'w') as f:
        json.dump(extracted_fact, f, ensure_ascii=False)


def _save_data(name, text_arr, table_arr):
    data_json = {
        'text': text_arr,
        'tables': []
    }

    for table in table_arr:
        table_rows = []
        for row in table.rows:
            temp_row = [str(i) for i in row]
            table_rows.append(temp_row)
        data_json['tables'].append(table_rows)

    with open(f'out/{name}.json', 'w') as f:
        json.dump(data_json, f, ensure_ascii=False)


def extract_text(image):
    return pytesseract.image_to_string(image, lang='rus',
                                       config=TESSERACT_CONF).rstrip().replace('\n', ' ')


if __name__ == '__main__':
    extracted_fact = dict()
    total_text_arr = []
    total_table_arr = []
    pdf_name = '110423401'
    forgetten_tabels_name = ['огрн', 'инн', 'страница:']
    pdf = convert_from_path(f'pdf/{pdf_name}.pdf', 500)

    for i in range(len(pdf)):
        # Save pages as images in the pdf
        pdf[i].save(f'temp/{pdf_name}_page_' + str(i) + '.jpg', 'JPEG')

    files = [join('temp', f) for f in listdir('temp') if
             isfile(join('temp', f))]
    files.sort()

    raw_text = ''
    with fitz.open(f'pdf/{pdf_name}.pdf') as doc:
        raw_text = ''
        for page in doc:
            raw_text += page.get_text()
    splited_raw_text = raw_text.split('\n')
    extracted_fact['Дата включения в ЕГРЮЛ'] = splited_raw_text[splited_raw_text.index('ЕГРЮЛ') + 1].replace(' ', '')
    extracted_fact['ОГРН'] = splited_raw_text[splited_raw_text.index('ОГРН:') + 1].replace(' ', '')
    extracted_fact['инн'] = splited_raw_text[splited_raw_text.index('ИНН/КПП:') + 1].replace(' ', '')
    extracted_fact['кпп'] = splited_raw_text[splited_raw_text.index('ИНН/КПП:') + 2].replace(' ', '').replace('/', '')
    extracted_fact['Наименование отчетного периода'] = [i for i in splited_raw_text if i.startswith('за ')][0]

    for image_path in files:
        main_image = cv2.imread(image_path)

        detailed_data = get_detailed_cell_info(main_image)
        detailed_data.reverse()
        region_blocks = get_cell_blocks(main_image)
        _, dots = detect_table(main_image)

        for region_block in region_blocks:
            x, y, w, h = cv2.boundingRect(region_block)
            y -= 10
            h += 10
            text = extract_text(main_image[y:y + h, x:x + w])

            if check_image_contain_all_white_pixels(region_block, dots, cv2.countNonZero):
                # определяем просто как текст
                total_text_arr.append(text)
            else:
                # запрещенные слова игнорим
                text = extract_text(main_image[y:y + h, x:x + w])
                if any([bool(i) for i in forgetten_tabels_name if i in text.lower() and text.lower().startswith(i)]):
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

                        # иногда при распознавании текста совершенно пустой флагмент определяется как имеющий информацию
                        # ставлю проверку на это
                        text = extract_text(main_image[y:y + h, x:x + w])
                        if check_image_contain_all_white_pixels(detailed_data_cell, ~main_image, int_mean):
                            text = ''

                        # Проверка на присутствие галочек
                        cur_options = []
                        for option_name, path in options.items():
                            rule = _compare_images(main_image[y:y + h, x:x + w], cv2.imread(path))
                            if rule:
                                cur_options.append((option_name, rule))

                        if cur_options:
                            if len(cur_options) > 1 and cur_options[0][1][0] > cur_options[1][1][0]:
                                cur_options.reverse()
                            text = ';'.join(k for k, v in cur_options)

                        # проверка на попадание на повтор всей таблицы. Проверяем на вхождение подъячеек
                        if not check_image_contain_all_white_pixels(detailed_data_cell, dots, cv2.countNonZero) and \
                                OK_BUTTON_AREA * 100 < Cell(*detailed_data_cell_rect).get_area():
                            continue

                        detailed_data_cell_in_region.append(detailed_data_cell)
                        table_struct.add_value(Cell(x, y, w, h, text))

                total_table_arr.append(table_struct)

    delete_dir_content()
    prepare_and_save_data(pdf_name, total_text_arr, total_table_arr, extracted_fact, raw_text)
