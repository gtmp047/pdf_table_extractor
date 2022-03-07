import json
import math
import random
import re
import shutil
from copy import copy
from os import walk, unlink
from os.path import join
from dataclasses import dataclass

import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance
from dataclasses import dataclass
from imutils import contours
from pdf2image import convert_from_bytes
from pyaspeller import YandexSpeller
from scipy import ndimage

from cas_http_anonym_web import HttpAnonymWeb, DisposableHostingSessionManagerWithNetworkErrorsHandler
from vitok_info_lib_performers.exceptions import handle_exception
from vitok_info_lib_performers.interaction import (get_input_data, generate_input_schema, set_output_data,
                                                   generate_output_schema)
from vitok_info_lib_performers.ontology import CompanyName, INN, TypeOfReport, OGRN, URL, EgrulDateOfInclusion, KPP, \
    ReportYear, ReportRecipient, ReportName, Address, OKVEDDescription, BusinessActivities, SourcesFormingProperty, \
    TypeOfGoverningAuthority, TypeofGoverningBody, MeetingsHeld, MeetingsFrequency, AuthorityName, FullNameOfSigner, \
    Position, SignDate, Fullname, BirthdayString, Citizenship, IdentityDocumentData

"""
Краулер обрабатывает данные pdf, полученные с сайта минюста
На вход тип отчета и ссылка на сам отчет
"""

INPUT_SCHEMA = {
    'type': 'object',
    'properties': {
        URL.name: URL,
        TypeOfReport.name: TypeOfReport
    }
}

OUTPUT_SCHEMA = {
    'type': 'object',
    'properties': {
        'result': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'input': INPUT_SCHEMA,
                    'output': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                CompanyName.name: CompanyName,
                                INN.name: INN,
                                OGRN.name: OGRN,
                                EgrulDateOfInclusion.name: EgrulDateOfInclusion,
                                KPP.name: KPP,
                                ReportYear.name: ReportYear,
                                ReportName.name: ReportName,
                                ReportRecipient.name: ReportRecipient,
                                Address.name: Address,

                                OKVEDDescription.name: {
                                    'type': 'array',
                                    'items': {'type': OKVEDDescription.type}
                                },

                                BusinessActivities.name: {
                                    'type': 'array',
                                    'items': {'type': BusinessActivities.type}
                                },

                                SourcesFormingProperty.name: {
                                    'type': 'array',
                                    'items': {'type': SourcesFormingProperty.type}
                                },

                                'Управление деятельностью': {
                                    'type': 'array',
                                    'items': {
                                        'type': 'object',
                                        'properties': {
                                            AuthorityName.name: AuthorityName,
                                            TypeOfGoverningAuthority.name: TypeOfGoverningAuthority,
                                            TypeofGoverningBody.name: TypeofGoverningBody,
                                            MeetingsHeld.name: MeetingsHeld,
                                            MeetingsFrequency.name: MeetingsFrequency
                                        }
                                    }
                                },

                                'Подпись': {
                                    'type': 'object',
                                    'properties': {
                                        AuthorityName.name: AuthorityName,
                                        FullNameOfSigner.name: FullNameOfSigner,
                                        Position.name: Position,
                                        SignDate.name: SignDate
                                    }

                                },

                                'Состав руководящих органов': {
                                    'type': 'array',
                                    'items': {
                                        'type': 'object',
                                        'properties': {
                                            Fullname.name: Fullname,
                                            BirthdayString.name: BirthdayString,
                                            Citizenship.name: Citizenship,
                                            IdentityDocumentData.name: IdentityDocumentData,
                                            Address.name: Address,
                                            Position.name: Position
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

INPUT_JSON_SCHEMA = generate_input_schema(INPUT_SCHEMA)
OUTPUT_JSON_SCHEMA = generate_output_schema(OUTPUT_SCHEMA)
ID = '75c0e4c9-b67d-4140-be8f-02a83e8ee58a'

ALLOWED_REPORT_TYPE = {'ОН0001'}
SIMPLISITY_TRESHOLD = 0.9
MIN_AREA = 2000
OK_BUTTON_AREA = 89 * 87
TRESHOLD_PIXEL = 20
TRESHOLD_HEIGHT = 40
THRESHOLD_MIN = 127
THRESHOLD_MAX = 255

TESSERACT_CONF = r'--oem 3 --psm 6'
SPELLER = YandexSpeller()

YES = 'ДА'
NO = 'НЕТ'
BOOLEAN_STRING_SET = {NO, YES}
BUTTON_OPTIONS = {
    'НЕТ': 'table_content/not_ok.png',
    YES: 'table_content/ok.png',
}



# region classes
@dataclass
class Cell:
    x: int
    y: int
    width: int
    height: int
    text: str

    def __init__(self, x, y, width, height, text=''):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text

    def get_area(self):
        return self.height * self.width

    def startswith(self, char):
        return self.text.startswith(char)

    @classmethod
    def cells_interception_area(cls, first_cell, second_cell):
        #  x11, y11 - левая верхняя точка первого прямоугольника
        #  x12, y12 - правая нижняя точка первого прямоугольника
        #  x21, y21 - левая верхняя точка второго прямоугольника
        #  x22, y22 - правая нижняя точка второго прямоугольника

        x11, y11 = first_cell.x, first_cell.y
        x12, y12 = first_cell.x + first_cell.width, first_cell.y + first_cell.height
        x21, y21 = second_cell.x, second_cell.y
        x22, y22 = second_cell.x + second_cell.width, second_cell.y + second_cell.height

        inter_x1 = min(x11, x21)
        inter_x2 = max(x12, x22)
        inter_y1 = min(y11, y21)
        inter_y2 = max(y12, y22)

        width = (x12 - x11) + (x22 - x21) - (inter_x2 - inter_x1)
        height = (y12 - y11) + (y22 - y21) - (inter_y2 - inter_y1)

        if width < 0 or height < 0:
            return 0

        return width * height

    @classmethod
    def interception_perc(cls, first_cell, second_cell):
        first_cell_area = first_cell.get_area()
        second_cell_area = second_cell.get_area()

        interception_area = Cell.cells_interception_area(first_cell, second_cell)

        return interception_area / min(first_cell_area, second_cell_area)

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


class Table:

    def __init__(self):
        self.rows = list()
        self.cur_row_index = 0
        self.total_area = 0

    def __getitem__(self, index):
        if self.cell_count() > 0:
            return self.rows[index]
        else:
            return ''

    def cell_count(self):
        return sum([len(v) for v in self.rows])

    def add_value(self, item: Cell):
        if len(self.rows) == 0:
            self.rows.append([item])
            self.total_area += item.get_area()
            return

        cur_row_y = self.rows[self.cur_row_index][0].y
        cur_row_height = self.rows[self.cur_row_index][0].height

        # проверка если попали на внитренние блоки. Их нужно удалить
        while Cell.interception_perc(item, self.rows[self.cur_row_index][-1]) >= SIMPLISITY_TRESHOLD:
            del self.rows[self.cur_row_index][-1]

            if not self.rows[self.cur_row_index] and self.cur_row_index:
                self.cur_row_index -= 1
                del self.rows[self.cur_row_index]
                break

            if not bool(self.rows[self.cur_row_index]):
                break

        # проверка на одну линию
        if (cur_row_y - TRESHOLD_PIXEL <= item.y and
                cur_row_y + cur_row_height + TRESHOLD_HEIGHT >= item.y + item.height):
            self.rows[self.cur_row_index].append(item)
            self.total_area += item.get_area()
        else:
            if len(self.rows[self.cur_row_index]):
                # переносим все на новую строку
                self.cur_row_index += 1
                self.rows.append([item])
                self.total_area += item.get_area()
            else:
                self.rows[self.cur_row_index].append(item)
                self.total_area += item.get_area()


# endregion

# region table_extract_funcs
def sharp_image(numpy_image):
    image_pil = Image.fromarray(numpy_image.astype('uint8'), 'L')
    # image_pil = image_pil.crop((0, 30, image_pil.width, image_pil.height-30)).save()
    enhancer = ImageEnhance.Sharpness(image_pil)
    factor = 5
    sharped_image = enhancer.enhance(factor)
    sharped_image.save('sharpened-image.png')

    return np.array(sharped_image)


def centrate_image(img_gray):
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    angles = []

    for [[x1, y1, x2, y2]] in lines:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)
    if median_angle <= 30 and median_angle >= -30:
        img_rotated = ndimage.rotate(img_gray, median_angle, reshape=False)
    else:
        return img_gray

    return img_rotated


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


def get_detailed_cell_info(grey_table_image_contour):
    image = cv2.threshold(grey_table_image_contour, THRESHOLD_MIN, THRESHOLD_MAX, cv2.THRESH_BINARY)[1]
    thresh = cv2.inRange(image, THRESHOLD_MIN, THRESHOLD_MAX)

    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_area_cnts = []
    for cnt in contours:
        _, _, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > MIN_AREA and h >= 25 and w >= 25:
            min_area_cnts.append(cnt)

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


def get_image_with_vert_and_hor(gray):
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    blank_image = ~np.zeros(gray.shape, np.uint8)

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


def extract_text(image):
    return pytesseract.image_to_string(image, lang='rus',
                                       config=TESSERACT_CONF).rstrip().replace('\n', ' ')


# endregion


def _extract_ОН0001(raw_text, extracted_text_arr, extracted_table_arr):
    extracted_fact = dict()

    splited_raw_text = raw_text.split('\n')
    extracted_fact[EgrulDateOfInclusion.name] = splited_raw_text[splited_raw_text.index('ЕГРЮЛ') + 1].replace(' ', '')
    extracted_fact[OGRN.name] = splited_raw_text[splited_raw_text.index('ОГРН:') + 1].replace(' ', '')
    extracted_fact[INN.name] = splited_raw_text[splited_raw_text.index('ИНН/КПП:') + 1].replace(' ', '')
    extracted_fact[KPP.name] = splited_raw_text[splited_raw_text.index('ИНН/КПП:') + 2].replace(' ', '').replace('/',
                                                                                                                 '')
    extracted_fact[ReportYear.name] = int(
        re.search(r'\d{4}', [i for i in splited_raw_text if i.startswith('за ')][0]).group())
    extracted_fact[ReportName.name] = str(extracted_text_arr[1])
    extracted_fact[ReportRecipient.name] = extracted_text_arr[0]
    extracted_fact[CompanyName.name] = extracted_text_arr[2].replace(
        '(полное наименование некоммерческой организации)', '').strip()
    extracted_fact[Address.name] = extracted_text_arr[3].replace(
        '(адрес (место нахождения) некоммерческой организации)', '').strip()

    # Получение списка деятельности
    activity_list = []
    activity_table = [table for table in extracted_table_arr if table[0][0].startswith('1') and table][0]
    for row in activity_table:
        if row[0].text != '1' and row[1].text:
            activity_list.append(row[1].text)
    extracted_fact.update({
        OKVEDDescription.name: activity_list
    })

    # получение предпринимательской деятельности
    enterprise_activity_list = []
    enterprise_activity_table = [table for table in extracted_table_arr if table[0][0].startswith('2')][0]
    for row in enterprise_activity_table:
        if row[-1].text == YES:
            enterprise_activity_list.append({row[-2].text})
    extracted_fact.update({
        BusinessActivities.name: enterprise_activity_list
    })

    # получение источников формирования имущества
    sources_of_property_list = []
    sources_of_property_tables = [table for table in extracted_table_arr if table[0][0].startswith('3')]
    for table in sources_of_property_tables:
        for row in table:
            if row[-1].text == YES:
                sources_of_property_list.append(row[-1].text)
        extracted_fact.update({
            SourcesFormingProperty.name: sources_of_property_list
        })

    # ищем забагованную 4.2 которая лежит в Управление деятельностью:
    table_initiator_index, row_initiator_index = \
        [(table_index, row_index) for table_index, table in enumerate(extracted_table_arr)
         for row_index, row in enumerate(table.rows)
         if row[0].text == '4.2'][0]

    if table_initiator_index:
        new_table = Table()
        new_table.rows = extracted_table_arr[table_initiator_index][row_initiator_index:]
        del extracted_table_arr[table_initiator_index].rows[row_initiator_index:]
        extracted_table_arr.append(new_table)

    # получение списка агентов управления деятельностью:
    agent_list = []
    agent_tables = [table for table in extracted_table_arr if table[0][0].startswith('4')]
    for table in agent_tables:
        temp_agent = {}

        if table[0][0].text != '4':
            #  попали на нормальную таблицу
            for row in table:
                for i, item in enumerate(row):

                    if 'Исполнительный орган' in item.text:
                        temp_agent[AuthorityName.name] = row[i + 1].text

                    if ';' in item.text:
                        if item.text.split(';')[0] == YES:
                            temp_agent[TypeofGoverningBody.name] = 'коллегиальный'
                        if item.text.split(';')[1] == YES:
                            temp_agent[TypeofGoverningBody.name] = 'единоличный'

                    if 'Периодичность проведения заседаний' in item.text:
                        temp_agent[MeetingsFrequency.name] = row[i + 1].text

                    if 'Проведено заседаний' in item.text:
                        if row[i + 1].text.isdigit():
                            temp_agent[MeetingsHeld.name] = int(row[i + 1].text)

        else:
            # попали на первую таблицу где забили на форматирование. Ищем в тупую
            for row in table:
                for item in row:
                    if 'Высший орган управления' in item.text:
                        temp_str = item.text
                        temp_agent[TypeOfGoverningAuthority.name] = 'Высший орган управления'
                        temp_str = temp_str.replace('Высший орган управления', '')
                        temp_str = temp_str.replace('(сведения о персональном составе указываются в листе А)', '')
                        temp_agent[AuthorityName.name] = temp_str.strip()

                    if 'Периодичность проведения заседаний' in item.text:
                        temp_str = item.text
                        temp_str = temp_str.replace('Периодичность проведения заседаний в соответствии с', '')
                        temp_str = temp_str.replace('учредительными документами', '')
                        temp_agent[MeetingsFrequency.name] = temp_str.strip()

                    if 'Проведено заседаний' in item.text:
                        temp_str = item.text
                        temp_str = temp_str.replace('Проведено заседаний', '')
                        temp_agent[MeetingsHeld.name] = int(temp_str.strip())

        agent_list.append(temp_agent)
    extracted_fact.update({
        'Управление деятельностью': agent_list
    })

    # подпись
    sing_info = dict()
    com_name_index = \
        [i for i, v in enumerate(extracted_text_arr) if
         v.startswith('Сведения о персональном составе руководящих органов')][0]
    if com_name_index:
        com_name = extracted_text_arr[com_name_index + 1].replace('(полное наименование руководящего органа)', '')
        sing_info.update({
            AuthorityName.name: com_name.strip()
        })

    index_of_sign = [i for i, v in enumerate(extracted_text_arr) if
                     v.startswith('Лицо, имеющее право без доверенности действовать')][0]
    if index_of_sign:
        sign_text = extracted_text_arr[index_of_sign + 1].replace(
            '(подпись)', '').replace('(дата)', '').replace('(фамилия, имя, отчество, занимаемая должность)', '')

        # по протоколу должность должна быть отделена от фио, но на это правило иногда забивают (ошибочно)
        if ',' in sign_text:
            fio, other = sign_text.split(',')
        else:
            fio, other = ' '.join(sign_text.split(' ')[:3]), ' '.join(sign_text.split(' ')[3:])
        sign_date = re.search(r'\d{2}.\d{2}.\d{4}', other).group()
        sign_position = other.replace(sign_date, '')

        sing_info.update({
            FullNameOfSigner.name: fio.strip(),
            Position.name: sign_position.strip(),
            SignDate.name: sign_date
        })

        extracted_fact.update({
            'Подпись': sing_info
        })

    # определение списка руководящего состава
    ruler_list = []
    ruler_tables = [table for table in extracted_table_arr if table[0][0].startswith('Фамилия')]
    for table in ruler_tables:
        temp_ruller = []
        for row in table:
            temp_ruller.append(row[-1].text)

        ruler_list.append(
            {
                Fullname.name: temp_ruller[0],
                BirthdayString.name: temp_ruller[1],
                Citizenship.name: temp_ruller[2],
                IdentityDocumentData.name: temp_ruller[3],
                Address.name: temp_ruller[4],
                Position.name: temp_ruller[5]
            }
        )

    extracted_fact.update({
        'Состав руководящих органов': ruler_list
    })

    return extracted_fact


def print_region_on_image(image, regions_data, filename, rgb=(255, 0, 0)):
    table_image = copy(image)
    for c in regions_data:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > MIN_AREA:
            rgb = (random.randint(10, 255), random.randint(10, 255),random.randint(10, 255))
            table_image = cv2.rectangle(image, (x, y), (x + w, y + h), rgb, 2)

    cv2.imwrite(f'{filename}.jpg', table_image)


def get_image_bin(image):
    MAX_COLOR_VAL = 255
    BLOCK_SIZE = 15
    SUBTRACT_FROM_MEAN = -3
    BLUR_KERNEL_SIZE = (15, 15)
    STD_DEV_X_DIRECTION = 0
    STD_DEV_Y_DIRECTION = 0

    blurred = cv2.GaussianBlur(image, BLUR_KERNEL_SIZE, STD_DEV_X_DIRECTION, STD_DEV_Y_DIRECTION)
    img_bin = cv2.adaptiveThreshold(
        ~sharp_image(blurred),
        MAX_COLOR_VAL,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        BLOCK_SIZE,
        SUBTRACT_FROM_MEAN,
    )

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (400, 1))
    horizontally_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vertically_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, vertical_kernel)

    horizontally_dilated = cv2.dilate(horizontally_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1)))
    vertically_dilated = cv2.dilate(vertically_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10)))

    return img_bin, horizontally_dilated + vertically_dilated


def extract_info_pdf(pdf_pages):
    total_text_arr = []
    total_table_arr = []

    translate_list = []
    import time


    for page_index, page in enumerate(pdf_pages):
        # поворачиваем картинку
        main_image = cv2.cvtColor(np.array(page.convert('RGB')), cv2.COLOR_BGR2GRAY)
        main_image = centrate_image(main_image)
        main_image = sharp_image(main_image)

        # cut image
        image_pil = Image.fromarray(main_image.astype('uint8'), 'L')
        image_pil = image_pil.crop((50, 50, image_pil.width - 50, image_pil.height - 50))
        image_pil = np.array(image_pil)
        cv2.imwrite(f'trash/{page_index}_rotated.jpg', image_pil)

        img_bin, mask = get_image_bin(image_pil)
        cv2.imwrite(f'trash/{page_index}_img_bin.jpg', img_bin)
        cv2.imwrite(f'trash/{page_index}_mask.jpg', mask)

        # get detailed data
        detailed_data = get_detailed_cell_info(mask)
        print_region_on_image(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), detailed_data, f'trash/{page_index}_detailed_data',
                              rgb=(0, 0, 255))
        detailed_data.reverse()


        for region_index, region_block in enumerate(detailed_data):
            time.sleep(0.2)
            x, y, w, h = cv2.boundingRect(region_block)
            unique_region_index = f'{page_index}_{region_index}'



            extracted_text = extract_text(image_pil[y:y + h, x:x + w])
            yandex_text = SPELLER._apply_suggestion(extracted_text, SPELLER._spell_text(extracted_text))
            if extracted_text!=yandex_text:
                cv2.imwrite(f'data_image/{unique_region_index}_region.jpg', image_pil[y:y + h, x:x + w])
                translate_list.append(tuple([unique_region_index, extracted_text, yandex_text]))

            print(f'done {unique_region_index}')

    with open('cvr_results.txt', 'a') as f:
        for item in translate_list:
            f.write(f'{str(item)}\n')



        # table_struct = Table()

    #     # выделяем блоки в таблице
    #     detailed_data_cell_in_region = []
    #     for detailed_data_cell in detailed_data:
    #         detailed_data_cell_rect = cv2.boundingRect(detailed_data_cell)
    #         region_block_rect = cv2.boundingRect(region_block)
    #
    #         simplisity = Cell.interception_perc(Cell(*region_block_rect), Cell(*detailed_data_cell_rect))
    #         if simplisity >= SIMPLISITY_TRESHOLD and region_block_rect != detailed_data_cell_rect:
    #             x, y, w, h = cv2.boundingRect(detailed_data_cell)
    #
    #             # иногда при распознавании текста совершенно пустой флагмент определяется как имеющий информацию
    #             # ставлю проверку на это
    #             text = extract_text(main_image[y:y + h, x:x + w])
    #             if check_image_contain_all_white_pixels(detailed_data_cell, ~main_image, int_mean):
    #                 text = ''
    #
    #             # Проверка на присутствие галочек
    #             cur_options = []
    #             for option_name, path in BUTTON_OPTIONS.items():
    #                 rule = _compare_images(main_image[y:y + h, x:x + w], cv2.imread(path))
    #                 if rule:
    #                     cur_options.append((option_name, rule))
    #
    #             if cur_options:
    #                 if len(cur_options) > 1 and cur_options[0][1][0] > cur_options[1][1][0]:
    #                     cur_options.reverse()
    #                 text = ';'.join(k for k, v in cur_options)
    #
    #             # проверка на попадание на повтор всей таблицы. Проверяем на вхождение подъячеек
    #             if not check_image_contain_all_white_pixels(detailed_data_cell, dots, cv2.countNonZero) and \
    #                     OK_BUTTON_AREA * 100 < Cell(*detailed_data_cell_rect).get_area():
    #                 continue
    #
    #             detailed_data_cell_in_region.append(detailed_data_cell)
    #             table_struct.add_value(Cell(x, y, w, h, text))
    #
    #     total_table_arr.append(table_struct)
    #
    # return total_text_arr, total_table_arr


def get_data_from_pdf(pdf_bytes, report_type):
    raw_text = ''
    # получаем постраничное представление pdf
    pdf_pages = convert_from_bytes(pdf_bytes)
    extracted_text_arr, extracted_table_arr = extract_info_pdf(pdf_pages)
    pass
    return _extract_ОН0001(raw_text, extracted_text_arr, extracted_table_arr)


def process() -> dict:
    output = []

    result = {
        'result': [{
            # 'input': input_data,
            # 'output': output
        }],
        'raw_output': {
        }
    }

    input_url = 'http://unro.minjust.ru/CMS/documentmanagement/directlink.aspx?Inline=false&OwnerKey=3&qid=1&doc=103217'
    # report_type = input_data.get(TypeOfReport.name)

    # if report_type not in ALLOWED_REPORT_TYPE:
    #     return result

    # # with HttpAnonymWeb(session_manager=DisposableHostingSessionManagerWithNetworkErrorsHandler) as sess:
    # import requests
    # resp = requests.get(input_url)
    with open('pdf/1.pdf', 'rb') as f:
        # resp_bytes = resp.content
        resp_bytes = f.read()
    temp_output = get_data_from_pdf(resp_bytes, 'ОИА001')
    if temp_output:
        output.append(temp_output)

    return result

output_data = process()
# if __name__ == '__main__':
#     input_data = get_input_data(INPUT_SCHEMA)
#
#     try:
#         output_data = process()
#     except Exception as e:
#         handle_exception(e)
#     else:
#         set_output_data(output_data, OUTPUT_SCHEMA)
