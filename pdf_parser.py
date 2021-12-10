import json
import re
from typing import Callable

import cv2 as opencv
import fitz
import numpy
from PIL import Image
from tika import parser

"""
PDF_FILE — путь к pdf-файлу
MAPPING — описывает маппинг.
    field — название результирующего поля
    value_regexp — регулярное выражение для значения
    postprocessor — постобработчик значения. если не указан явно (например, как lambda-функция), то постобработчик
        берётся из класса Postprocessors по названию поля
"""

PDF_FILE = 'pdf/110431401.pdf'

MAPPING = [
    {
        'field': 'egrul_date_of_inclusion',
        'value_regexp': re.compile(r"дата\s*включения\s*в\s*ЕГРЮЛ\s*(\d{1,2}.\d{1,2}.\d{4})", re.MULTILINE | re.DOTALL),
    },
    {
        'field': 'ogrn',
        'value_regexp': re.compile(r"ОГРН:\s*([\s\d]+)\s*дата", re.MULTILINE | re.DOTALL),
        'postprocessor': lambda x: re.sub(r'\D', '', x)
    },
    {
        'field': 'inn',
        'value_regexp': re.compile(r"ИНН/КПП:\s*([\s\d]+)/", re.MULTILINE | re.DOTALL),
        'postprocessor': lambda x: re.sub(r'\D', '', x)
    },
    {
        'field': 'kpp',
        'value_regexp': re.compile(r"ИНН/КПП:.*?/\s*([\s\d]+)", re.MULTILINE | re.DOTALL),
        'postprocessor': lambda x: re.sub(r'\D', '', x)
    },
    {
        'field': 'report_year',
        'value_regexp': re.compile(r"Отчет.*?за (\d{4})\s*г\.?", re.MULTILINE | re.DOTALL),
        'postprocessor': lambda x: int(x)

    },
    {
        'field': 'report_name',
        'value_regexp': re.compile(r"(Отчет.*?за \d{4}\s*г\.?)", re.MULTILINE | re.DOTALL),
        'postprocessor': lambda x: re.sub(r'\s+', ' ', x)
    },
    {
        'field': 'report_recipient',
        'value_regexp': re.compile(r"B\s*(.*?)\s*Отчет", re.MULTILINE | re.DOTALL),
        'postprocessor': lambda x: re.sub(r'\s+', ' ', x)
    },
    {
        'field': 'company_name',
        'value_regexp': re.compile(r"за \d{4}\s*г\.?(.*?)\n\(полное наименование некоммерческой организации\)",
                                   re.MULTILINE | re.DOTALL),
        'postprocessor': lambda x: (re.sub(r'\s+', ' ', x)).strip()
    },
    {

        'field': 'okved_description',
        'value_regexp': re.compile(r"Основные виды деятельности в отчетном периоде в соответствии с "
                                   r"учредительными документами(.*?)2 Предпринимательская деятельность",
                                   re.MULTILINE | re.DOTALL)
    },
    {
        # тут необходимо обработать галочки
        'field': 'entrepreneurial_activity',  # предпринимательская деятельность
        'value_regexp': re.compile(r"Предпринимательская деятельность.*?:(.*?)3 Источники формирования имущества",
                                   re.MULTILINE | re.DOTALL)
    },
    {
        # тут необходимо обработать галочки
        'field': 'source_of_financing',  # Источники формирования имущества
        'value_regexp': re.compile(r"Источники формирования имущества(.*?)4 Управление деятельностью:",
                                   re.MULTILINE | re.DOTALL)
    },
    {
        # тут необходимо обработать галочки
        'field': 'activity_management',  # Управление деятельностью
        'value_regexp': re.compile(r"Управление деятельностью:(.*?)Приложение:",
                                   re.MULTILINE | re.DOTALL)
    },
]


def detect_marked_item(item: str, *, width, threshold: float = 0.7):
    """
    Определяет отмечен ли галочкой горизонтальный элемент таблицы (галочка расположена справа от текста)
    :param item: текст, возле которого необходимо определить галочку
    :param width: если True, галочка должна располагаться в одной строке с текстом
                    если False, галочка должна располагаться ниже текста # TODO: доработать это
    :param threshold: порог совпадения шаблона с картинкой
    """
    doc = fitz.Document(PDF_FILE)
    for i, page in enumerate(doc):
        rect = page.search_for(item)
        if rect:
            # если нашли интересующую фразу, сохраняем страницу, на которой нашли фразу, как картинку
            # matrix = fitz.Matrix(3, 3)
            pix = page.get_pixmap()
            pix.save('img.png')

            im = Image.open('img.png')

            if width:
                # обрезаем всю строку в ширину
                im_crop = im.crop((rect[0][0] - 2, rect[0][1] - 6, page.cropbox[3], rect[0][3]))
            else:
                # обрезаем в длину
                im_crop = im.crop((rect[0][0], rect[0][1], rect[0][2], rect[0][3] + 20))
            im_crop.save('img_crop.png')

            mapping = {
                'table_content/not_marked.png': False,
                'table_content/marked.png': True
            }

            for template_name, condition in mapping.items():
                res = opencv.matchTemplate(opencv.imread('img_crop.png'), opencv.imread(template_name),
                                           opencv.TM_CCOEFF_NORMED)

                loc = numpy.where(res >= threshold)
                for l in loc:
                    if l.any():
                        return condition


class Postprocessors:
    @classmethod
    def okved_description(cls, value):
        """
        Сплитит блок по пунктам (1.1., например).
        """

        items = []
        for item in re.split(r'\d+.\d+.?', value):
            item = item.strip()

            if not item:
                continue

            items.append(re.sub(r'\s+', ' ', item))

        return items

    @classmethod
    def source_of_financing(cls, value):
        """
        Сплитит блок по пунктам (1.1., например).
        """

        phrases_to_skip = (
            'отметить знаком "V"',
            'Иные источники формирования имущества',
        )

        items = []
        for item in re.split(r'\d+.\d+.?', value):
            item = item.strip()

            if not item:
                continue

            need_skip = False
            for phrase in phrases_to_skip:
                if phrase in item:
                    need_skip = True
                    break
            if need_skip:
                continue

            items.append(re.sub(r'\s+', ' ', item))

        result = []

        # разбираемся с галочками
        for item in items:
            res = detect_marked_item(item, width=True)

            if res is True:
                result.append(item)

        return result

    @classmethod
    def entrepreneurial_activity(cls, value):
        """
        Сплитит блок по пунктам (1.1., например).
        """

        phrases_to_skip = (
            'иная деятельность:',
        )

        items = []
        for item in re.split(r'\d+.\d+.?', value):
            item = item.strip()

            if not item:
                continue

            need_skip = False
            for phrase in phrases_to_skip:
                if phrase in item:
                    need_skip = True
                    break
            if need_skip:
                continue

            items.append(re.sub(r'\s+', ' ', item))

        result = []

        # разбираемся с галочками
        for item in items:
            res = detect_marked_item(item, width=True)

            if res is True:
                result.append(item)

        return result

    @classmethod
    def activity_management(cls, value):
        """
        Сплитит блок по пунктам (1.1., например).
        """

        # это удалится из текста полностью
        phrases_to_remove = (
            'Полное наименование высшего органа управления',
            '(сведения о персональном составе указываются в листе А)',
            'Полное наименование исполнительного органа (нужное отметить знаком <V>)',
            'коллегиальный',
            'единоличный'
        )

        for phrase in phrases_to_remove:
            value = value.replace(phrase, '')

        # region Формируем блоки данных по подпунктам
        blocks = []
        for item in re.split(r'\d+.\d+.?', value):
            item = item.strip()

            if not item:
                continue

            blocks.append(re.sub(r'\s+', ' ', item))
        # endregion

        mapping = {
            'высший орган управления': re.compile(r'Высший орган управления\s*(?P<governing>.*?) '
                                                  r'Периодичность проведения заседаний в соответствии с учредительными '
                                                  r'документами\s*(?P<meetings_frequency>.*?) '
                                                  r'Проведено заседаний\s*(?P<meetings_held>\d+)'),
            'исполнительный орган': re.compile(r'Исполнительный орган\s*(?P<governing>.*?) '
                                               r'Периодичность проведения заседаний в соответствии с учредительными '
                                               r'документами\s*(\(\d*\))?(?P<meetings_frequency>.*?) '
                                               r'Проведено заседаний\s*(?P<meetings_held>.*?)')
        }

        result = []

        for block in blocks:
            for gov_level, regexp in mapping.items():

                res = None

                if regexp.search(block):
                    res = regexp.search(block).groupdict()

                if gov_level == 'исполнительный орган':
                    # для исполнительного органа управления необходимо распознать галочку
                    # в "коллегиальный" или "единоличный"
                    if res:

                        for x in ('коллегиальный', 'единоличный',):
                            r = detect_marked_item(x, width=False)
                            if r:
                                res.update({
                                    'type_of_governing_body': x
                                })

                if res:
                    result.append(res)

        return result


def content_preprocessing(content: str):
    """
    Предварительная обработка всего текста pdf
    """
    # убираем из текста название формы и нумерацию страниц
    content = ''.join(re.split(r'Страница:\s*\d*\s*\d*\s*Форма:\s*.*?\n', content))

    return content


def main():
    raw = parser.from_file(PDF_FILE)
    content = content_preprocessing(raw['content'])

    result = {}

    all_postprocessors = Postprocessors()

    for mapping_item in MAPPING:  # type: dict
        field = mapping_item['field']

        if field in result:
            raise Exception(f'поле `{field}` уже заполнено')

        value_reg = mapping_item['value_regexp'].search(content)

        if not value_reg:
            continue

        value = value_reg.group(1)

        postprocessor: Callable = mapping_item.get('postprocessor') or getattr(all_postprocessors, field, None)

        if postprocessor:
            # если указали постобработчик, запускаем его
            value = postprocessor(value)

        result.update({
            field: value
        })

    with open('result.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(result)


if __name__ == '__main__':
    main()
