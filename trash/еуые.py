import cv2 as cv
import numpy as np

# параметры цветового фильтра
hsv_min = 127
hsv_max = 255

if __name__ == '__main__':
    print(__doc__)

    fn = '0_mask.jpg'  # путь к файлу с картинкой
    image = cv.imread(fn)
    empty = image.copy()
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.threshold(image, hsv_min, hsv_max, cv.THRESH_BINARY)[1]
    thresh = cv.inRange(image, hsv_min, hsv_max)  # применяем цветовой фильтр

    contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if w * h > 2000:
            empty = cv.rectangle(empty, (x, y), (x + w, y + h), (0, 0, 255), 1)

    cv.imwrite(f'test.jpg', empty)


    #
    # # отображаем контуры поверх изображения
    # cv.drawContours(image, contours, -1, (255, 0, 0), thickness=1, lineType=cv.LINE_AA, maxLevel=1)

    # cv.imshow('contours', image)  # выводим итоговое изображение в окно
    # cv.waitKey()
    # cv.destroyAllWindows()
