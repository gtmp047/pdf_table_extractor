from table_ocr import pdf_to_images, extract_tables, extract_cells
import cv2
from imutils import contours
from PIL import Image, ImageEnhance
import numpy as np

tess_params = ["--psm", "6", "--oem", "3", "-l", "rus"]
image_path = 'rotated.jpg'


# image_pil = Image.open(image_path)
# image_pil = image_pil.crop((50, 50, image_pil.width-50, image_pil.height-50)).save(image_path)

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

MAX_COLOR_VAL = 255
BLOCK_SIZE = 15
SUBTRACT_FROM_MEAN = -2
BLUR_KERNEL_SIZE = (15, 15)
STD_DEV_X_DIRECTION = 0
STD_DEV_Y_DIRECTION = 0

blurred = cv2.GaussianBlur(image, BLUR_KERNEL_SIZE, STD_DEV_X_DIRECTION, STD_DEV_Y_DIRECTION)
img_bin = cv2.adaptiveThreshold(
    ~blurred,
    MAX_COLOR_VAL,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    BLOCK_SIZE,
    SUBTRACT_FROM_MEAN,
)
cv2.imwrite("img_bin.png", img_bin)


vertical = horizontal = img_bin.copy()
SCALE = 5
image_width, image_height = horizontal.shape
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (400, 1))
horizontally_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, horizontal_kernel)
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
vertically_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, vertical_kernel)

horizontally_dilated = cv2.dilate(horizontally_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
vertically_dilated = cv2.dilate(vertically_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))

mask = horizontally_dilated + vertically_dilated

cv2.imwrite("mask_rotated.png", mask)

# get_cell_blocks
