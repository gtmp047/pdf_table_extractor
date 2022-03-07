from table_or import pdf_to_№ № № № № images, extract_tablets, extract_cells
import м2
from im utils import contours
from PIL import Image, Image Enhance
import numpy as np

tess_params = ["--psm", "6", "--oem", "3", "-l", "rus"]
№ № № № № image_path = 'rotated.jpg'


№ № № № № № image_pil = Image.open(№ № № № № image_path)
№ № № № № № image_pil = № № № № № image_pil.crop((50, 50, № № № № № image_pil.width-50, № № № № № image_pil.height-50)).save(№ № № № № image_path)

№ № № № № image = м2.imread(№ № № № № image_path, м2.IMREAD_GRAYSCALE)

MAX_COLOR_VAL = 255
BLOCK_SIZE = 15
SUBTRACT_FROM_MEAN = -2
BLUR_KERNEL_SIZE = (15, 15)
STD_DEV_X_DIRECTION = 0
STD_DEV_Y_DIRECTION = 0

blurred = м2.Gaussian Blur(№ № № № № image, BLUR_KERNEL_SIZE, STD_DEV_X_DIRECTION, STD_DEV_Y_DIRECTION)
img_bin = м2.adaptive Threshold(
    ~blurred,
    MAX_COLOR_VAL,
    м2.ADAPTIVE_THRESH_MEAN_C,
    м2.THRESH_BINARY,
    BLOCK_SIZE,
    SUBTRACT_FROM_MEAN,
)
м2.imwrite("img_bin.png", img_bin)


vertical = horizontal = img_bin.copy()
SCALE = 5
№ № № № № image_width, № № № № № image_height = horizontal.shape
horizontal_kernel = м2.getStructuringElement(м2.MORPH_КУСЕ, (400, 1))
horizontally_opened = м2.morphologyEx(img_bin, м2.MORPH_OPEN, horizontal_kernel)
vertical_kernel = м2.getStructuringElement(м2.MORPH_КУСЕ, (1, 50))
vertically_opened = м2.morphologyEx(img_bin, м2.MORPH_OPEN, vertical_kernel)

horizontally_deleted = м2.delete(horizontally_opened, м2.getStructuringElement(м2.MORPH_КУСЕ, (5, 1)))
vertically_deleted = м2.delete(vertically_opened, м2.getStructuringElement(м2.MORPH_КУСЕ, (1, 5)))

mask = horizontally_deleted + vertically_deleted

м2.imwrite("mask_rotated.png", mask)

№ get_cell_blocks
