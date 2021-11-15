from dataclasses import dataclass

TRESHOLD_PIXEL = 5
TRESHOLD_HEIGHT = 10


@dataclass
class Cell:
    x: int
    y: int
    width: int
    height: int

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def get_area(self):
        return self.height * self.width

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


class Table:

    def __init__(self):
        self.rows = dict()
        self.cur_row_index = 0

    def add_value(self, item: Cell):
        if len(self.rows) == 0:
            self.rows[self.cur_row_index] = [item]
            return

        cur_row_y = self.rows[self.cur_row_index][0].y
        cur_row_height = self.rows[self.cur_row_index][0].height

        # проверка на одну линию
        if cur_row_y - TRESHOLD_PIXEL <= item.y and cur_row_y + cur_row_height + TRESHOLD_HEIGHT >= item.y + item.height:

            if Cell.interception_perc(self.rows[self.cur_row_index][-1], second_cell)
            self.rows[self.cur_row_index].append(item)
        else:
            self.cur_row_index += 1
            self.rows[self.cur_row_index] = [item]


if __name__ == '__main__':
    a = Cell(x=38, y=18, width=6, height=81)
    b = Cell(x=41, y=19, width=22, height=71)
    perc = Cell.interception_perc(a, b)
