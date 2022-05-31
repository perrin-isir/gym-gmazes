class Cell(object):
    """Class for representing a cell in a 2D grid.

    Attributes:
        row (int): The row that this cell belongs to
        col (int): The column that this cell belongs to
        visited (bool): True if this cell has been visited by an algorithm
        active (bool):
        is_entry_exit (bool): True when the cell is the beginning or end of the maze
        walls (list):
        neighbours (list):
    """

    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.visited = False
        self.active = False
        self.is_entry_exit = None
        self.walls = {"top": True, "right": True, "bottom": True, "left": True}
        self.neighbours = list()

    def is_walls_between(self, neighbour):
        """Function that checks if there are walls between self and a neighbour cell.
        Returns true if there are walls between. Otherwise returns False.

        Args:
            neighbour The cell to check between

        Return:
            True: If there are walls in between self and neighbor
            False: If there are no walls in between the neighbors and self

        """
        if (
            self.row - neighbour.row == 1
            and self.walls["top"]
            and neighbour.walls["bottom"]
        ):
            return True
        elif (
            self.row - neighbour.row == -1
            and self.walls["bottom"]
            and neighbour.walls["top"]
        ):
            return True
        elif (
            self.col - neighbour.col == 1
            and self.walls["left"]
            and neighbour.walls["right"]
        ):
            return True
        elif (
            self.col - neighbour.col == -1
            and self.walls["right"]
            and neighbour.walls["left"]
        ):
            return True

        return False

    def remove_walls(self, neighbour_row, neighbour_col):
        if self.row - neighbour_row == 1:
            self.walls["top"] = False
            return True, ""
        elif self.row - neighbour_row == -1:
            self.walls["bottom"] = False
            return True, ""
        elif self.col - neighbour_col == 1:
            self.walls["left"] = False
            return True, ""
        elif self.col - neighbour_col == -1:
            self.walls["right"] = False
            return True, ""
        return False

    def add_walls(self, neighbour_row, neighbour_col):
        if self.row - neighbour_row == 1:
            self.walls["top"] = True
            return True, ""
        elif self.row - neighbour_row == -1:
            self.walls["bottom"] = True
            return True, ""
        elif self.col - neighbour_col == 1:
            self.walls["left"] = True
            return True, ""
        elif self.col - neighbour_col == -1:
            self.walls["right"] = True
            return True, ""
        return False
