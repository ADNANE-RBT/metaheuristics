class Graph:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = set()

    def in_bounds(self, node):
        (x, y) = node
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, node):
        return node not in self.walls

    def neighbors(self, node):
        (x, y) = node
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),  # Cardinal directions
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal directions
        ]
        results = [(x + dx, y + dy) for dx, dy in directions]
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results

    def cost(self, from_node, to_node):
        (x1, y1) = from_node
        (x2, y2) = to_node
        # Diagonal movement costs sqrt(2)
        if abs(x1 - x2) == 1 and abs(y1 - y2) == 1:
            return 1.414  # sqrt(2)
        else:
            return 1

    def add_wall(self, node):
        self.walls.add(node)