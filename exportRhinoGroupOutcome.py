from misc import *
import pickle, csv
from placer import *

class Floor:
    def __init__(self, location, dimensions, direction, walls):
        self.location = location
        self.dimensions = dimensions
        self.direction = direction
        self.walls = walls

class Wall:
    def __init__(self, location, dimensions, direction, floor):
        self.location = location
        self.dimensions = dimensions
        self.direction = direction
        self.floor = floor

class Exporter:
    def __init__(self, puzzle, scale, nr):
        self.puzzle = puzzle
        self.board = puzzle.board
        self.groups = puzzle.finishedGroups
        self.cMatrix = puzzle.cMatrix

        self.scale = scale
        self.wallWidth = self.scale[0]/20
        self.wallHeight = 4000
        self.objects = []
        self.getObjects()

        self.nr = nr

    def getObjects(self):
        for gi in self.groups:
            g = self.groups[gi]
            floors = []
            for b in g.blocksPlaced:
                f = Floor([b.x * self.scale[0], b.y * self.scale[1], 0], [self.scale[0], self.scale[1], b.value[0] * self.scale[2]], [1,0], [] )
                try:
                    c = self.board.cells[b.x-1, b.y]
                    if c and c.nr != b.nr:
                        l = [f.location[0], f.location[1], f.location[2] + f.dimensions[2]]
                        w = Wall(l, [self.wallWidth, f.dimensions[1], self.cMatrix[b.nr, c.nr] * self.wallHeight], [1,0], f)
                        f.walls.append(w)
                except IndexError:
                    pass

                try:
                    c = self.board.cells[b.x+1, b.y]
                    if c and c.nr != b.nr:
                        l = [f.location[0] + f.dimensions[0] - self.wallWidth, f.location[1], f.location[2] + f.dimensions[2]]
                        w = Wall(l, [self.wallWidth, f.dimensions[1], self.cMatrix[b.nr, c.nr] * self.wallHeight], [1,0], f)
                        f.walls.append(w)
                except IndexError:
                    pass

                try:
                    c = self.board.cells[b.x, b.y-1]
                    if c and c.nr != b.nr:
                        l = [f.location[0], f.location[1], f.location[2] + f.dimensions[2]]
                        w = Wall(l, [f.dimensions[0], self.wallWidth, self.cMatrix[b.nr, c.nr] * self.wallHeight], [1,0], f)
                        f.walls.append(w)
                except IndexError:
                    pass

                try:
                    c = self.board.cells[b.x, b.y+1]
                    if c and c.nr != b.nr:
                        l = [f.location[0], f.location[1] + f.dimensions[1] - self.wallWidth, f.location[2] + f.dimensions[2]]
                        w = Wall(l, [f.dimensions[0], self.wallWidth, self.cMatrix[b.nr, c.nr] * self.wallHeight], [1,0], f)
                        f.walls.append(w)
                except IndexError:
                    pass

                floors.append(f)
            self.objects.append(floors)

    def export(self):
        with open("RhinoCSVs/testRhino7"+ str(self.nr) +".csv", 'w') as newcsvfile:
            writer = csv.writer(newcsvfile)
            for o in self.objects:
                for f in o:
                    writer.writerow([f.location[0], f.location[1], f.location[2], f.dimensions[0], f.dimensions[1], f.dimensions[2]])
                    for w in f.walls:
                        writer.writerow([w.location[0], w.location[1], w.location[2], w.dimensions[0], w.dimensions[1], w.dimensions[2]])
