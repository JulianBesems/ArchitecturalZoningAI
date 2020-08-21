from typing import List, Tuple, Union, Dict
import shapely
from shapely.geometry import LineString
import math

class Slope(object):
    __slots__ = ('rise', 'run')
    def __init__(self, rise: int, run: int):
        self.rise = rise
        self.run = run

class Line(object):
    __slots__ = ('a', 'b')
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def intersects(self, l):
        a1 = (self.a.x, self.a.y)
        b1 = (self.b.x, self.b.y)
        a2 = (l.a.x, l.a.y)
        b2 = (l.b.x, l.b.y)
        l1 = LineString([a1, b1])
        l2 = LineString([a2, b2])
        ip = l1.intersection(l2)
        if ip:
            return((ip.x, ip.y))
        else:
            return None


class Point(object):
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x: int, y: int, z: int):
        self.x = x
        self.y = y
        self.z = z

    def copy(self) -> 'Point':
        x = self.x
        y = self.y
        z = self.z
        return Point(x, y, z)

    def to_dict(self) -> Dict[str, int]:
        d = {}
        d['x'] = self.x
        d['y'] = self.y
        d['z'] = self.z
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> 'Point':
        return Point(d['x'], d['y'], d['z'])

    def __eq__(self, other: Union['Point', Tuple[int, int, int]]) -> bool:
        if isinstance(other, tuple) and len(other) == 2:
            return other[0] == self.x and other[1] == self.y and other[2] == self.z
        elif isinstance(other, Point) and self.x == other.x and self.y == other.y and self.z == other.z:
            return True
        return False

    def __sub__(self, other: Union['Point', Tuple[int, int, int]]) -> 'Point':
        if isinstance(other, tuple) and len(other) == 2:
            diff_x = self.x - other[0]
            diff_y = self.y - other[1]
            diff_z = self.z - other[2]
            return Point(diff_x, diff_y, diff_z)
        elif isinstance(other, Point):
            diff_x = self.x - other.x
            diff_y = self.y - other.y
            diff_z = self.z - other.z
            return Point(diff_x, diff_y, diff_z)
        return None

    def __rsub__(self, other: Tuple[int, int, int]):
        diff_x = other[0] - self.x
        diff_y = other[1] - self.y
        diff_z = other[2] - self.z
        return Point(diff_x, diff_y, diff_z)

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))

    def __str__(self) -> str:
        return '({}, {})'.format(self.x, self.y, self.z)

    def collidesWithBox(self, box):
        x = self.x
        y = self.y
        z = self.z

        x_max2 = box.b.x
        x_min2 = box.a.x
        y_max2 = box.b.y
        y_min2 = box.a.y
        z_max2 = box.b.z
        z_min2 = box.a.z

        isColliding = (x >= x_min2 and x_max2 >= x) and \
                      (y >= y_min2 and y_max2 >= y) and \
                      (z >= z_min2 and z_max2 >= z)

        """isColliding = ((x_max >= x_min2 and x_max <= x_max2) \
                    or (x_min <= x_max2 and x_min >= x_min2)) \
                    and ((y_max >= y_min2 and y_max <= y_max2) \
                    or (y_min <= y_max2 and y_min >= y_min2)) \
                    and ((z_max >= z_min2 and z_max <= z_max2) \
                    or (z_min <= z_max2 and z_min >= z_min2))"""
        return isColliding

    def move(self, v):
        self.x = self.o.x + v[0]
        self.y = self.o.y + v[1]
        self.z = self.o.z + v[2]

    def rotate(self, o, angle):
        point = [self.x, self.y]
        origin = [o.x, o.y]
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

class Circle(object):
    def __init__(self, p: Point, r: float):
        self.x = p.x
        self.y = p.y
        self.z = p.z
        self.r = r

class Vector(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.x = self.b.x - self.a.x
        self.y = self.b.y - self.a.y
        self.z = self.b.z - self.a.z
        length = math.sqrt(self.x **2 + self.y **2 + self.z **2)

    def unionise(self):
        d = math.sqrt(self.x **2 + self.y **2 + self.z** 2)
        self.x = self.x/d
        self.y = self.y/d
        self.z = self.z/d
        length = 1

    def invert(self):
        self.x = -self.x
        self.y = -self.y
        self.z = -self.z

class Wall(object):
    def __init__(self, a, b, cv):
        self.a = a

        self.b = b

        self.minx = min(self.a.x, self.b.x)
        self.miny = min(self.a.y, self.b.y)
        self.minz = min(self.a.z, self.b.z)
        self.maxx = max(self.a.x, self.b.x)
        self.maxy = max(self.a.y, self.b.y)
        self.maxz = max(self.a.z, self.b.z)
        self.connectionV = cv
        self.middle = Point((self.a.x + self.b.x)/2, (self.a.y + self.b.y)/2, (self.a.z + self.b.z)/2)
        self.freeSpace = abs(self.b.x - self.a.x) + abs(self.b.y - self.a.y)
        self.v = cv#self.getDirection()
        self.width = math.sqrt((self.maxx - self.minx)**2 + (self.maxy - self.miny)**2)

    def getDirection(self):
        dx = self.b.x - self.a.x
        dy = self.b.y - self.a.y
        if dx > 0:
            dx = 1
        elif dx < 0:
            dx = -1
        if dy > 0:
            dy = 1
        elif dy < 0:
            dy = -1
        dz = 0
        return [dx, dy, dz]


class ArtWall(object):
    def __init__(self, a, b, cv):
        self.a = a

        self.b = b

        self.minx = min(self.a.x, self.b.x)
        self.miny = min(self.a.y, self.b.y)
        self.minz = min(self.a.z, self.b.z)
        self.maxx = max(self.a.x, self.b.x)
        self.maxy = max(self.a.y, self.b.y)
        self.maxz = max(self.a.z, self.b.z)
        self.connectionV = cv
        self.middle = Point((self.a.x + self.b.x)/2, (self.a.y + self.b.y)/2, (self.a.z + self.b.z)/2)
        self.freeSpace = abs(self.b.x - self.a.x) + abs(self.b.y - self.a.y)
        self.v = self.getDirection()
        self.width = math.sqrt((self.maxx - self.minx)**2 + (self.maxy - self.miny)**2)

    def getDirection(self):
        dx = self.b.x - self.a.x
        dy = self.b.y - self.a.y
        if dx > 0:
            dx = -1
        elif dx < 0:
            dx = 1
        if dy > 0:
            dy = -1
        elif dy < 0:
            dy = 1
        dz = 0
        return [dx, dy, dz]


class Box(object):
    def __init__(self, o, width, depth, height, v):
        self.width = width
        self.depth = depth
        self.height = height
        #origin
        self.o = o

        #opposite
        self.p = Point(self.o.x + (width * v[0]) - (depth * v[1]),
                       self.o.y + (depth * v[0]) + (width * v[1]),
                       self.o.z + height)

        #min x,y,z
        self.a = Point(min(self.o.x, self.p.x), min(self.o.y, self.p.y), min(self.o.z, self.p.z))

        #max x,y,z
        self.b = Point(max(self.o.x, self.p.x), max(self.o.y, self.p.y), max(self.o.z, self.p.z))
        self.v = v


    def move(self, v):
        self.o = Point(self.o.x + v[0], self.o.y + v[1], self.o.z + v[2])
        self.p = Point(self.p.x + v[0], self.p.y + v[1], self.p.z + v[2])
        self.a = Point(self.a.x + v[0], self.a.y + v[1], self.a.z + v[2])
        self.b = Point(self.b.x + v[0], self.b.y + v[1], self.b.z + v[2])

    def getAxis(self):
        a = Line(self.o, Point(abs(self.v[0]) * self.o.x + abs(self.v[1]) * self.p.x, abs(self.v[1]) * self.o.y + abs(self.v[0]) * self.p.y, self.o.z))
        b = Line(self.o, Point(abs(self.v[1]) * self.o.x + abs(self.v[0]) * self.p.x, abs(self.v[0]) * self.o.y + abs(self.v[1]) * self.p.y, self.o.z))
        return [a,b]

    def getWalls(self):
        left = Wall(Point(self.o.x, self.o.y, self.o.z),
                    Point(abs(self.v[0]) * self.o.x + abs(self.v[1]) * self.p.x, abs(self.v[1]) * self.o.y + abs(self.v[0]) * self.p.y, self.o.z),
                (-1 * self.v[1], self.v[0], self.v[2]))
        opposite = Wall(Point(abs(self.v[0]) * self.o.x + abs(self.v[1]) * self.p.x, abs(self.v[1]) * self.o.y + abs(self.v[0]) * self.p.y, self.o.z),
                Point(self.p.x, self.p.y, self.p.z),
                (self.v[0]+0, 1 * self.v[1], self.v[2]+0))
        right = Wall(Point(self.p.x, self.p.y, self.o.z),
                    Point(abs(self.v[1]) * self.o.x + abs(self.v[0]) * self.p.x, abs(self.v[0]) * self.o.y + abs(self.v[1]) * self.p.y, self.o.z),
                (self.v[1], -1 * self.v[0], self.v[2]))
        bottom = Wall(Point(abs(self.v[1]) * self.o.x + abs(self.v[0]) * self.p.x, abs(self.v[0]) * self.o.y + abs(self.v[1]) * self.p.y, self.o.z),
                Point(self.o.x, self.o.y, self.p.z),
                (-1 * self.v[0], -1 * self.v[1], self.v[2]))
        return([right, opposite, left, bottom])

    def getArtWalls(self):
        left = ArtWall(Point(self.a.x, self.a.y, self.a.z), Point(self.a.x, self.b.y, self.b.z),
                (-1 * self.v[1], self.v[0], self.v[2]))
        opposite = ArtWall(Point(self.a.x, self.b.y, self.a.z), Point(self.b.x, self.b.y, self.b.z),
                (self.v[0]+0, self.v[1]+0, self.v[2]+0))
        right = ArtWall(Point(self.b.x, self.b.y, self.a.z), Point(self.b.x, self.a.y, self.b.z),
                (self.v[1], -1 * self.v[0], self.v[2]))
        bottom = ArtWall(Point(self.b.x, self.a.y, self.a.z), Point(self.a.x, self.a.y, self.b.z),
                (-1 * self.v[0], -1 * self.v[1], self.v[2]))
        return([left, right, opposite, bottom])


    def collides(self, box):
        x_max = self.b.x
        x_min = self.a.x
        y_max = self.b.y
        y_min = self.a.y
        z_max = self.b.z
        z_min = self.a.z

        x_max2 = box.b.x
        x_min2 = box.a.x
        y_max2 = box.b.y
        y_min2 = box.a.y
        z_max2 = box.b.z
        z_min2 = box.a.z

        isColliding = (x_max > x_min2 and x_max2 > x_min) and \
                      (y_max > y_min2 and y_max2 > y_min) and \
                      (z_max > z_min2 and z_max2 > z_min)

        """isColliding = ((x_max >= x_min2 and x_max <= x_max2) \
                    or (x_min <= x_max2 and x_min >= x_min2)) \
                    and ((y_max >= y_min2 and y_max <= y_max2) \
                    or (y_min <= y_max2 and y_min >= y_min2)) \
                    and ((z_max >= z_min2 and z_max <= z_max2) \
                    or (z_min <= z_max2 and z_min >= z_min2))"""
        return isColliding


class ArtBox(object):
    def __init__(self, o, width, depth, height, v):
        #origin
        self.o = o

        #frameOrigin
        self.fo = Point(self.o.x  - (depth * v[1]),
                       self.o.y + (depth * v[0]),
                       self.o.z)


        #opposite
        self.p = Point(self.o.x + (width * v[0]) - (depth * v[1] - 2),
                       self.o.y + (depth * v[0] - 2) + (width * v[1]),
                       self.o.z + height)

        #min x,y,z
        self.a = Point(min(self.o.x, self.p.x), min(self.o.y, self.p.y), min(self.o.z, self.p.z))

        #max x,y,z
        self.b = Point(max(self.o.x, self.p.x), max(self.o.y, self.p.y), max(self.o.z, self.p.z))
        self.v = v


    def move(self, v):
        self.o = Point(self.o.x + v[0], self.o.y + v[1], self.o.z + v[2])
        self.fo = Point(self.fo.x + v[0], self.fo.y + v[1], self.fo.z + v[2])
        self.p = Point(self.p.x + v[0], self.p.y + v[1], self.p.z + v[2])
        self.a = Point(self.a.x + v[0], self.a.y + v[1], self.a.z + v[2])
        self.b = Point(self.b.x + v[0], self.b.y + v[1], self.b.z + v[2])

    def getWalls(self):
        left = Wall(Point(self.o.x, self.o.y, self.o.z), Point(self.o.x, self.p.y, self.p.z),
                (-self.v[1], self.v[0], self.v[2]))
        opposite = Wall(Point(self.o.x, self.p.y, self.o.z), Point(self.p.x, self.p.y, self.p.z),
                self.v)
        right = Wall(Point(self.p.x, self.p.y, self.o.z), Point(self.p.x, self.o.y, self.p.z),
                (self.v[1], -self.v[0], self.v[2]))
        bottom = Wall(Point(self.p.x, self.o.y, self.o.z), Point(self.o.x, self.o.y, self.p.z),
                (self.v[0] * (-1), self.v[1] * (-1), self.v[2]))
        return([right, left, bottom, opposite])

    def collides(self, box):
        x_max = self.b.x
        x_min = self.a.x
        y_max = self.b.y
        y_min = self.a.y
        z_max = self.b.z
        z_min = self.a.z

        x_max2 = box.b.x
        x_min2 = box.a.x
        y_max2 = box.b.y
        y_min2 = box.a.y
        z_max2 = box.b.z
        z_min2 = box.a.z

        isColliding = (x_max >= x_min2 and x_max2 >= x_min) and \
                      (y_max >= y_min2 and y_max2 >= y_min) and \
                      (z_max >= z_min2 and z_max2 >= z_min)

        """isColliding = ((x_max >= x_min2 and x_max <= x_max2) \
                    or (x_min <= x_max2 and x_min >= x_min2)) \
                    and ((y_max >= y_min2 and y_max <= y_max2) \
                    or (y_min <= y_max2 and y_min >= y_min2)) \
                    and ((z_max >= z_min2 and z_max <= z_max2) \
                    or (z_min <= z_max2 and z_min >= z_min2))"""
        return isColliding
