import numpy as np
from typing import Tuple, Optional, Union, Set, Dict, Any, List
from fractions import Fraction
import random
from collections import deque
import sys
import os
import json
import itertools
import copy, pickle, math
import PIL
from PIL import Image
from copy import copy

from misc import *
from genetic_algorithm.individual import Individual
from settings import settings
from neural_network import FeedForwardNetwork, linear, sigmoid, tanh, relu, leaky_relu, ActivationFunction, get_activation_by_name

class Block:
    def __init__(self, origin: Point, width: int, length: int, direction: Tuple[int, int], height: Optional[int] = 2400):
        self.origin = origin
        self.width = width
        self.length = length
        self.direction = direction
        fv = Point(self.direction[0], self.direction[1], 0).rotate(Point(0,0,0), math.radians(90))
        self.forwardDir = [fv[0], fv[1], self.direction[2]]
        self.height = height
        self.box = Box(self.origin, self.width, self.length, self.height, self.direction)
        self.endCentre = Point(int((abs(self.box.v[0]) * self.box.o.x + abs(self.box.v[1]) * self.box.p.x) + self.box.p.x / 2),
                        int((abs(self.box.v[1]) * self.box.o.y + abs(self.box.v[0]) * self.box.p.y)+ self.box.p.y/2), self.box.p.z)
        nrViews = settings['vision_type']
        viewRange = settings['vision_range']
        self.views = self.getViews(nrViews, viewRange)

    def getViews(self, nr: int, length: int):
        o = self.endCentre
        v = self.forwardDir
        bv = Point(self.direciton[0], self.direciton[1], self.direciton[2])
        straight = copy(o)
        straight.move([v[0]*length, v[1]*length, v[2]*length])
        range = 360 / nr
        views = []
        for i in range(nr):
            views.append(View(o, straight, range, bv, self.box))
            straight.rotate(o, math.radians(range))
            bv.rotate(Point(0,0,0), math.radians(range))

    def checkViews(self, views):
        a,b = self.box.getAxis()
        dist = settings['vision_range']
        for v in views:
            as1 = a.intersects(v.la)
            as2 = a.intersects(v.lb)
            bs1 = b.intersects(v.la)
            bs2 = b.intersects(v.lb)
            if as1:
                dist = min(dist, math.sqrt(as1[0]**2 + as1[1]**2))
            if as2:
                dist = min(dist, math.sqrt(as2[0]**2 + as2[1]**2))
            if bs1:
                dist = min(dist, math.sqrt(bs1[0]**2 + bs1[1]**2))
            if bs2:
                dist = min(dist, math.sqrt(bs2[0]**2 + bs2[1]**2))
            if not (as1 or as2 or bs1 or bs2):
                rm = self.getMiddle()
                rvl = Vector(v.origin, rm)
                rv = copy(rvl).unionise()
                if ((v.va.x < rv.x < v.vb.x or v.vb.x < rv.x < v.va.x) and
                    (v.va.y < rv.y < v.vb.y or v.vb.y < rv.y < v.va.y)):
                    dist = min(dist, rvl.length)
            if dist < v.pathDist:
                v.pathDist = dist

class View:
    def __init__(self, origin: Point, s: Point, range: float, boxV: Point, box: Box):
        self.origin = origin
        self.s = s
        self.boxV = boxV
        self.range = range
        self.a = self.s.rotate(self.origin, math.radians(self.range/2))
        self.b = self.s.rotate(self.origin, math.radians(-self.range/2))
        self.la = Line(self.origin, self.a)
        self.lb = Line(self.origin, self.b)
        self.va = Vector(self.origin, self.a).unionise()
        self.vb = Vector(self.origin, self.b).unionise()
        bO = copy(self.origin).move([-self.boxV.x * box.width/2, -self.boxV.y * box.depth/2, 0])
        self.testBox = Box(bO, box.width, box.depth, box.height, [self.boxV[0],self.boxV[1], self.boxV[2]])
        self.roomDist = settings['vision_range']
        self.artDist = settings['vision_range']
        self.pathDist = settings['vision_range']
        self.inRoom = 0
        self.closestRooms = []


class Art:
    def __init__(self,
                nr: Optional[int] = None,
                colour: Optional[Tuple[int, int, int]] = [0,0,0],
                x: Optional[int] = None,
                y: Optional[int] = None,
                z: Optional[int] = None,
                size: Optional[Tuple[int,int, int]] = [1000,10,1000]):
        self.nr = nr
        self.x = x
        self.y = y
        self.z = z
        self.colour = colour
        self.size = size
        self.width = size[0]
        self.depth = size[0]
        self.height = size[0]
        self.box = None
        self.occupiedArea = None
        self.blockedArea = None
        self.blockedWall = None

    def getOccupiedBox(self, o, v, width = None, depth = None):
        if width and depth:
            box = ArtBox(o, width, depth, self.height, v)
        else:
            box = ArtBox(o, self.width, self.depth, self.height, v)
        return box

    def getBlockedBox(self, obox, v, width = None, depth = None):
        o = obox.fo
        p = Point(o.x, o.y, min(o.z, o.z - 1500 + self.height/2))
        box = ArtBox(p, self.width, 800, max(self.height, 1500 + self.height/2), v)
        return box

    def checkViews(self, views):
        a,b = self.box.getAxis()
        dist = settings['vision_range']
        for v in views:
            as1 = a.intersects(v.la)
            as2 = a.intersects(v.lb)
            bs1 = b.intersects(v.la)
            bs2 = b.intersects(v.lb)
            if as1:
                dist = min(dist, math.sqrt(as1[0]**2 + as1[1]**2))
            if as2:
                dist = min(dist, math.sqrt(as2[0]**2 + as2[1]**2))
            if bs1:
                dist = min(dist, math.sqrt(bs1[0]**2 + bs1[1]**2))
            if bs2:
                dist = min(dist, math.sqrt(bs2[0]**2 + bs2[1]**2))
            if not (as1 or as2 or bs1 or bs2):
                rm = self.getMiddle()
                rvl = Vector(v.origin, rm)
                rv = copy(rvl).unionise()
                if ((v.va.x < rv.x < v.vb.x or v.vb.x < rv.x < v.va.x) and
                    (v.va.y < rv.y < v.vb.y or v.vb.y < rv.y < v.va.y)):
                    dist = min(dist, rvl.length)
            if dist < v.artDist:
                v.artDist = dist

class Room:
    def __init__(self, nr: int, board,
                art: Optional[List[Art]] = [],
                origin: Optional[Point] = Point(0,0,0),
                size: Optional[Tuple[int, int, int]] = [[],[],[]],
                parent: Optional = None,
                children: Optional = None,
                density: Optional[float] = 1.75,
                orientation: Optional[Tuple[int, int, int]] = (1,0,0)):
        self.nr = nr
        self.art = art
        self.placedArt = []
        self.origin = origin
        self.box = None
        self.artBox = None
        self.size = size
        self.parent = parent
        self.children = children
        self.density = density
        self.orientation = orientation
        placed = False

    def checkViews(self, views):
        a,b = self.box.getAxis()
        dist = settings['vision_range']
        for v in views:
            if v.testBox.collides(self.box):
                v.inRoom = 1
                v.closestRooms.append(self)
                return True
            as1 = a.intersects(v.la)
            as2 = a.intersects(v.lb)
            bs1 = b.intersects(v.la)
            bs2 = b.intersects(v.lb)
            if as1:
                dist = min(dist, math.sqrt(as1[0]**2 + as1[1]**2))
            if as2:
                dist = min(dist, math.sqrt(as2[0]**2 + as2[1]**2))
            if bs1:
                dist = min(dist, math.sqrt(bs1[0]**2 + bs1[1]**2))
            if bs2:
                dist = min(dist, math.sqrt(bs2[0]**2 + bs2[1]**2))
            if not (as1 or as2 or bs1 or bs2):
                rm = self.getMiddle()
                ro = self.box.o
                rp = self.box.p
                rvl = Vector(v.origin, rm)
                rv = copy(rvl).unionise()
                if ((v.va.x < rv.x < v.vb.x or v.vb.x < rv.x < v.va.x) and
                    (v.va.y < rv.y < v.vb.y or v.vb.y < rv.y < v.va.y)):
                    dist = min(dist, rvl.length, Vector(v.origin, ro).length, Vector(v.origin, rp).length )
            if dist < v.roomDist:
                v.roomDist = dist
                v.closestRooms.append(self)


    def getMiddle(self):
        m = Point(self.box.a.x + int(self.size[0]/2),
                            self.box.a.y + int(self.size[0]/2),
                            self.box.a.z + int(self.size[0]/2))
        return m

    def getSize(self):
        totWidth = 0
        minX = 1200
        minY = 1200
        minZ = 2400
        objectFloorSpace = 0
        for art in self.art:
            totWidth += art.size[0] + 250

            requiredHeight = max(art.size[2]/2 + 1600 + 200, art.size[2] + 400)
            minZ = max(minZ, requiredHeight)

            requiredDepth = max(art.size[1] + 800, art.size[1] + art.size[2])
            minX = max(minX, requiredDepth)

            minY = max(minY, art.size[0] + 500)


        z = minZ

        totArea = ((totWidth * self.density)/4)**2

        ratio = random.randint(1,5)/random.randint(1,5) + 1

        goldenRatio = 1.61803398875

        XtoY = 1 + ratio * (goldenRatio - 1)

        x = math.sqrt(totArea/XtoY)
        y = XtoY*x

        if x < minX:
            x = minX
        if y < minY:
            y = minY

        self.size[0] = int(min(x,y)) + 300
        self.size[1] = int(max(x,y)) + 300
        self.size[2] = int(z)

    def makeArtBox(self):
        minx = min(self.box.o.x, self.box.p.x)
        maxx = max(self.box.o.x, self.box.p.x)
        miny = min(self.box.o.y, self.box.p.y)
        maxy = max(self.box.o.y, self.box.p.y)
        o = Point(minx + 150, miny + 150, self.box.a.z)
        width = maxx - minx - 300
        depth = maxy - miny - 300
        height = self.size[2]
        v = (1,0,0)
        box = Box(o, width, depth, height, v)
        self.artBox = box

class Board:
    def __init__(self, nrRooms: int, nrArt: int):
        self.nrRooms = nrRooms
        self.nrArt = nrArt
        self.Art = []
        self.Rooms = self.createRooms()
        self.placedRooms = []
        self.doors = []
        self.placeRooms()
        minx = 0
        maxx = 0
        miny = 0
        maxy = 0
        minz = 0
        maxz = 0
        ri = 0
        for r in self.placedRooms:
            print(ri)
            self.placeArtCollection(r)
            minx = min(minx, r.box.a.x)
            maxx = max(maxx, r.box.b.x)
            miny = min(miny, r.box.a.y)
            maxy = max(maxy, r.box.b.y)
            minz = min(minz, r.box.a.z)
            maxz = max(maxz, r.box.b.z)
            ri +=1
        self.start = self.placedRooms[0].getMiddle()


    def createRooms(self):
        rooms = []
        for i in range(self.nrRooms):
            room = Room(nr = i, board = self)
            artSize = (random.randint(200, 10000), random.randint(10, 100), random.randint(200, 5000))
            art = Art(nr = i, size = artSize)
            self.Art.append(art)
            room.art.append(art)
            rooms.append(room)
        for _ in range(self.nrArt - self.nrRooms):
            i = random.randint(0, self.nrRooms-1)
            artSize = (random.randint(200, 1000), random.randint(10, 100), random.randint(200, 1000))
            art = Art(nr = i, size = artSize)
            self.Art.append(art)
            rooms[i].art.append(art)
        for r in rooms:
            r.getSize()
        return rooms

    def placeRooms(self):
        self.placeRoom(self.Rooms[0], None)
        while len(self.placedRooms) < len(self.Rooms):
            i = random.randint(0, len(self.placedRooms) -1)
            self.placeRoom(self.Rooms[len(self.placedRooms)], self.placedRooms[i])

    def placeRoom(self, room, parent):
        if not parent:
            room.box = Box(room.origin, room.size[0], room.size[1], room.size[2], room.orientation)
            room.placed = True
            room.makeArtBox()
            self.placedRooms.append(room)
        else:
            previous = parent
            previousWalls = previous.box.getWalls()
            random.shuffle(previousWalls)
            found = False
            for w in previousWalls:
                if self.checkWallPlace(room, w):
                    found = True
                    break

    def checkCollisions(self, box, boxes):
        for b in boxes:
            if box.collides(b.artBox):
                return True
        return False

    def checkWallPlace(self, room, w):
        wall = copy(w)
        pa = Point(-1 * wall.v[0] * (room.size[0] - 1200) + wall.a.x, -1 * wall.v[1] * (room.size[0] - 1200) + wall.a.y, wall.a.z)
        pb = Point(1 * wall.v[0] * (wall.width - 1200) + wall.a.x, 1 * wall.v[1] * (wall.width - 1200) + wall.a.y, wall.a.z)

        maxRange = [pa, pb]

        bigWall = Wall(pa, pb, wall.v)


        minX = min(maxRange[0].x, maxRange[1].x)
        maxX = max(maxRange[0].x, maxRange[1].x)
        minY = min(maxRange[0].y, maxRange[1].y)
        maxY = max(maxRange[0].y, maxRange[1].y)

        origin = Point(random.randint(minX, maxX),
                        random.randint(minY, maxY), max(0, maxRange[0].z - int(room.size[2])))

        box1 = Box(origin, room.size[0], room.size[1], room.size[2], copy(wall.v))
        box2 = copy(box1)

        found = False
        while ((not found) and ((minX <= box1.o.x <= maxX) and (minY <= box1.o.y <= maxY))
                and ((minX <= box2.o.x <= maxX) and (minY <= box2.o.y <= maxY))):
            if not self.checkCollisions(box1, self.placedRooms):
                wall.freeSpace -= 1000 * 2200
                #box1.move((-1 * wall.v[1] * 200, wall.v[0] * 200, 0))
                room.box = box1
                room.makeArtBox()
                room.placed = True
                self.placedRooms.append(room)
                #print(room.box.v)
                found = True
            elif not self.checkCollisions(box2, self.placedRooms):
                wall.freeSpace -= 1000 * 2200
                #box2.move((-1 * wall.v[1] * 200, wall.v[0] * 200, 0))
                room.box = box2
                room.makeArtBox()
                room.placed = True
                self.placedRooms.append(room)
                #print(room.box.v)
                found = True
            if not found:
                box1.move([-1 * wall.v[0] * 500,  -1 * wall.v[1] * 500, 0])
                box2.move([wall.v[0] * 500, wall.v[1] * 500, 0])

        if found:
            self.placeDoor(wall, room)
            return True
        else:
            return False

    def placeDoor(self, w, r):
        wall = copy(w)
        room = copy(r)
        try:
            wminx = min(wall.a.x, wall.b.x)
            wmaxx = max(wall.a.x, wall.b.x)
            wminy = min(wall.a.y, wall.b.y)
            wmaxy = max(wall.a.y, wall.b.y)

            minX = int(max(min(room.box.o.x, room.box.p.x), wminx))+450
            maxX = int(min(max(room.box.o.x, room.box.p.x), wmaxx))-450
            minY = int(max(min(room.box.o.y, room.box.p.y), wminy))+450
            maxY = int(min(max(room.box.o.y, room.box.p.y), wmaxy))-450
            if maxX-minX > maxY - minY:
                x = random.randint(minX, maxX)
                y = wall.miny
            else:
                y = random.randint(minY, maxY)
                x = wall.minx
            origin = Point(x, y, room.box.a.z)
            box = Box(origin, 1000, 2000, 2200, copy(wall.v))
            box.move([-1 * wall.v[0] * 450,  -1 * wall.v[1] * 450, 0])
            box.move((1 * wall.v[1] * 1000, -1 * wall.v[0] * 1000, 0))
            self.doors.append(box)
        except ValueError:
            #print("shit")
            self.doors.append(Box(Point(0,0,0), 1, 1, 1, (1,0,0)))

    def placeArtCollection(self, room):
        room.art.sort(key = lambda x : x.size[0] * x.size[1], reverse = True)

        placed = 0
        for art in room.art:
            if not self.placeArtPiece(art, room):
                pass
                #print(art.id, art.url, art.width, art.depth)
            else:
                placed +=1
                #print(placed, '/', len(ArtPieces))

    def placeArtPiece(self, art, room):
        if self.checkPlace(art, room):
            return True
        return False

    def checkPlace(self, art, room):
        walls = room.artBox.getArtWalls()
        random.shuffle(walls)
        for w in walls:
            if self.checkArtWallPlace(art, w, room):
                room.placedArt.append(art)
                return True
        return False

    def checkArtWallPlace(self, art, wall, room):
        if art.size[0] > wall.freeSpace:
            return False

        #origin = Point(wall.middle.x, wall.middle.y, wall.minz)
        if wall.minx != wall.maxx:
            x = random.randint(wall.minx+int(art.size[0]/2), wall.maxx-int(art.size[0]/2))
            y = wall.miny
        else:
            x = wall.minx
            y = random.randint(wall.miny+int(art.size[0]/2), wall.maxy-int(art.size[0]/2))

        origin = Point(x, y, wall.minz)
        box1 = art.getOccupiedBox(origin, wall.v)
        box1.move([-1 * wall.v[0] * (art.size[0]/2), -wall.v[1] * (art.size[0]/2), max(1600 - (art.size[2]/2), 200)])
        boxBlock1 = art.getBlockedBox(box1, wall.v)

        box2 = copy(box1)
        boxBlock2 = copy(boxBlock1)

        found = False
        while ((not found) and (((wall.maxx > box1.a.x > wall.minx and wall.maxx > box1.b.x > wall.minx) or
        (wall.maxy > box1.a.y > wall.miny and wall.maxy > box1.b.y > wall.miny)) or
        ((wall.maxx > box2.a.x > wall.minx and wall.maxx > box2.b.x > wall.minx) or
        (wall.maxy > box2.a.y > wall.miny and wall.maxy > box2.b.y > wall.miny)))):
            if ((not self.checkArtCollisions(box1, boxBlock1, room.placedArt)) and
                ((wall.maxx > box1.a.x > wall.minx and wall.maxx > box1.b.x > wall.minx) or
                (wall.maxy > box1.a.y > wall.miny and wall.maxy > box1.b.y > wall.miny))):
                wall.freeSpace -= art.size[0]
                art.occupiedArea = box1
                art.blockedArea = boxBlock1
                room.placedArt.append(art)
                #print(art.occupiedArea)
                found = True
            elif ((not self.checkArtCollisions(box2, boxBlock2, room.placedArt)) and
            ((wall.maxx > box2.o.x > wall.minx and wall.maxx > box2.p.x > wall.minx) or
            (wall.maxy > box2.o.y > wall.miny and wall.maxy > box2.p.y > wall.miny))):
                wall.freeSpace -= art.size[0]
                art.occupiedArea = box2
                art.blockedArea = boxBlock2
                #print(art.occupiedArea)
                room.placedArt.append(art)
                found = True
            if not found:
                box1.move([-1 * wall.v[0] * (100), -1 * wall.v[1] * (100), 0])
                boxBlock1.move([-1 * wall.v[0] * (100), -1 * wall.v[1] * (100), 0])
                box2.move([wall.v[0] * (100), wall.v[1] * (100), 0])
                boxBlock2.move([wall.v[0] * (100), wall.v[1] * (100), 0])


        if found:
            return True
        else:
            return False

    def checkArtCollisions(self, box, boxBlock, placedArt):
        for a in placedArt:
            boxA = a.occupiedArea
            boxABlock = a.blockedArea
            if box.collides(boxA) or box.collides(boxABlock) or boxBlock.collides(boxA):# or boxBlock.collides(boxABlock):
                return True
        for d in self.doors:
            if d.collides(box):
                return True
        return False
