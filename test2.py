from misc import *
import math
import numpy as np

def getCircleIntersect(c1, c2):
    d = math.sqrt(abs(c1.x-c2.x)**2 + abs(c1.y-c2.y)**2)
    if (d > c1.r + c2.r) or (d < abs(c1.r - c2.r) or (d==0 and c1.r == c2.r)):
        return None
    a = (c1.r**2 - c2.r**2 + d**2)/(2*d)

    p3x = c1.x + a*(c2.x - c1.x)/d
    p3y = c1.y + a*(c2.y - c1.y)/d

    h = math.sqrt(c1.r**2 - a**2)

    #if d == c1.r + c2.r:
    #    return (p3x, p3y)

    p4x = p3x + h*(c2.y - c1.y)/d
    p4y = p3y - h*(c2.x - c1.x)/d

    p5x = p3x - h*(c2.y - c1.y)/d
    p5y = p3y + h*(c2.x - c1.x)/d

    return [(p4x, p4y), (p5x, p5y)]

c1 = Circle(Point(0,0), 5)
c2 = Circle(Point(5,0), 1)

print(getCircleIntersect(c1, c2))
