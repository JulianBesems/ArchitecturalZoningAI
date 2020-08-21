import time, random, datetime, pygame
from threading import Thread
from graphics import Graphics
from placer import Placer
from nn_viz import NeuralNetworkViz

NEW = True
SHOW =  not NEW
NR_OLD_ONES = 10
from_nr = 350

class Main:
    def __init__(self):
        self.time = datetime.datetime.now()

    def run(self):
        placer = Placer(NEW, NR_OLD_ONES, from_nr)
        if SHOW:
            graphics = Graphics(placer)
            graphics.display()

            while True:
                pass

        else:
            placer.run()

main = Main()
main.run()
