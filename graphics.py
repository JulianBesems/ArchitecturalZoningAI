import pygame, sys, random, time, csv, ast, requests, math, pickle, operator
from io import BytesIO
from PIL import Image
from urllib.request import urlopen
import numpy as np
from colors import rgb, hex
from geopy.geocoders import Nominatim
from datetime import datetime
from colorhash import ColorHash
from shapely import geometry
from copy import copy
from placer import *
from misc import *
from exportRhinoGroupOutcome import *

pygame.font.init()
myfont = pygame.font.Font('/Users/julianbesems/Library/Fonts/HELR45W.ttf', 22)
myfontI = pygame.font.Font('/Users/julianbesems/Library/Fonts/HELR45W.ttf', 35)
myfontL = pygame.font.Font('/Users/julianbesems/Library/Fonts/HELR45W.ttf', 50)
myfontS = pygame.font.Font('/Users/julianbesems/Library/Fonts/HELR45W.ttf', 14)

bcc = 220
backgroundColour = pygame.Color(255,255,255)
textColour = pygame.Color(86,82,71)

class Node:
    def __init__(self, layer, number, pos = None):
        self.layer = layer
        self.number = number
        self.pos = pos
        self.value = 0

class Graphics:
    screen_width = 3360 #3360 #1920 #1440 #2560 #1500 #1400 #2000 #1440
    screen_height = 2100 #2100 #1080 #823 #1600 #1000 #800 #1143 #823
    screen_centre = [int(screen_width/2), int(screen_height/2)]
    buffer = int(screen_height/100)
    ps = int(buffer/10)

    frames = 1
    ShowNN = True

    NEW_Network = True

    def __init__(self, placer):
        self._screen = pygame.display.set_mode((self.screen_width, self.screen_height))#, pygame.FULLSCREEN)
        self.grid_surface = pygame.surface.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA, 32)
        self.dot_surface = pygame.surface.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA, 32)
        self.connection_surface = pygame.surface.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA, 32)
        self.nn_surface = pygame.surface.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA, 32)
        self.placer = placer
        self.puzzle = self.placer.puzzle
        self.nn = self.puzzle.network

    def draw_screen(self, screen):
        pygame.init()
        pygame.display.set_caption('proximity AI')

    def restart(self):
        self._screen.fill(backgroundColour)
        self._screen.blit(self.grid_surface, [0,0])

    def draw_dot(self, dot, c = None, surface = None, radius = None):
        if radius:
            r = radius
        else:
            r = max(4*self.ps, 10)
        centre = self.map_coordinates(dot)
        br = self.map_coordinates(Point(dot.x +1, dot.y + 1))
        w = centre[0] - br[0]
        h = centre[1] - br[1]
        if c:
            color = c
        else:
            color = dot.colour
        if surface:
            pygame.draw.rect(surface, color, (centre[0] - int(r/2), centre [1] - int(r/2), int(r/2)*2, int(r/2)*2))
            #pygame.draw.circle(surface, color, centre, r)
        else:
            try:
                v = max(int((1 - (1-dot.value) * 2) * r), 1)
                pygame.draw.rect(self.dot_surface, color, (centre, (w,h)))
                #pygame.draw.circle(self.dot_surface, color, centre, r, v)
            except AttributeError:
                pygame.draw.rect(self.dot_surface, color, (centre, (w,h)))
                #pygame.draw.circle(self.dot_surface, color, centre, r)

    def map_coordinates(self, p):
        m = min((self.screen_width-4*self.buffer)/self.puzzle.board_size[0], (self.screen_height-4*self.buffer)/self.puzzle.board_size[1])
        x = int(p.x * m + 4*self.buffer)
        y = int(p.y * m + 4*self.buffer)
        return (x,y)

    def draw_grid(self):
        for i in range(self.puzzle.board_size[0]):
            for j in range(self.puzzle.board_size[1]):
                self.draw_dot(Point(i,j), c = textColour, surface = self.grid_surface, radius = 1)

    def draw_nn(self):
        self.puzzle = self.placer.puzzle
        self.nn = self.puzzle.network
        r = int(self.buffer * 0.6)
        inbetweenX =  min(int((self.screen_width - 10 * self.buffer)/len(self.nn.layer_nodes)), 12 * self.buffer)
        inbetweenY = min(5 * self.buffer, int((self.screen_height - 10 * self.buffer)/max(self.nn.layer_nodes)))
        #Centric NN
        #startX = self.screen_centre[0] - int((len(self.nn.layer_nodes)/2) * inbetweenX)
        #FromRight NN
        startX = self.screen_width - int((len(self.nn.layer_nodes)) * inbetweenX)

        nodes = []
        locations = []
        for i in range(len(self.nn.layer_nodes)):
            layer = []
            startY = self.screen_centre[1] - int((self.nn.layer_nodes[i]/2) * inbetweenY)
            for j in range(self.nn.layer_nodes[i]):
                pos = [startX + i * inbetweenX, startY + j * inbetweenY]
                nodes.append(Node(i, j, pos))
                layer.append([startX + i * inbetweenX, startY + j * inbetweenY])
            locations.append(layer)

        if self.NEW_Network:
            for l in range(1, len(self.nn.layer_nodes)):
                weights = self.nn.params['W' + str(l)]
                prev_nodes = weights.shape[1]
                curr_nodes = weights.shape[0]
                # For each node from the previous layer
                for prev_node in range(prev_nodes):
                    # For all current nodes, check to see what the weights are
                    for curr_node in range(curr_nodes):
                        # If there is a positive weight, make the line blue
                        if weights[curr_node, prev_node] > 0:
                            draw = True
                            c = min(weights[curr_node, prev_node],1) * 200
                            colour = [200, 200-c, 200-c]
                        # If there is a negative (impeding) weight, make the line red
                        else:
                            draw = True
                            c = abs(max(weights[curr_node, prev_node],-1)) * 200
                            colour = [200-c, 200-c, 200]
                        if draw:
                            # Grab locations of the nodes
                            start = locations[l-1][prev_node]
                            end = locations[l][curr_node]
                            # Offset start[0] by diameter of circle so that the line starts on the right of the circle
                            pygame.draw.line(self.nn_surface, colour, start, end, max(self.ps, 1))
            self.NEW_Network = False

        inputs = self.puzzle.input_values_as_array

        activations = []
        for i in range(1, len(self.nn.layer_nodes)-1):
            try:
                activations.append(self.nn.params['A' + str(i)])
            except KeyError:
                activations.append(0)


        for n in nodes:
            # Inputs
            if n.layer == 0:
                n.value = inputs[n.number][0]

            # Hidden Layers
            if n.layer > 0 and n.layer < len(self.nn.layer_nodes)-1:
                try:
                    n.value = activations[n.layer - 1][n.number][0]
                except TypeError:
                    pass

            # Output Layer
            if n.layer == len(self.nn.layer_nodes)-1:
                try:
                    n.value = self.nn.out[n.number][0]
                    outputText = myfont.render(str(round(n.value, 8)), False, textColour)
                    pygame.draw.rect(self.nn_surface, backgroundColour, pygame.Rect(n.pos[0] + 2*self.buffer,n.pos[1] - 3*self.ps,200,50))
                    self.nn_surface.blit(outputText, (n.pos[0] + 2*self.buffer, n.pos[1] - 3*self.ps))
                except TypeError:
                    pass

            if n.value>0.5:
                v = int((abs(min(n.value,1)-1) * 200))
                pygame.draw.circle(self.nn_surface, backgroundColour, n.pos, r +1)
                pygame.draw.line(self.nn_surface, [v,v,v], (n.pos[0]-r, n.pos[1]), (n.pos[0]+r, n.pos[1]), int(r/2))
                pygame.draw.line(self.nn_surface, [v,v,v], (n.pos[0], n.pos[1]-r), (n.pos[0], n.pos[1]+r), int(r/2))
            else:
                v = int(max(n.value,0) * 200)
                pygame.draw.circle(self.nn_surface, backgroundColour, n.pos, r+1)
                pygame.draw.circle(self.nn_surface, [v,v,v], n.pos, r, int(r/2))


    def display(self):
        # Setup pygame screen
        clock = pygame.time.Clock()
        SCREENSHOT_NR = 0
        groupNr = 0
        self._screen.fill(backgroundColour)
        self.draw_screen(self._screen)
        #self.draw_grid()
        pygame.display.update()

        # Animation loop
        while True:
            events = pygame.event.get()
            for event in events:
                # Check exit
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

                    if event.key == pygame.K_n:
                        if self.ShowNN:
                            self._screen.fill(backgroundColour)
                            self.draw_grid()
                            self.ShowNN = False
                        else:
                            self.NEW_Network = True
                            self.ShowNN = True

            self._screen.blit(self.grid_surface, [0,0])

            for g in self.puzzle.groups:
                self.draw_dot(self.puzzle.groups[g].point, c = self.puzzle.groups[g].colour, radius = 15)

            """pygame.image.save(self._screen, "screenshotsTests/screenshot" + str(SCREENSHOT_NR) + ".jpeg")
            SCREENSHOT_NR +=1"""


            d = self.placer.update()
            """
            dots = []
            for _ in range(self.frames):
                d = self.placer.update()
                dots.append(d)"""


            if d == -1:
                g = self.puzzle.groups[d.nr]
                for kz in g.zones:
                    z = g.zones[kz]
                    keys = z.keys()
                    if kz == (0,1):
                        col = (255,0,0)
                    elif kz == (0,-1):
                        col = (255,255,0)
                    elif kz == (1,1):
                        col = (0,255,255)
                    elif kz == (1,-1):
                        col = (255,0,255)
                    elif kz == (1,0):
                        col = (0,255,0)
                    elif kz == (-1,1):
                        col = (0,0,255)
                    elif kz == (-1,-1):
                        col = (0,0,0)
                    elif kz == (-1,0):
                        col = (255,255,255)

                    for k in keys:
                        for cell in z[k]:
                            if True: #k < 3:
                                self.draw_dot(Point(cell[0][0], cell[0][1]), c = col, radius = 10)
                            #self.draw_dot(Point(cell[0][0], cell[0][1]), c = (255 - 10*k, 255 - 10*k, 255 - 10*k), radius = 10)

            for c in self.puzzle.board.cells:
                for b in c:
                    if b:
                        self.draw_dot(b)

            if d:
                pass
                #self.draw_dot(d)
            else:
                #pygame.image.save(self._screen, "screenshotsTests3/screenshot7" + str(SCREENSHOT_NR) + ".jpeg")
                #e = Exporter(self.puzzle, [1000,1000,1000], SCREENSHOT_NR)
                #e.export()
                SCREENSHOT_NR +=1

                self.restart()
                self.dot_surface.fill([0,0,0,0])
                self.NEW_Network = True

            self._screen.blit(self.dot_surface, [0,0])

            if self.ShowNN:
                self.draw_nn()
                self._screen.blit(self.nn_surface, [0,0])

            pygame.display.update()
            clock.tick()
