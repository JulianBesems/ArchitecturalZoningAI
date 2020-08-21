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


from misc import *
from genetic_algorithm.individual import Individual
from settings import settings
from neural_network import FeedForwardNetwork, linear, sigmoid, tanh, relu, leaky_relu, ActivationFunction, get_activation_by_name
from pieces import *

class Path(Individual):
    def __init__(self,
                 nrRooms: int,
                 nrArt: int,
                 chromosome: Optional[Dict[str, List[np.ndarray]]] = None,
                 hidden_layer_architecture: Optional[List[int]] = [24, 18],
                 hidden_activation: Optional[ActivationFunction] = 'relu',
                 output_activation: Optional[ActivationFunction] = 'sigmoid',
                 groups: Optional = None,
                 cMatrix: Optional[List[List[float]]] = None,
                 values: Optional[List[float]] = None,
                 sizes: Optional[List[int]] = None
                 ):

        self.nrRooms = nrRooms
        self.nrArt = nrArt

        self.failed = False
        self.settings = settings

        self.progress = 0
        self._fitness = 0
        self.board = Board(self.nrRooms, self.nrArt)
        self.start = self.board.start
        self.blocks = []
        self.cost = 0

        self.hidden_layer_architecture = hidden_layer_architecture
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        # Setting up network architecture
        # Each "Vision" has 3 distances it tracks: wall, apple and self
        # there are also one-hot encoded direction and one-hot encoded tail direction,
        # each of which have 4 possibilities.
        num_inputs = 36 #@TODO: Add one-hot back in
        self.input_values_as_array: np.ndarray = np.zeros((num_inputs, 1))
        self.network_architecture = [num_inputs]                          # Inputs
        self.network_architecture.extend(self.hidden_layer_architecture)  # Hidden layers
        self.network_architecture.append(12)                               # 4 outputs, ['u', 'd', 'l', 'r']
        self.network = FeedForwardNetwork(self.network_architecture,
                                          get_activation_by_name(self.hidden_activation),
                                          get_activation_by_name(self.output_activation))

        if chromosome:
            self.network.params = chromosome
        else:
            pass

        self.target_pos = self.board.placedRooms[1]


    @property
    def fitness(self):
        return self._fitness

    @property
    def finished(self):
        if self.progress < len(self.board.placedRooms):
            return False
        return True

    def calculate_fitness(self):
        #self._fitness = (self.progress/self.nrBlocks) * 100
        fitnessList = []

        for i in self.finishedGroups:
            groupFitness = 0

            g1 = self.finishedGroups[i]
            nrBlocksTotal = len(g1.blocksPlaced)
            vals = self.cMatrix[i]

            if not g1.blocksPlaced:
                fitnessList.append(0)
            else:
                targetGroups = []
                for j in range(len(vals)):
                    if (not i == j) and vals[j] > 0.5:
                        targetGroups.append((j, vals[j]))

                for gti in targetGroups:
                    gt = self.finishedGroups[gti[0]]
                    smallestDist = self.board.width + self.board.height
                    nearestBlock = None

                    for b in g1.blocksPlaced:
                        d = (abs(b.x - gt.point.x) + abs(b.y - gt.point.y))
                        if d < smallestDist:
                            smallestDist = d
                            nearestBlock = b

                    smallestDistBlock = smallestDist
                    for b in gt.blocksPlaced:
                        d = (abs(b.x - nearestBlock.x) + abs(b.y - nearestBlock.y))
                        if d < smallestDistBlock:
                            smallestDistBlock = d
                            nearestBlock = b

                    totDist = (abs(g1.point.x - gt.point.x) + abs(g1.point.y - gt.point.y))
                    if not totDist:
                        print(g1.point, g2.point)
                        groupFitness += 0
                    else:
                        groupFitness += ((totDist - smallestDistBlock) / nrBlocksTotal) * gti[1]
                #print(gFitness)

                internalDist = 0
                for b in g1.blocksPlaced:
                    internalDist += (abs(b.x - g1.point.x) + abs(b.y - g1.point.y))

                groupFitness += max(0, (nrBlocksTotal / 4) - (internalDist/nrBlocksTotal))

                fitnessList.append(groupFitness)



        fitness = 0

        for f in fitnessList:
            fitness += f
        self._fitness = fitness
        return self._fitness

    @property
    def chromosome(self):
        # return self._chromosome
        pass

    def fill(self) -> bool:
        if self.finished or self.failed:
            return False
        i = self.progress + 1
        head = self.takeStep()
        if not head:
            self.failed = True
            return False
        if self.target_pos.collidesWithBox(head):
            self.progress += 1
        return b

    def takeStep(self):
        self.network.feed_forward(self.input_values_as_array)
        out = list(self.network.out)
        outputDir = []

        head = self.blocks[-1]
        headDir = head.direction
        hv = Point(headDir[0], headDir[1], headDir[2])
        for i in range(4):
            outputDir.append(hv)
            hv.rotate(Point(0,0,0), math.radians(90))

        dir = outputDir[np.argmax(out)]
        endCentre =  head.endCentre
        width = settings['path_width']
        length = settings['path_length']
        height = head.height

        origin = copy.copy(endCentre).move([-dir.x * width/2, -dir.y * length, 0])
        block = Block(origin, width, length, dir, height = height)
        inRoom = False

        for v in head.views:
            for b in v.closestRooms:
                for a in b.art:
                    if a.box.collides(block.box):
                        return False
                if block.box.collides(b.box):
                    inRoom = True
        if inRoom:
            self.cost += 0.5
        else:
            self.cost +=1
        self.blocks.append(block)
        return block

        if not found:
            print(e)
            print(outputDirCopy)
            print(empties)
            return None


    def look(self, block):
        array = self.input_values_as_array
        head = self.blocks[-1]
        views = head.views

        for r in self.board.placedRooms:
            r.checkViews(views)

        for b in blocks:
            b.checkViews(views)

        for v in views:
            for b in v.closestRooms:
                for a in b.art:
                    a.checkViews(views)

        i = 0
        for v in views:
            array[i*4] = 1 / ((v.roomDist + settings['pathLength'])/settings['pathLength'])
            array[i*4 + 1] = 1 / ((v.artDist + settings['pathLength'])/settings['pathLength'])
            array[i*4 + 2] = 1 / ((v.pathDist + settings['pathLength'])/settings['pathLength'])
            array[i*4 + 3] = v.inRoom

def save_path(population_folder: str, individual_name: str, path: Path, settings: Dict[str, Any]) -> None:
    # Make population folder if it doesn't exist
    if not os.path.exists(population_folder):
        os.makedirs(population_folder)

    # Save off settings
    if 'settings.json' not in os.listdir(population_folder):
        f = os.path.join(population_folder, 'settings.json')
        with open(f, 'w', encoding='utf-8') as out:
            json.dump(settings, out, sort_keys=True, indent=4)

    # Make directory for the individual
    individual_dir = os.path.join(population_folder, individual_name)
    os.makedirs(individual_dir)

    # Save some constructor information for replay
    # @NOTE: No need to save chromosome since that is saved as .npy
    # @NOTE: No need to save board_size or hidden_layer_architecture
    #        since these are taken from settings
    constructor = {}
    constructor['nrGroups'] = path.nrGroups
    constructor['nrBlocks'] = path.nrBlocks
    #constructor['groups'] = path.initialGroups
    path_constructor_file = os.path.join(individual_dir, 'constructor_params.json')

    # Save
    with open(path_constructor_file, 'w', encoding='utf-8') as out:
        json.dump(constructor, out, sort_keys=True, indent=4)

    L = len(path.network.layer_nodes)
    for l in range(1, L):
        w_name = 'W' + str(l)
        b_name = 'b' + str(l)

        weights = path.network.params[w_name]
        bias = path.network.params[b_name]

        np.save(os.path.join(individual_dir, w_name), weights)
        np.save(os.path.join(individual_dir, b_name), bias)

def load_path(population_folder: str, individual_name: str, settings: Optional[Union[Dict[str, Any], str]] = None) -> Path:
    if not settings:
        f = os.path.join(population_folder, 'settings.json')
        if not os.path.exists(f):
            raise Exception("settings needs to be passed as an argument if 'settings.json' does not exist under population folder")

        with open(f, 'r', encoding='utf-8') as fp:
            settings = json.load(fp)

    elif isinstance(settings, dict):
        settings = settings

    elif isinstance(settings, str):
        filepath = settings
        with open(filepath, 'r', encoding='utf-8') as fp:
            settings = json.load(fp)

    params = {}
    for fname in os.listdir(os.path.join(population_folder, individual_name)):
        extension = fname.rsplit('.npy', 1)
        if len(extension) == 2:
            param = extension[0]
            params[param] = np.load(os.path.join(population_folder, individual_name, fname))
        else:
            continue

    # Load constructor params for the specific snake
    constructor_params = {}
    path_constructor_file = os.path.join(population_folder, individual_name, 'constructor_params.json')
    with open(path_constructor_file, 'r', encoding='utf-8') as fp:
        constructor_params = json.load(fp)

    path = Path(settings['board_size'], settings['nrGroupRange'][0],
                  settings['nrBlocksRange'][0],
                  chromosome=params,
                  hidden_layer_architecture=settings['hidden_network_architecture'],
                  hidden_activation=settings['hidden_layer_activation'],
                  output_activation=settings['output_layer_activation'],
                  #groups = constructor_params['groups']
                  )
    return path
