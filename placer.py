import sys

from typing import List
from puzzle import *
import numpy as np

from settings import settings
from math import sqrt
from decimal import Decimal
import random, pickle
import csv

from exportRhinoGroupOutcome import *

from neural_network import FeedForwardNetwork, sigmoid, linear, relu

from genetic_algorithm.population import Population
from genetic_algorithm.selection import elitism_selection, roulette_wheel_selection, tournament_selection
from genetic_algorithm.mutation import gaussian_mutation, random_uniform_mutation
from genetic_algorithm.crossover import simulated_binary_crossover as SBX
from genetic_algorithm.crossover import uniform_binary_crossover, single_point_binary_crossover

class Placer:
    def __init__(self, NEW, old_ones, from_nr):
        self.new = NEW
        self.NR_OLD_ONES = old_ones
        self.from_nr = from_nr
        self.settings = settings

        self._SBX_eta = self.settings['SBX_eta']
        self._mutation_bins = np.cumsum([self.settings['probability_gaussian'],
                                        self.settings['probability_random_uniform']])
        self._crossover_bins = np.cumsum([self.settings['probability_SBX'],
                                         self.settings['probability_SPBX']])
        self._SPBX_type = self.settings['SPBX_type'].lower()
        self._mutation_rate = self.settings['mutation_rate']

        # Determine size of next gen based off selection type
        self._next_gen_size = None
        if self.settings['selection_type'].lower() == 'plus':
            self._next_gen_size = self.settings['num_parents'] + self.settings['num_offspring']
        elif self.settings['selection_type'].lower() == 'comma':
            self._next_gen_size = self.settings['num_offspring']
        else:
            raise Exception('Selection type "{}" is invalid'.format(self.settings['selection_type']))

        self.board_size = settings['board_size']

        self.nrRoomRange = settings['nrRoomRange']
        self.nrArtRange = (settings['nrArtRange'])

        individuals: List[Individual] = []

        if self.new:
            for _ in range(self.settings['num_parents']):
                nrR = random.randint(self.nrRoomRange[0], self.nrRoomRange[1])
                nrA = random.randint(self.nrArtRange[0], self.nrArtRange[1])
                individual = Path(nrR, nrA,
                                    hidden_layer_architecture=self.settings['hidden_network_architecture'],
                                    hidden_activation=self.settings['hidden_layer_activation'],
                                    output_activation=self.settings['output_layer_activation'])
                individuals.append(individual)
                print(len(individuals))

        else:
            for i in range(self.NR_OLD_ONES):
                individual = load_puzzle('population9', 'best_snake' + str(i + self.from_nr) , self.settings)
                individuals.append(individual)

        self.best_fitness = 0

        self._current_individual = 0
        self.population = Population(individuals)

        self.puzzle = self.population.individuals[self._current_individual]
        self.current_generation = 0

    def run(self):
        while True:
            self.update()

    def update(self) -> None:
        if not (self.puzzle.finished or self.puzzle.failed):
            b = self.puzzle.fill()
            if b:
                return self.puzzle.fill()
            else:
                return False

        # Current individual is dead:
        else:
            fitness = self.puzzle.calculate_fitness()
            print(self._current_individual, fitness)

            if fitness > self.best_fitness:
                self.best_fitness = fitness

            #e = Exporter(self.puzzle, [1000,1000,1000], self._current_individual)
            #e.export()

            #self.exportNNWeights()
            self._current_individual += 1

            if (self.current_generation > 0 and self._current_individual == self._next_gen_size) or\
                (self.current_generation == 0 and self._current_individual == settings['num_parents']):
                print(self.settings)
                print('======================= Gneration {} ======================='.format(self.current_generation))
                print('----Max fitness:', self.best_fitness)
                print('----Average fitness:', self.population.average_fitness)

                with open("preprocessedPhotos" +str(self.current_generation)+ ".p", "wb") as fp:
                    pickle.dump(self.population.fittest_individual, fp, protocol = pickle.HIGHEST_PROTOCOL)
                save_puzzle('populationNew0', 'best_snake'+str(self.current_generation), self.population.fittest_individual, self.settings)
                #self.exportNNWeights()
                self.next_generation()
            else:
                current_pop = self.settings['num_parents'] if self.current_generation == 0 else self._next_gen_size

            self.puzzle = self.population.individuals[self._current_individual]
            return False

    def exportNNWeights(self):
        nn = self.puzzle.network
        nodes = []
        connections = []
        for l in range(1, len(nn.layer_nodes)):
            layerNodes = []
            weights = nn.params['W' + str(l)]
            connections.append(weights)
            for n in range(weights.shape[1]):
                layerNodes.append(n)
            nodes.append(layerNodes)
            if l == len(nn.layer_nodes)-1:
                layerNodes = []
                for n in range(weights.shape[0]):
                    layerNodes.append(n)
                nodes.append(layerNodes)
        entries = []
        for a in nodes[3]:
            entriesLayer = []
            for _ in nodes[0]:
                entriesLayer.append([])
            for b in nodes[2]:
                for c in nodes[1]:
                    for d in nodes[0]:
                        score = connections[2][a,b] * connections[1][b,c] * connections[0][c,d]
                        entriesLayer[d].append(score)
            entries.append(entriesLayer)

        with open("nnWeights/" + str("populationNew0") + ".csv", 'a') as newcsvfile:
            writer = csv.writer(newcsvfile)
            row = []
            #print(entries[0][0])
            for e in entries:
                for c in e:
                    row.extend(c)
            #print(len(row))
            writer.writerow(row)






    def next_generation(self):
        self._increment_generation()
        self._current_individual = 0

        self.population.individuals = elitism_selection(self.population, self.settings['num_parents'])

        random.shuffle(self.population.individuals)
        next_pop: List[Path] = []

        # parents + offspring selection type ('plus')
        if self.settings['selection_type'].lower() == 'plus':

            for individual in self.population.individuals:
                params = individual.network.params
                board_size = individual.board_size
                hidden_layer_architecture = individual.hidden_layer_architecture
                hidden_activation = individual.hidden_activation
                output_activation = individual.output_activation

                # If the individual is still alive, they survive
                if individual._fitness < 1000:
                    nrG = random.randint(self.nrRoomRange[0], self.nrRoomRange[1])
                    nrC = random.randint(self.nrBlocksRange[0], self.nrBlocksRange[1])
                    s = Path(board_size, nrG, nrC, chromosome=params, hidden_layer_architecture=hidden_layer_architecture,
                            hidden_activation=hidden_activation, output_activation=output_activation)
                    next_pop.append(s)

        while len(next_pop) < self._next_gen_size:
            p1,p2 = roulette_wheel_selection(self.population, 2)

            L = len(p1.network.layer_nodes)
            c1_params = {}
            c2_params = {}

            # Each W_l and b_l are treated as their own chromosome.
            # Because of this I need to perform crossover/mutation on each chromosome between parents
            for l in range(1, L):
                p1_W_l = p1.network.params['W' + str(l)]
                p2_W_l = p2.network.params['W' + str(l)]
                p1_b_l = p1.network.params['b' + str(l)]
                p2_b_l = p2.network.params['b' + str(l)]

                # Crossover
                # @NOTE: I am choosing to perform the same type of crossover on the weights and the bias.
                c1_W_l, c2_W_l, c1_b_l, c2_b_l = self._crossover(p1_W_l, p2_W_l, p1_b_l, p2_b_l)

                # Mutation
                # @NOTE: I am choosing to perform the same type of mutation on the weights and the bias.
                self._mutation(c1_W_l, c2_W_l, c1_b_l, c2_b_l)

                # Assign children from crossover/mutation
                c1_params['W' + str(l)] = c1_W_l
                c2_params['W' + str(l)] = c2_W_l
                c1_params['b' + str(l)] = c1_b_l
                c2_params['b' + str(l)] = c2_b_l

                # Clip to [-1, 1]
                np.clip(c1_params['W' + str(l)], -1, 1, out=c1_params['W' + str(l)])
                np.clip(c2_params['W' + str(l)], -1, 1, out=c2_params['W' + str(l)])
                np.clip(c1_params['b' + str(l)], -1, 1, out=c1_params['b' + str(l)])
                np.clip(c2_params['b' + str(l)], -1, 1, out=c2_params['b' + str(l)])

            # Create children from chromosomes generated above
            nrR = random.randint(self.nrRoomRange[0], self.nrRoomRange[1])
            nrA = random.randint(self.nrArtRange[0], self.nrArtRange[1])
            c1 = Path(nrR, nrA, chromosome=c1_params, hidden_layer_architecture=p1.hidden_layer_architecture,
                       hidden_activation=p1.hidden_activation, output_activation=p1.output_activation)

            nrG = random.randint(self.nrRoomRange[0], self.nrRoomRange[1])
            nrC = random.randint(self.nrBlocksRange[0], self.nrBlocksRange[1])
            c2 = Path(nrR, nrA, chromosome=c2_params, hidden_layer_architecture=p2.hidden_layer_architecture,
                       hidden_activation=p2.hidden_activation, output_activation=p2.output_activation)


            # Add children to the next generation
            next_pop.extend([c1, c2])

        # Set the next generation
        random.shuffle(next_pop)
        self.population.individuals = next_pop

    def _increment_generation(self):
        self.current_generation += 1

    def _crossover(self, parent1_weights: np.ndarray, parent2_weights: np.ndarray,
                   parent1_bias: np.ndarray, parent2_bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        rand_crossover = random.random()
        crossover_bucket = np.digitize(rand_crossover, self._crossover_bins)
        child1_weights, child2_weights = None, None
        child1_bias, child2_bias = None, None

        # SBX
        if crossover_bucket == 0:
            child1_weights, child2_weights = SBX(parent1_weights, parent2_weights, self._SBX_eta)
            child1_bias, child2_bias =  SBX(parent1_bias, parent2_bias, self._SBX_eta)

        # Single point binary crossover (SPBX)
        elif crossover_bucket == 1:
            child1_weights, child2_weights = single_point_binary_crossover(parent1_weights, parent2_weights, major=self._SPBX_type)
            child1_bias, child2_bias =  single_point_binary_crossover(parent1_bias, parent2_bias, major=self._SPBX_type)

        else:
            raise Exception('Unable to determine valid crossover based off probabilities')

        return child1_weights, child2_weights, child1_bias, child2_bias

    def _mutation(self, child1_weights: np.ndarray, child2_weights: np.ndarray,
                  child1_bias: np.ndarray, child2_bias: np.ndarray) -> None:
        scale = .2
        rand_mutation = random.random()
        mutation_bucket = np.digitize(rand_mutation, self._mutation_bins)

        mutation_rate = self._mutation_rate
        if self.settings['mutation_rate_type'].lower() == 'decaying':
            mutation_rate = mutation_rate / sqrt(self.current_generation + 1)

        # Gaussian
        if mutation_bucket == 0:
            # Mutate weights
            gaussian_mutation(child1_weights, mutation_rate, scale=scale)
            gaussian_mutation(child2_weights, mutation_rate, scale=scale)

            # Mutate bias
            gaussian_mutation(child1_bias, mutation_rate, scale=scale)
            gaussian_mutation(child2_bias, mutation_rate, scale=scale)

        # Uniform random
        elif mutation_bucket == 1:
            # Mutate weights
            random_uniform_mutation(child1_weights, mutation_rate, -1, 1)
            random_uniform_mutation(child2_weights, mutation_rate, -1, 1)

            # Mutate bias
            random_uniform_mutation(child1_bias, mutation_rate, -1, 1)
            random_uniform_mutation(child2_bias, mutation_rate, -1, 1)

        else:
            raise Exception('Unable to determine valid mutation based off probabilities.')
