import sys
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt

from typing import List
from puzzle import *
import numpy as np

from settings import settings
from math import sqrt
from decimal import Decimal
import random, pickle
import csv

from nn_viz import NeuralNetworkViz
from neural_network import FeedForwardNetwork, sigmoid, linear, relu

from genetic_algorithm.population import Population
from genetic_algorithm.selection import elitism_selection, roulette_wheel_selection, tournament_selection
from genetic_algorithm.mutation import gaussian_mutation, random_uniform_mutation
from genetic_algorithm.crossover import simulated_binary_crossover as SBX
from genetic_algorithm.crossover import uniform_binary_crossover, single_point_binary_crossover

SQUARE_SIZE = (10, 10)

NEW = False
NR_OLD_ONES = 10

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings, show= (not NEW), fps=5000):
        super().__init__()
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QtGui.QColor(0, 0, 0))
        self.setPalette(palette)
        self.settings = settings
        self._SBX_eta = self.settings['SBX_eta']
        self._mutation_bins = np.cumsum([self.settings['probability_gaussian'],
                                        self.settings['probability_random_uniform']
        ])
        self._crossover_bins = np.cumsum([self.settings['probability_SBX'],
                                         self.settings['probability_SPBX']
        ])
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
        self.border = (0, 10, 0, 10)  # Left, Top, Right, Bottom
        self.puzzle_widget_width = SQUARE_SIZE[0] * self.board_size[0]
        self.puzzle_widget_height = SQUARE_SIZE[1] * self.board_size[1]

        # Allows padding of the other elements even if we need to restrict the size of the play area
        self._puzzle_widget_width = max(self.puzzle_widget_width, 620)
        self._puzzle_widget_height = max(self.puzzle_widget_height, 600)

        self.top = 150
        self.left = 150
        self.width = self._puzzle_widget_width + 700 + self.border[0] + self.border[2]
        self.height = self._puzzle_widget_height + self.border[1] + self.border[3] + 200

        self.nrGroupRange = settings['nrGroupRange']
        self.nrBlocksRange = (settings['nrBlocksRange'][0] * self.nrGroupRange[0], settings['nrBlocksRange'][1] * self.nrGroupRange[1])

        individuals: List[Individual] = []

        if NEW:
            for _ in range(self.settings['num_parents']):
                nrG = random.randint(self.nrGroupRange[0], self.nrGroupRange[1])
                nrC = random.randint(self.nrBlocksRange[0], self.nrBlocksRange[1])
                individual = Puzzle(self.board_size, nrG, nrC,
                                    hidden_layer_architecture=self.settings['hidden_network_architecture'],
                                    hidden_activation=self.settings['hidden_layer_activation'],
                                    output_activation=self.settings['output_layer_activation'])
                individuals.append(individual)

        else:
            for i in range(NR_OLD_ONES):
                individual = load_puzzle('population15', 'best_snake'+str(i+320), self.settings)
                individuals.append(individual)

        self.best_fitness = np.inf

        self._current_individual = 0
        self.population = Population(individuals)

        self.puzzle = self.population.individuals[self._current_individual]
        self.current_generation = 0

        self.init_window()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(1000./fps)

        if show:
            self.show()

    def init_window(self):
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setWindowTitle('Puzzle AI')
        self.setGeometry(self.top, self.left, self.width, self.height)

        # Create the Neural Network window
        self.nn_viz_window = NeuralNetworkViz(self.centralWidget, self.puzzle)
        self.nn_viz_window.setGeometry(QtCore.QRect(0, 0, 600, self._puzzle_widget_height + self.border[1] + self.border[3] + 200))
        self.nn_viz_window.setObjectName('nn_viz_window')

        # Create PuzzleWidget window
        self.puzzle_widget_window = PuzzleWidget(self.centralWidget, self.board_size, self.puzzle)
        self.puzzle_widget_window.setGeometry(QtCore.QRect(600 + self.border[0],
                                            self.border[1], self.puzzle_widget_width, self.puzzle_widget_height))
        self.puzzle_widget_window.setObjectName('puzzle_widget_window')

        # Genetic Algorithm Stats window
        self.ga_window = GeneticAlgoWidget(self.centralWidget, self.settings)
        self.ga_window.setGeometry(QtCore.QRect(600, self.border[1] + self.border[3] + self.puzzle_widget_height, self._puzzle_widget_width + self.border[0] + self.border[2] + 100, 200-10))
        self.ga_window.setObjectName('ga_window')

    def update(self) -> None:
        self.puzzle_widget_window.update()
        self.nn_viz_window.update()
        # Current individual is alive
        if not self.puzzle.finished:
            self.puzzle.fill()

        # Current individual is dead
        else:
            # Calculate fitness of current individual
            self.puzzle.calculate_fitness()
            fitness = self.puzzle.fitness
            print(self._current_individual, fitness)

            # fieldnames = ['frames', 'score', 'fitness']
            # f = os.path.join(os.getcwd(), 'test_del3_1_0_stats.csv')
            # write_header = True
            # if os.path.exists(f):
            #     write_header = False

            # #@TODO: Remove this stats write
            # with open(f, 'a') as csvfile:
            #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')
            #     if write_header:
            #         writer.writeheader()

            #     d = {}
            #     d['frames'] = self.snake._frames
            #     d['score'] = self.snake.score
            #     d['fitness'] = fitness

            #     writer.writerow(d)


            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.ga_window.best_fitness_label.setText('{:.2E}'.format(Decimal(fitness)))

            self._current_individual += 1
            # Next generation
            if (self.current_generation > 0 and self._current_individual == self._next_gen_size) or\
                (self.current_generation == 0 and self._current_individual == settings['num_parents']):
                print(self.settings)
                print('======================= Gneration {} ======================='.format(self.current_generation))
                print('----Max fitness:', self.population.fittest_individual.fitness)
                print('----Average fitness:', self.population.average_fitness)
                save_puzzle('population', 'best_snake'+str(self.current_generation), self.population.fittest_individual, self.settings)
                self.next_generation()
            else:
                current_pop = self.settings['num_parents'] if self.current_generation == 0 else self._next_gen_size
                self.ga_window.current_individual_label.setText('{}/{}'.format(self._current_individual + 1, current_pop))

            self.puzzle = self.population.individuals[self._current_individual]
            self.puzzle_widget_window.puzzle = self.puzzle
            self.nn_viz_window.puzzle = self.puzzle

    def next_generation(self):
        self._increment_generation()
        self._current_individual = 0

        # Calculate fitness of individuals
        for individual in self.population.individuals:
            individual.calculate_fitness()

        self.population.individuals = elitism_selection(self.population, self.settings['num_parents'])

        random.shuffle(self.population.individuals)
        next_pop: List[Puzzle] = []

        # parents + offspring selection type ('plus')
        if self.settings['selection_type'].lower() == 'plus':

            for individual in self.population.individuals:
                params = individual.network.params
                board_size = individual.board_size
                hidden_layer_architecture = individual.hidden_layer_architecture
                hidden_activation = individual.hidden_activation
                output_activation = individual.output_activation

                start_pos = individual.start_pos

                # If the individual is still alive, they survive
                if individual._fitness < 1000:
                    nrG = random.randint(self.nrGroupRange[0], self.nrGroupRange[1])
                    nrC = random.randint(self.nrBlocksRange[0], self.nrBlocksRange[1])
                    s = Puzzle(board_size, nrG, nrC, chromosome=params, hidden_layer_architecture=hidden_layer_architecture,
                            hidden_activation=hidden_activation, output_activation=output_activation)
                    next_pop.append(s)


        while len(next_pop) < self._next_gen_size:
            p1, p2 = roulette_wheel_selection(self.population, 2)

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
            nrG = random.randint(self.nrGroupRange[0], self.nrGroupRange[1])
            nrC = random.randint(self.nrBlocksRange[0], self.nrBlocksRange[1])
            c1 = Puzzle(p1.board_size, nrG, nrC, chromosome=c1_params, hidden_layer_architecture=p1.hidden_layer_architecture,
                       hidden_activation=p1.hidden_activation, output_activation=p1.output_activation)

            nrG = random.randint(self.nrGroupRange[0], self.nrGroupRange[1])
            nrC = random.randint(self.nrBlocksRange[0], self.nrBlocksRange[1])
            c2 = Puzzle(p2.board_size, nrG, nrC, chromosome=c2_params, hidden_layer_architecture=p2.hidden_layer_architecture,
                       hidden_activation=p2.hidden_activation, output_activation=p2.output_activation)

            # Add children to the next generation
            next_pop.extend([c1, c2])

        # Set the next generation
        random.shuffle(next_pop)
        self.population.individuals = next_pop

    def _increment_generation(self):
        self.current_generation += 1
        self.ga_window.current_generation_label.setText(str(self.current_generation + 1))
        # self.ga_window.current_generation_label.setText("<font color='red'>" + str(self.loaded[self.current_generation]) + "</font>")

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


class GeneticAlgoWidget(QtWidgets.QWidget):
    def __init__(self, parent, settings):
        super().__init__(parent)

        font = QtGui.QFont('Times', 11, QtGui.QFont.Normal)
        font_bold = QtGui.QFont('Times', 11, QtGui.QFont.Bold)

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(1, 5)
        TOP_LEFT = Qt.AlignLeft | Qt.AlignTop

        LABEL_COL = 0
        STATS_COL = 1
        ROW = 0

        #### Generation stuff ####
        # Generation
        self._create_label_widget_in_grid('Generation: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self.current_generation_label = self._create_label_widget('1', font)
        grid.addWidget(self.current_generation_label, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        # Current individual
        self._create_label_widget_in_grid('Individual: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self.current_individual_label = self._create_label_widget('1/{}'.format(settings['num_parents']), font)
        grid.addWidget(self.current_individual_label, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        # Best fitness
        self._create_label_widget_in_grid('Best Fitness: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self.best_fitness_label = self._create_label_widget('{:.2E}'.format(Decimal('0.1')), font)
        grid.addWidget(self.best_fitness_label, ROW, STATS_COL, TOP_LEFT)

        ROW = 0
        LABEL_COL, STATS_COL = LABEL_COL + 2, STATS_COL + 2

        #### GA setting ####
        self._create_label_widget_in_grid('GA Settings', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        ROW += 1

        # Selection type
        selection_type = ' '.join([word.lower().capitalize() for word in settings['selection_type'].split('_')])
        self._create_label_widget_in_grid('Selection Type: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self._create_label_widget_in_grid(selection_type, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        # Crossover type
        prob_SBX = settings['probability_SBX']
        prob_SPBX = settings['probability_SPBX']
        crossover_type = '{:.0f}% SBX\n{:.0f}% SPBX'.format(prob_SBX*100, prob_SPBX*100)
        self._create_label_widget_in_grid('Crossover Type: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self._create_label_widget_in_grid(crossover_type, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        # Mutation type
        prob_gaussian = settings['probability_gaussian']
        prob_uniform = settings['probability_random_uniform']
        mutation_type = '{:.0f}% Gaussian\t\n{:.0f}% Uniform'.format(prob_gaussian*100, prob_uniform*100)
        self._create_label_widget_in_grid('Mutation Type: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self._create_label_widget_in_grid(mutation_type, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        # Mutation rate
        self._create_label_widget_in_grid('Mutation Rate:', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        mutation_rate_percent = '{:.0f}%'.format(settings['mutation_rate'] * 100)
        mutation_rate_type = settings['mutation_rate_type'].lower().capitalize()
        mutation_rate = mutation_rate_percent + ' + ' + mutation_rate_type
        self._create_label_widget_in_grid(mutation_rate, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1


        ROW = 0
        LABEL_COL, STATS_COL = LABEL_COL + 2, STATS_COL + 2

        #### NN setting ####
        self._create_label_widget_in_grid('NN Settings', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        ROW += 1

        # Hidden layer activation
        hidden_layer_activation = ' '.join([word.lower().capitalize() for word in settings['hidden_layer_activation'].split('_')])
        self._create_label_widget_in_grid('Hidden Activation: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self._create_label_widget_in_grid(hidden_layer_activation, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        # Output layer activation
        output_layer_activation = ' '.join([word.lower().capitalize() for word in settings['output_layer_activation'].split('_')])
        self._create_label_widget_in_grid('Output Activation: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self._create_label_widget_in_grid(output_layer_activation, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1

        # Network architecture
        network_architecture = '[{}, {}, 4]'.format(settings['vision_type'] * 3 + 4 + 4,
                                                    ', '.join([str(num_neurons) for num_neurons in settings['hidden_network_architecture']]))
        self._create_label_widget_in_grid('NN Architecture: ', font_bold, grid, ROW, LABEL_COL, TOP_LEFT)
        self._create_label_widget_in_grid(network_architecture, font, grid, ROW, STATS_COL, TOP_LEFT)
        ROW += 1


        grid.setSpacing(0)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        grid.setColumnStretch(5, 2)

        self.setLayout(grid)

        self.show()

    def _create_label_widget(self, string_label: str, font: QtGui.QFont) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel()
        label.setStyleSheet("QLabel {color: white;}")
        label.setText(string_label)
        label.setFont(font)
        label.setContentsMargins(0,0,0,0)
        return label

    def _create_label_widget_in_grid(self, string_label: str, font: QtGui.QFont,
                                     grid: QtWidgets.QGridLayout, row: int, col: int,
                                     alignment: Qt.Alignment) -> None:
        label = QtWidgets.QLabel()
        label.setStyleSheet("QLabel {color: white;}")
        label.setText(string_label)
        label.setFont(font)
        label.setContentsMargins(0,0,0,0)
        grid.addWidget(label, row, col, alignment)


class PuzzleWidget(QtWidgets.QWidget):
    def __init__(self, parent, board_size=(50, 50), puzzle=None):
        super().__init__(parent)
        self.board_size = board_size

        if puzzle:
            self.puzzle = puzzle
        self.setFocus()

        self.show()

    def update(self):
        if not self.puzzle.finished:
            self.puzzle.update()
            self.repaint()
        else:
            pass

    def draw_border(self, painter: QtGui.QPainter) -> None:
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.setPen(QtGui.QPen(Qt.black))
        width = self.frameGeometry().width()
        height = self.frameGeometry().height()
        painter.setPen(QtCore.Qt.white)
        painter.drawLine(0, 0, width, 0)
        painter.drawLine(width, 0, width, height)
        painter.drawLine(0, height, width, height)
        painter.drawLine(0, 0, 0, height)

    def draw_puzzle(self, painter: QtGui.QPainter) -> None:
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        pen = QtGui.QPen()
        pen.setColor(QtGui.QColor(0, 0, 0))

        painter.setPen(pen)
        brush = QtGui.QBrush()

        for p in self.puzzle.filled_array:
            cell = self.puzzle.board[p.x][p.y]
            if cell:
                c = cell.colour
                brush.setColor(QtGui.QColor(c[0], c[1], c[2]))
                pen.setColor(QtGui.QColor(c[0], c[1], c[2]))
                painter.setPen(pen)
                painter.setBrush(QtGui.QBrush(QtGui.QColor(c[0], c[1], c[2])))
                painter.drawRect(p.x * SQUARE_SIZE[0],  # Upper left x-coord
                                 p.y * SQUARE_SIZE[1],  # Upper left y-coord
                                 SQUARE_SIZE[0],            # Width
                                 SQUARE_SIZE[1])

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter()
        painter.begin(self)

        self.draw_border(painter)
        self.draw_puzzle(painter)

        painter.end()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(settings)
    sys.exit(app.exec_())
