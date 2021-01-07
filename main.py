import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import csv
import os
from decimal import Decimal

### Vars

# Parameter indices
index_a = 0; index_b = 1; index_c = 2;
# Item params csv file
items_file = 'input/items_small.csv'
# Number of examinees for the initial test
num_of_examinees = 200

### IRT

# Probabilty function for the 3PL model
#   a, b, c - [float] item parameters
#   t - [float] examinee theta
def irt_prob(a, b, c, t):
    return c + (1 - c) / (1 + math.exp(-a * (t - b)))

### Load item data

items = np.array([[0, 0, 0]])
with open(items_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if (len(row) == 3):
            items = np.append(items, [[float(row[0]), float(row[1]), float(row[2])]], axis=0)
items = np.delete(items, 0, 0)


### Generate examinee theta values

examinees = sorted(np.random.normal(0, 1, num_of_examinees))


### Generate initial test results

init_test = np.array([[0]*len(items)])
raw_score = np.array([[0]])
index_examinee = 1
for theta in examinees:
    init_test = np.append(init_test, [[0]*len(items)], axis=0)
    index_item = 0
    raw_score_tmp = 0
    for i in items:
        P = irt_prob(i[index_a], i[index_b], i[index_c], theta)
        init_test[index_examinee][index_item] = random.choices([0, 1], [1 - P, P], k=1)[0]
        raw_score_tmp += init_test[index_examinee][index_item] 
        index_item += 1
    raw_score = np.append(raw_score, [[raw_score_tmp]], axis=0)
    index_examinee += 1
init_test = np.delete(init_test, 0, 0)
raw_score = np.delete(raw_score, 0, 0)


### Eliminate out of scope data
# TODO Eliminate examinees and items with all correct or incorrect answers


### Genetic algorithm

# Gene encoding parameters
precision = [12, 12, 12]    # Precision of encoding for each parameter (a, b and c)
precision_total = 0         # Total lengh of the genome (sum of parameter precisions)
precision_max = [0, 0, 0]   # Maximum integer value that can be stored with above specified precision
# Calculate precision total and maximum
for i in range(len(precision)):
    precision_total += precision[i]
    precision_max[i] = math.pow(2, precision[i]) - 1
# Lower and upper boundaries for each of the three parameters
range_a_lower = .3; range_a_upper =  3
range_b_lower = -3; range_b_upper =  3
range_c_lower =  0; range_c_upper = .5

# Translates value from old to new range
#   value - [float] value to translate
#   omin, omax - [float] old range [omin, omax]
#   nmin, nmax - [float] new range [nmin, nmax]
def translate(value, omin, omax, nmin, nmax):
    return ((((value - omin) * (nmax - nmin)) / (omax - omin)) + nmin)

# Makes sure value is in specified range by setting it to the near 
#   boundary if it's below/above it
#   value - [float]
#   vmin, vmax - [float] range [vmin, vmax]
def inrange(value, vmin, vmax):
    if (value < vmin):
        value = vmin
    elif (value > vmax):
        value = vmax
    return value

# Generates a random number with overall normal distribution
#   loc - [float] distribution location
#   scale - [float] distribution scale
def variation(loc, scale):
    return np.random.normal(loc, scale, 1)[0]

# Fitness (Error) function for the GA: error = -log(likelihood)
# *Lower result = better
#   index_item - [int] index of an item from the items array loaded from the csv file
#   a, b, c - [float] parameters
def fitness_func(index_item, a, b, c):
    L = 1
    for i in range(len(init_test[:, index_item])):
        u = init_test[i, index_item]
        P = irt_prob(a, b, c, (translate(raw_score[i], 0, len(items), -3, 3)))
        Q = 1 - P
        L *= pow(P, u) * pow(Q, 1 - u)
    d = Decimal(L)
    return -(d.ln())


# Chromosome class

class chromosome:
    # Initialization
    #   chromosome().generate(...)  to generate a randomized chromosome
    #   chromosome().create(...)    to create a chromosome with the specified genome
    def __init__(self):
        pass

    # Radomly generate a chromosome
    #   index_item - [int] used to determine a location for parameter b variation based on initial test raw score
    #   c - [float] starting value for parameter c (guessing parameter)
    #   var - [boolean] enables/disables variation
    def generate(self, index_item, c, var):
        # Normaly distributed parameter variation: np.random.normal(0, 1, 1)[0]
        self.index_item = index_item
        self.genes = np.array([0]*precision_total)
        # a, init to random value with left side of normal distribution translated to [.3, 1] and right side to [1, 3]
        a = inrange(variation(0, 1), -3, 3)
        if (a < 0):
            a = translate(a, -3, 0, .3, 1)
        else:
            a = translate(a, 0, 3, 1, 3)
        if (not var):
            a = 1
        a = inrange(a, range_a_lower, range_a_upper)
        a = bin((int) (translate(a, range_a_lower, range_a_upper, 0, precision_max[index_a])))[2:]
        offset = precision[index_a] - len(a)
        for i in range(len(a)):
            self.genes[offset + i] = (int) (a[i])
        # b, init value based on number of correct answers in the initial test
        b = 0
        for t in init_test:
            b += t[index_item]
        b = (1 - (b / (num_of_examinees * 1.0)))
        b = translate(b, 0, 1, -3, 3)
        if (var):
            b += inrange(variation(0, .1), -1, 1)
        b = bin((int) (translate(b, -3, 3, 0, precision_max[index_b])))[2:]
        offset = precision[index_b] - len(b)
        for i in range(len(b)):
            self.genes[precision[index_a] + offset + i] = (int) (b[i])
        # c, defaul value passed in constructor
        if (var):
            c += translate(inrange(variation(0, 1), -3, 3), -3, 3, -range_c_upper, range_c_upper)
        c = inrange(c, range_c_lower, range_c_upper)
        c = bin((int) (translate(c, range_c_lower, range_c_upper, 0, precision_max[index_c])))[2:]
        offset = precision[index_c] - len(c)
        for i in range(len(c)):
            self.genes[precision[index_a] + precision[index_b] + offset + i] = (int) (c[i])
        return self

    # Creates a chromosome with the specified genome
    #   index_item - [int] index of the item this chromosome coresonds to, needed for fitness calculation
    #   genes - [numpy array]
    def create(self, index_item, genes):
        self.index_item = index_item
        self.genes = genes
        return self

    # [numpy array] Returns genes
    def get_genes(self):
        return self.genes

    # [3*float] Returns a tuplet of three parameters
    def get_params(self):
        a = 0
        for i in range(precision[index_a]):
            a += self.genes[i] * math.pow(2, precision[index_a] - i - 1)
        a = translate(a, 0, precision_max[index_a], range_a_lower, range_a_upper)
        b = 0
        for i in range(precision[index_b]):
            b += self.genes[precision[index_a] + i] * math.pow(2, precision[index_b] - i - 1)
        b = translate(b, 0, precision_max[index_b], range_b_lower, range_b_upper)
        c = 0
        for i in range(precision[index_c]):
            c += self.genes[precision[index_a] + precision[index_b] + i] * math.pow(2, precision[index_c] - i - 1)
        c = translate(c, 0, precision_max[index_c], range_c_lower, range_c_upper)
        return a, b, c

    # [Decimal] Calculates, sets and returns a fitness value
    def calc_fitness(self):
        # error function, lower is better
        # error = -log(likelihood)
        a, b, c = self.get_params()
        self.fitness = fitness_func(self.index_item, a, b, c)
        return self.fitness


# Algorithm

class genalg:
    # Initialize to default parameters
    #   index_item - [int] coresponding item, needed for plotting functions and calculating fitness
    def __init__(self, index_item):
        self.population = 25
        self.mutation_rate = .1
        self.crossover_rate = .3
        self.max_iterations = 10000
        self.chromosomes = [None] * self.population
        self.f_sum = 0
        self.f_sum_old = 0
        self.f_min = 0
        self.f_min_old = 0
        self.f_avg = 0
        self.f_avg_old = 0
        self.weights = [.0] * self.population
        self.index_item = index_item
        self.top_to_print = 50

    # Calculates fitness and weights, then outputs some stats and the population
    #   generation - [int] generation index
    #   save - [boolean] used to calculate algo wide fitness deltas when iterating throught
    #       multiple generations
    def print(self, generation, save):
        # calc generation stats
        if (not save):
            self.f_sum_old = self.f_sum
            self.f_min_old = self.f_min
            self.f_avg_old = self.f_avg
        self.f_sum = 0
        self.f_min = 0
        for j in range(self.population):
            fitness = (float) (self.chromosomes[j].fitness)
            self.f_sum += fitness
            if (self.f_min == 0):
                self.f_min = fitness
            else:
                self.f_min = min(self.f_min, fitness)
        self.f_avg = self.f_sum / self.population
        # print generation stats
        print(
            "Item:", self.index_item,
            "\nGeneration:", generation,
            "\nFitness:",
            'sum={:.3f}(Δ={:.3f})'.format(self.f_sum, self.f_sum - self.f_sum_old),
            'avg={:.3f}(Δ={:.3f})'.format(self.f_avg, self.f_avg - self.f_avg_old),
            'min={:.3f}(Δ={:.3f})'.format(self.f_min, self.f_min - self.f_min_old)
        )
        print()
        # target
        print(" Gen :",
                '{:6.3f}'.format(items[self.index_item][index_a]),
                '{:6.3f}'.format(items[self.index_item][index_b]),
                '{:6.3f}'.format(items[self.index_item][index_c]),
                'f={:6.3f}'.format(fitness_func(
                    self.index_item,
                    items[self.index_item][index_a],
                    items[self.index_item][index_b],
                    items[self.index_item][index_c])
                    )
                )
        # pick probability
        for j in range(self.population):
            self.weights[j] = 1 - ((float) (self.chromosomes[j].fitness) / self.f_sum)
        # population
        for j in range(min(self.top_to_print, self.population)):
            a, b, c = self.chromosomes[j].get_params()
            print('{:4d}'.format(j), ":",
                    '{:6.3f}'.format(a),
                    '{:6.3f}'.format(b),
                    '{:6.3f}'.format(c),
                    'f={:6.3f}'.format(self.chromosomes[j].fitness),
                    self.chromosomes[j].get_genes()
                    )
        print()

    # Generates a population
    def generate(self):
        item_chromosomes = [None] * self.population
        for j in range(self.population):
            item_chromosomes[j] = chromosome().generate(self.index_item, items[self.index_item][index_c], j != 0)
        self.chromosomes = item_chromosomes

    # Creates a new chromosome with crossover and mutation
    def create(self):
        pair = random.choices(self.chromosomes, self.weights, k=2)
        result = pair[0].get_genes().copy()
        for j in range(precision_total):
            if (random.random() < self.mutation_rate):
                result[j] = random.randint(0, 1)
            elif (random.randint(0, 1) == 1):
                result[j] = pair[1].get_genes()[j]
        return chromosome().create(self.index_item, result)

    # Creates and sets the next generation of chromosomes
    def next_gen(self):
        next_chromosomes = [None] * self.population
        for j in range(self.population):
            if (j < (int) (self.population * self.crossover_rate)):
                next_chromosomes[j] = chromosome().create(self.index_item, self.chromosomes[j].get_genes().copy())
            else:
                next_chromosomes[j] = self.create()
        self.chromosomes = next_chromosomes

    # Plots grafs of the raw item score and probabilty functions for parameters used to generate
    #   data and parameters from the chromosome of the sepcified index
    #   index - [int] chromosome index for probability function plotting
    def plot(self, index):
        mpl.style.use('seaborn')
        fig, axs = plt.subplots(1)
        plt.subplots_adjust(wspace=.5, hspace=.3)

        plot_raw_p = np.array([[.0]]*7)
        plot_raw_s = np.array([0]*7)
        plot_raw_t = np.array([0]*7)
        for e in range(num_of_examinees):
            theta = inrange(round(examinees[e]), -3, 3) + 3
            plot_raw_t[theta] += 1
            if (init_test[e][self.index_item] == 1):
                plot_raw_s[theta] += 1
            plot_raw_p[theta] = np.array([plot_raw_s[theta] / plot_raw_t[theta]])
        axs.plot(np.arange(-3, 3.1, 1), plot_raw_p, label='Raw score')

        item = items[self.index_item]
        plot_target_params = np.array([[0]])
        plot_best_ga = np.array([[0]])
        a, b, c = self.chromosomes[index].get_params()
        for x in range(70):
            theta = -3.5 + x * .1
            P = irt_prob(item[index_a], item[index_b], item[index_c], theta)
            plot_target_params = np.append(plot_target_params, [[P]], axis=0)
            P = irt_prob(a, b, c, theta)
            plot_best_ga = np.append(plot_best_ga, [[P]], axis=0)
        plot_target_params = np.delete(plot_target_params, 0, 0)
        plot_best_ga = np.delete(plot_best_ga, 0, 0)
        axs.plot(np.arange(-3.5, 3.5, .1), plot_target_params, label='Gen params')
        axs.plot(np.arange(-3.5, 3.5, .1), plot_best_ga, label='GA params')

        axs.set_xlim(-3.5, 3.5)
        axs.set_xticks(range(-3, 4))
        axs.set_yticks(np.arange(0, 1.1, .1))
        axs.set_ylim(0, 1)
        plt.legend();
        plt.show()
        pass

    # Interactivly iterates through generations of the GA
    def iterate(self):
        go_for = 1
        save = False
        generation = 0
        while (generation < self.max_iterations):
            if (generation != 0):
                if (go_for <= 0):
                    save = False
                    cmd = input("'q' to quit\n'n' for next item\n'p' for plot \
                            \n[number] for n generations\n<CR> for 1 generation\n>>>")
                    if (cmd == "q"):
                        return "q"
                    elif (cmd == "n"):
                        os.system('cls' if os.name=='nt' else 'clear')
                        return ""
                    elif (cmd.split()[0] == "p"):
                        if (len(cmd.split()) >= 2 and cmd.split()[1].isdigit()):
                            self.plot(int(cmd.split()[1]))
                        else:
                            self.plot(0)
                        os.system('cls' if os.name=='nt' else 'clear')
                        self.print(generation, False)
                        continue
                    elif (cmd.isdigit()):
                        go_for = int(cmd)
                        save = True
                    else:
                        go_for += 1
                self.next_gen()
            go_for -= 1

            self.chromosomes = \
                    sorted(self.chromosomes, key=lambda chromosome: chromosome.calc_fitness())
            os.system('cls' if os.name=='nt' else 'clear')
            self.print(generation, save)
            generation += 1
        return ""

## MAIN

# Run GA for each loaded item
for i in range(len(items)):
    g = genalg(i)
    g.generate()
    r = g.iterate()
    if (r == "q"):
        break;

