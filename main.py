import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import csv

### Vars

# Parameter indices
index_a = 0; index_b = 1; index_c = 2;
# Item params csv file
items_file = 'input/items.csv'
# Number of examinees for the initial test
num_of_examinees = 100 


### Plot init

mpl.style.use('seaborn')
fig, axs = plt.subplots(1)
plt.subplots_adjust(wspace=.5, hspace=.3)


### Load item data

items = np.array([[0, 0, 0]])
with open(items_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if (len(row) == 3):
            items = np.append(items, [[float(row[0]), float(row[1]), float(row[2])]], axis=0)
items = np.delete(items, 0, 0)

"""
print("Items (a, b, c):")
counter = 0
for i in items:
    print(f"\t{counter} | {i[index_a]}, {i[index_b]}, {i[index_c]}")
    counter += 1
"""


### Generate examinee theta values

examinees = sorted(np.random.normal(0, 1, num_of_examinees))

"""
print("Examinees (theta):")
counter = 0
for e in examinees:
    print(f"\t{counter} | {e}")
    counter += 1
#axs.plot(examinees, range(len(examinees)))
#plt.show()
"""


### Generate initial test results

init_test = np.array([[0]*len(items)])
raw_score = np.array([[0]])
index_examinee = 1
for theta in examinees:
    init_test = np.append(init_test, [[0]*len(items)], axis=0)
    index_item = 0
    raw_score_tmp = 0
    for i in items:
        P = i[index_c] + (1 - i[index_c]) / (1 + math.exp(-i[index_a] * (theta - i[index_b]))) 
        init_test[index_examinee][index_item] = random.choices([0, 1], [1 - P, P], k=1)[0]
        raw_score_tmp += init_test[index_examinee][index_item] 
        index_item += 1
    raw_score = np.append(raw_score, [[raw_score_tmp]], axis=0)
    index_examinee += 1
init_test = np.delete(init_test, 0, 0)
raw_score = np.delete(raw_score, 0, 0)

"""
print("Initial test results ([correct/incorect | 1/0], raw score):")
counter = 0
for t in init_test:
    print(f"\t{counter} | {t}, {raw_score[counter][0]}")
    counter += 1
"""


### Eliminate out of scope data
# Examinees and items with all correct or incorrect answers


### Genetic algorithm

# Chromosome class

precision = [12, 16, 8]
precision_total = 0
precision_max = [0, 0, 0]
for i in range(len(precision)):
    precision_max[i] = math.pow(2, precision[i]) - 1
    precision_total += precision[i]

range_a_lower = .3; range_a_upper =  3
range_b_lower = -3; range_b_upper =  3
range_c_lower =  0; range_c_upper = .5

def translate(value, omin, omax, nmin, nmax):
    return ((((value - omin) * (nmax - nmin)) / (omax - omin)) + nmin) 

class chromosome:
    def __init__(self, index_item, c):
        self.index_item = index_item
        self.genes = np.array([0]*precision_total)
        # a, init value = 1
        a = bin((int) (translate(1, range_a_lower, range_a_upper, 0, precision_max[index_a])))[2:]
        offset = precision[index_a] - len(a)
        for i in range(len(a)):
            self.genes[offset + i] = (int) (a[i])
        # b, init value based on number of correct answers in the initial test
        b = 0
        for t in init_test:
            b += t[index_item]
        # note: this translates b from range(0, 1) because inital b value is derived from 
        #   raw test results and is not in range(-3, 3)
        b = bin((int) (translate((1 - (b / (num_of_examinees * 1.0))), 0, 1, 0, precision_max[index_b])))[2:]
        offset = precision[index_b] - len(b)
        for i in range(len(b)):
            self.genes[precision[index_a] + offset + i] = (int) (b[i])
        # c, defaul value passed in constructor
        c = bin((int) (translate(c, range_c_lower, range_c_upper, 0, precision_max[index_c])))[2:]
        offset = precision[index_c] - len(c)
        for i in range(len(c)):
            self.genes[precision[index_a] + precision[index_b] + offset + i] = (int) (c[i])

    def get_genes(self):
        return self.genes

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

    def calc_fitness(self):
        # likelihood function
        L = 1
        a, b, c = self.get_params()
        for i in range(len(init_test[:, self.index_item])):
            u = init_test[i, self.index_item]
            P = c + (1 - c) / (1 + math.exp(-a * (translate(raw_score[i], 0, len(items), -3, 3) - b)))
            Q = 1 - P
            L *= pow(P, u) * pow(Q, 1 - u)
        return L


# Algorithm

ga_population = 1000
ga_mutation_rate = .1
ga_crossover_rate = .3
ga_max_iterations = 100

for i in range(len(items)):
    # TODO Generate chromosomes for each items
    c = chromosome(i, 0)

# TODO Iterate


## Plots

axs.set_xlim(-3.5, 3.5)
axs.set_xticks(range(-3, 4))
axs.set_ylim(0, 1)
axs.set_yticks(np.arange(0, 1.1, .1))

# plt.show()

