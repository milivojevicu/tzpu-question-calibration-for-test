### Import

import numpy as np

import random
import math
import csv

from utils import *

### Load item data

items = np.array([[0, 0, 0]])
with open(items_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if (len(row) == 3):
            items = np.append(items, [[float(row[0]), float(row[1]), float(row[2])]], axis=0)
items = np.delete(items, 0, 0)
print("il", len(items))

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

