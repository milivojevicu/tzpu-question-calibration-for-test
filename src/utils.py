### Imports

import math
import os

### Vars

# Parameter indices
index_a = 0; index_b = 1; index_c = 2;
# Item params csv file
items_file = 'input/items_small.csv'
# Number of examinees for the initial test
num_of_examinees = 200

### Functions

# Probabilty function for the 3PL model
#   a, b, c - [float] item parameters
#   t - [float] examinee theta
def irt_prob(a, b, c, t):
    return c + (1 - c) / (1 + math.exp(-a * (t - b)))

# Clear console
def cls():
    os.system('cls' if os.name=='nt' else 'clear')

# Returns true if passed string can be converted into a float
def isfloat(str):
    try:
        float(str)
    except ValueError:
        return False
    return True

