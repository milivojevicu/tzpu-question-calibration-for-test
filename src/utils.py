### Imports

import math
import os

from decimal import Decimal

### Vars

# Euler's number
e = Decimal("2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274")

# Parameter indices
index_a = 0; index_b = 1; index_c = 2;

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

