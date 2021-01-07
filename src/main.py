### Imports

from genalg import genalg
from utils import *
from datagen import *

### Main

# Run GA for each loaded item
for i in range(len(items)):
    g = genalg(i, items[i], init_test, raw_score, examinees)
    g.generate()
    r = g.iterate()
    if (r == "q"):
        break;

