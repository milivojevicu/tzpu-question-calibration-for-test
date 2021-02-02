### Imports

from genalg import genalg
from utils import *
from datagen import *
from mle import MLE
import numpy as np

### Main

if os.path.exists("output.csv"):
    os.remove("output.csv")
f = open("output.csv", "w")
f.close()


# NOT WORKING PROPERLY
# mle_model = MLE(np.asarray(examinees), np.asarray(init_test))
# mle_model.fit()


# Run GA for each loaded item
for i in range(len(items)):
    g = genalg(i, items[i], init_test, raw_score, examinees)
    print("Generating initial population...")
    g.generate()
    r = g.iterate()
    if (r == "q"):
        break;

