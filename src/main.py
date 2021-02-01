### Imports

from genalg import genalg
from utils import *
from datagen import *

### Main

if os.path.exists("output.csv"):
    os.remove("output.csv")
f = open("output.csv", "w")
f.close()


# Run GA for each loaded item
for i in range(len(items)):
    g = genalg(i, items[i], init_test, raw_score, examinees)
    print("Generating initial population...")
    g.generate()
    r = g.iterate()
    if (r == "q"):
        break;

