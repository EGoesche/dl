from pattern import *
from generator import *

# c1 = Circle(1024, 200, (512, 456))
# c1.show()

g1 = ImageGenerator('./data/exercise_data/', './data/Labels.json', 10, (100, 100))
g1.show()