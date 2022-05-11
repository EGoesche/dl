from pattern import *
from generator import *

# c1 = Circle(1024, 200, (512, 456))
# c1.show()

g1 = ImageGenerator(file_path='./data/exercise_data/',
                    label_path='./data/Labels.json',
                    batch_size=3,
                    image_size=(500, 500),
                    rotation=True,
                    mirroring=True)
g1.show()
g1.show()
