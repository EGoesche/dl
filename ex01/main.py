from pattern import *
from generator import *

'''c1 = Circle(1024, 200, (512, 456))
c1.show()'''

g1 = ImageGenerator(file_path='./data/exercise_data/',
                    label_path='./data/Labels.json',
                    batch_size=50,
                    image_size=[32, 32, 3],
                    rotation=True,
                    mirroring=True,
                    shuffle=True)

g1.show()
g1.show()
