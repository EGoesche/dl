import numpy as np
import matplotlib.pyplot as plt


class Checker:
    # https://stackoverflow.com/questions/32704485/drawing-a-checkerboard-in-python
    output = None

    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size

    def draw(self):
        """
        Draws the checkerboard.

        :return: the checkerboard
        :rtype: numpy.ndarray
        :raises ValueError: if the resolution is not evenly dividable by 2 times tile_size
        """
        if self.resolution % (2 * self.tile_size) != 0:
            raise ValueError("Resolution must be evenly dividable by 2 times tile_size")
        tile = np.concatenate((np.zeros(self.tile_size), np.ones(self.tile_size)))
        row = np.pad(tile, int((self.resolution ** 2) / 2 - self.tile_size), 'wrap').reshape((self.resolution,
                                                                                              self.resolution))
        row_overlap = (row + row.T)
        self.output = np.where(row_overlap == 1, 1, 0)
        return self.output

    def show(self):
        """
        Shows the checkerboard.
        """
        plt.imshow(self.draw(), cmap='gray')
        plt.show()


class Circle:
    # https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
    output = None

    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self):
        """
        Draws a circle.
        :return: numpy.ndarray
        """
        x = np.linspace(-10, 10, 491)
        y = np.linspace(-10, 10, 491)
        x, y = np.meshgrid(x, y)
        mask = np.sqrt((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2)
        mask = np.where(mask > self.radius, 0, 1)
        self.output = mask
        return self.output

    def show(self):
        """
        Shows the circle.
        """
        plt.imshow(self.draw(), cmap='gray')
        plt.show()


class Spectrum:
    output = None

    def __init__(self, resolution):
        self.resolution = resolution

    def draw(self):
        x = np.ones((self.resolution, self.resolution, 3))
        x[:, :, 0:3] = np.random.uniform(0, 1, (3,))
        plt.imshow(x)
        plt.figure()

        y = np.ones((self.resolution, self.resolution, 3))
        y[:, :, 0:3] = np.random.uniform(0, 1, (3,))
        plt.imshow(y)

        plt.figure()
        c = np.linspace(0, 1, self.resolution)[:, None, None]
        gradient = x + (y - x) * c
        self.output = gradient
        return self.output

    def show(self):
        """
        Shows the spectrum.
        """
        plt.imshow(self.draw())
        plt.show()

