import os.path
import json
import scipy.misc
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage.transform import resize, rotate


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next
# function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time
# it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    batch_index = 0         # Counts how many batches are created so far
    images_in_batch = 0     # Counts how many images are already in the current batch

    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.epoch = 0

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        images = []
        labels = []
        offset = self.batch_index * self.batch_size
        self.images_in_batch = 0    # Reset this variable to mind conflicts with earlier created batches
        with open(self.label_path, "r") as read_json:
            label_data = json.load(read_json)

        x = 0
        while x < self.batch_size and self.images_in_batch < self.batch_size:
            # If more data is needed than available, reset batch_index and thus offset to start again from beginning
            # In that case also update the epoch
            y = len([entry for entry in os.listdir(self.file_path) if
                    os.path.isfile(os.path.join(self.file_path, entry))])
            if x + offset >= len([entry for entry in os.listdir(self.file_path) if
                                  os.path.isfile(os.path.join(self.file_path, entry))]):
                self.batch_index = 0
                offset = 0
                self.epoch += 1
                x = 0
                # Reset the loop
                continue

            # Open image, resize and append it to the images list
            image = np.load(os.path.join(self.file_path, str(x + offset) + '.npy'))
            image = resize(image, (self.image_size[0], self.image_size[1]))

            images.append(self.augment(image))
            labels.append(label_data.get(str(x + offset)))
            self.images_in_batch += 1
            x += 1

        self.batch_index += 1
        # Zip images and labels together, shuffle them and unzip them again
        if self.shuffle:
            temp = list(zip(images, labels))
            random.shuffle(temp)
            images, labels = zip(*temp)
            images, labels = list(images), list(labels)

        images = np.array(images)   # convert from list to numpy array

        return images, labels

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        if self.mirroring and bool(random.getrandbits(1)):
            img = np.fliplr(img)

        if self.rotation and bool(random.getrandbits(1)):
            # Randomly choose the angle to rotate
            angle = random.choice([90, 180, 270])
            img = rotate(img, angle)

        return img

    def current_epoch(self):
        # return the epoch number
        return self.epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict.get(x)

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        images, labels = self.next()
        plt.figure(figsize=(10, 10))
        for i in range(self.batch_size):
            plt.subplot(10, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i])
            plt.xlabel(self.class_name(labels[i]))
        plt.show()

