import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# install image_slicer
!pip install image_slicer
import image_slicer

def Puzzle_RandomShuffle(path, n, seed):
    # """
    # @input  path:  image path
    # 		n:		number of tiles/puzzles
    # 		seed:	seed for Random
    # @return randomly shuffled puzzle image
    # """

    c = []
    i = 0

    # Read image
    #cv2_imshow(img)

    # split image into n tiles
    tiles=image_slicer.slice(path, n, save=False)
    #plot puzzle
    #plt.imshow(np.asarray(tiles[0].image))

    # Shuffle tiles
    random_index = np.arange(n)
    np.random.shuffle(random_index)

    # Collect oordinates for each tile
    for tile in tiles:
      c.append(tile.coords)

    # Re-assign coords
    for tile in tiles:
      tile.coords = c[random_index[i]]
      i = i + 1
      #print(tile.coords)

    return image_slicer.join(tiles)