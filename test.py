from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

path = '/home/liu/tested/e19.jpg'
src = Image.open(path)
src = np.array(src)
height,width, _ = src.shape
print(height, width)
plt.figure()
plt.imshow(src)
plt.show()
