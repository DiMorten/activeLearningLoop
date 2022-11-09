import numpy as np
import cv2

a = cv2.imread('output/depths/istockphoto-641755828-612x612.jpg')
b = cv2.imread('output_dropout1/depths/istockphoto-641755828-612x612.jpg')

print(a.shape, b.shape)
print(np.all(a == b))
