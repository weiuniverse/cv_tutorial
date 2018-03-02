'''
homework0:
Write an OpenCV program to do the following things:
1. Read an image from a file and display it to the screen
2. Add to, subtract from, multiply or divide each pixel with a scalar, display the result.
3. Resize the image uniformly by 1/2

@author Zhengwei Wei 111492958
'''


import numpy as np
import cv2

# 1. read and show
img = cv2.imread("1.jpg")
print("show the image")
cv2.imshow('image',img)
k = cv2.waitKey(0)
print("input enter to continue")

# 2.operation
# multiply
print("multiply 2")
img2 = 2 * img
cv2.imshow('multiply',img2)
k = cv2.waitKey(0)
print("input enter to continue")
# divide
print("divide 2")
img3 = img/2
cv2.imshow('divide',img3)
k = cv2.waitKey(0)
print("input enter to continue")
# add
print("add 100")
img4 = img + 100
cv2.imshow('add',img4)
k = cv2.waitKey(0)
print("input enter to continue")
# subtract
print("subtract 100")
img5 = img - 100
cv2.imshow('subtract',img5)
k = cv2.waitKey(0)
print("input enter to continue")

# 3. resize
height,width = img.shape[:2]
print("resize the picture")
res = cv2.resize(img,(width/2,height/2),interpolation=cv2.INTER_AREA)
cv2.imshow('resize',res)
k = cv2.waitKey(0)
print("input enter to continue")
cv2.destroyAllWindows()
