#!/usr/bin/python
# -*- coding: utf-8 -*-
# Written by Hugo Burke | C16486314
'''
The code works in the following way:
    1 - Converts image to HSV and create a mask
    2 - Sharpen the mask to create a more defined edge on the object
    3 - Convert each pixel white that doesn't fall within the defined parameter
    4 - Convert new image to grey
    5 - Create a boundary between surrounding area and extract the object
    6 - Again convert each pixel to white, as in step 3 and convert back to RGB
'''
import cv2
import numpy as np


# Function that displays the image
def viewImage(image):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Function that sharpens the image
def sharpen(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    output = cv2.filter2D(image, -1, kernel)
    return output

# This for loop checks if each value of the image is greater 
# than a certain value and therefore converts it to white.
def convert(image):
    RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i in range(0, rows):
        for j in range(0, cols):
            k = mask[i, j]
            if k.any() > 0:
                RGB[i, j] = [255, 255, 255]
    Pic = RGB
    return Pic


image = cv2.imread('Shark 1.png') # Reads in image

RGB = cv2.cvtColor(image,
                   cv2.COLOR_BGR2RGB)  # Convert to RGB (Red, Green, Blue)
HSV = cv2.cvtColor(image,
                   cv2.COLOR_BGR2HSV)  # Convert to HSV (Hue, Saturation, Value)

# Limits for the shark, these were the best tolerances that maintained a large amount of both sharks
low = np.array([3, 110, 194])
high = np.array([255, 255, 255])

mask = cv2.inRange(HSV, low, high)  # Finds any values between low and high
mask = sharpen(mask)

(rows, cols, layers) = image.shape  # Gets rows and column values of the image
# Has similar function to convert function
for i in range(0, rows):
    for j in range(0, cols):
        k = mask[i, j]
        if k.any() > 0:
            RGB[i, j] = [255, 255, 255]
Pic = RGB

# Set the boundary between the object and background
gray = cv2.cvtColor(Pic, cv2.COLOR_BGR2GRAY)
(th, threshed) = cv2.threshold(gray, 127, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Find the first contour that greater than 100
cnts = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
cnts = sorted(cnts, key=cv2.contourArea)

# Taking the height and width of the image
(H, W) = Pic.shape[:2]
for cnt in cnts:
    (x, y, w, h) = cv2.boundingRect(cnt)
    if cv2.contourArea(
            cnt
    ) > 100 and 0.7 < w / h < 1.3 and W / 4 < x + w // 2 < W * 3 / 4 and H / 4 < y + h // 2 < H * 3 / 4:
        break

# Redefine mask for image and do bitwise-op
mask = np.zeros(Pic.shape[:2], np.uint8)
cv2.drawContours(mask, [cnt], -1, 255, -1)
dst = cv2.bitwise_and(Pic, Pic, mask=mask)
# [1]

# Convert function commented above
convert_image = convert(dst)
whiten_image = cv2.bitwise_or(
    convert_image, Pic)  # Compute the bit-wise OR of two arrays element-wise.
final_image = cv2.cvtColor(whiten_image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
viewImage(final_image)

## REFERENCES ##
# [1]Cropping Images and Isolating Specific Objects. China: Stack Overflow, 2018.
