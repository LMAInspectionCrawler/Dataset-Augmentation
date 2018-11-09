import cv2
import numpy as np
import os

# Constants
RED_INDEX = 2
GREEN_INDEX = 1
BLUE_INDEX = 0
MAX_COLOR = 255

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

# TODO: Loop through images

num = "0"
directory ='C:/Users/UTEPI/Desktop/images'

print(len(os.listdir(directory)))

image1 = cv2.imread('C:/Users/UTEPI/Desktop/images/f.jpg')
image2 = cv2.imread('C:/Users/UTEPI/Desktop/images/3.jpg')
image3 = cv2.imread('C:/Users/UTEPI/Desktop/images/5.jpg')

for file in os.listdir(directory):
    redModifier = 0
    greenModifier = 50
    blueModifier = 100
    

    #print (filename)
    # get image
    name = directory + "/" + file
    print(name)
    image = cv2.imread(name)
    cv2.imshow(file, image)
    cv2.waitKey(0)
    # White balance image
    image_white_balanced = white_balance(image)
    cv2.imshow(file, image_white_balanced)
    cv2.waitKey(0)
    height, width, depth = image_white_balanced.shape
    imageCopy = image_white_balanced.copy()

    for x in range(0, width):
        for y in range(0, height):
            imageCopy[y, x, RED_INDEX] = min(imageCopy[y, x, RED_INDEX] + redModifier, MAX_COLOR)   # TODO: optimize numpy for loops
            imageCopy[y, x, GREEN_INDEX] = min(imageCopy[y, x, GREEN_INDEX] + greenModifier, MAX_COLOR)
            imageCopy[y, x, BLUE_INDEX] = min(imageCopy[y, x, BLUE_INDEX] + blueModifier, MAX_COLOR)
        cv2.imshow(file, imageCopy)
    cv2.waitKey(0)


# Use an image
#image = cv2.imread("dog.jpg")
#cv2.imshow('Dog Image', image)
#

# White balance image
#image_white_balanced = white_balance(image)
#cv2.imshow('White Balanced Dog', image_white_balanced)

# TODO: Loop through different filters to use. You can choose certain filters to use
# ex: filter1 = [0, 0 , 100] filter2 = [100, 0, 0] filter3 = [30, 30, 30] ...

# Filters to apply

"""redModifier = 50
greenModifier = 200
blueModifier = 100

height, width, depth = image_white_balanced.shape

imageCopy = image_white_balanced.copy()
for x in range(0, width):
	for y in range(0, height):
		imageCopy[y, x, RED_INDEX] = min(imageCopy[y, x, RED_INDEX] + redModifier, MAX_COLOR)	# TODO: optimize numpy for loops
		imageCopy[y, x, GREEN_INDEX] = min(imageCopy[y, x, GREEN_INDEX] + greenModifier, MAX_COLOR)
		imageCopy[y, x, BLUE_INDEX] = min(imageCopy[y, x, BLUE_INDEX] + blueModifier, MAX_COLOR)

# TODO: image name prefix should be the same as the one being used
imageLabel = "Dog_r" + str(redModifier) + "_g" + str(greenModifier) + "_b" + str(blueModifier) + ".jpg"
cv2.imshow(imageLabel, imageCopy)
print("Saving " + imageLabel)
cv2.imwrite(imageLabel + ".jpg", imageCopy) # TODO: Figure out a way to save with the same classification

cv2.waitKey(0)					            # wait until a key is pressed before exiting program

"""