import cv2
import imutils
import numpy as np
from keras.models import load_model

def displayImage(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def processImage(basic, image):
    image = cv2.GaussianBlur(image.copy(), (5, 5), 0)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    image = cv2.bitwise_not(image, image)

    contours = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    perimeter = cv2.arcLength(contours[0], True) 
    dimensions = cv2.approxPolyDP(contours[0], 0.03*perimeter, True)
    cv2.drawContours(basic, [dimensions], 0, (10,100,255), 1)
    displayImage('contour', basic)
    return dimensions

def widthHeight(rect):
    (tl, tr, br, bl) = rect
    widthTop = np.sqrt( (tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
    widthBottom = np.sqrt( (bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
    width = max(int(widthTop), int(widthBottom))
    heightTop = np.sqrt( (tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
    heightBottom = np.sqrt( (tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
    height = max(int(heightTop), int(heightBottom))
    return (width, height)

def cornerCoordinates(dimensions):
    dimensions = dimensions.reshape(4,2)
    corners = np.zeros((4,2), dtype="float32")
    s = np.sum(dimensions, axis=1)
    d = np.diff(dimensions, axis=1)
    corners[0] = dimensions[np.argmin(s)]
    corners[2] = dimensions[np.argmax(s)]
    corners[1] = dimensions[np.argmin(d)]
    corners[3] = dimensions[np.argmax(d)]
    width, height = widthHeight(corners)
    return (width, height, corners)

def croppedImage(image, width, height, corners):
    newCorners = np.array([ [0,0], [width, 0], [width, height], [0, height] ], dtype="float32")
    transformer = cv2.getPerspectiveTransform(corners, newCorners)
    finalImage = cv2.warpPerspective(image, transformer, (width, height))
    displayImage('cropped', finalImage)
    return finalImage

def show81Boxes(width, height, image):
    boxes = []
    for i in range(9):
        for j in range(9):
            topLeft = (j*width, i*height)
            bottomRight = ((j+1)*width, (i+1)*height)
            boxes.append((topLeft, bottomRight))
    
    for box in boxes:
        image = cv2.rectangle(image, tuple(x for x in box[0]), tuple(x for x in box[1]), (255,0,255), 0)
    displayImage('small boxes', image)
    return boxes

def identifyCharacters(image, boxes, model, width, height):
    x = 0
    numberCells = []
    for box in boxes:
        smallImage = image[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        img = cv2.cvtColor(smallImage, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # make all 4 edges white
        img[0:4, :] = 255
        img[:, 0:7] = 255
        img[24:28, :] = 255
        img[:, 21:28] = 255
        
        img = cv2.bitwise_not(img, img)
        displayImage('1/81', img)
        unique, counts = np.unique(img, return_counts=True)
        # otherwise, amount of white in the image is very less and can be considered empty.
        if 784 - counts[0] > 73:
            pred = model.predict(img.reshape(1, 28, 28, 1))
            value = pred.argmax()
            numberCells.append((x, value))
        x += 1
    return numberCells
        
def showSudoku(numberCells):
    grid = [[" " for _ in range(9)] for _ in range(9)]
    for item in numberCells:
        pos = tuple((int(item[0] / 9), int(item[0] % 9)))
        grid[pos[0]][pos[1]] = item[1]

    for row in grid:
        for item in row:
            print("[", item, end ="]")
        print("\n")
            

base = cv2.imread("image2.jpg")
image = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
displayImage('first', base)
dimensions = processImage(base, image)
width, height, corners = cornerCoordinates(dimensions)
finalImage = croppedImage(base, width, height, corners)

smallWidth = int(width/9) + 1
smallHeight = int(height/9) + 1

boxes = show81Boxes(smallWidth, smallHeight, finalImage)
model = load_model("Number.h5")
numberCells = identifyCharacters(finalImage, boxes, model, smallWidth, smallHeight)
showSudoku(numberCells)