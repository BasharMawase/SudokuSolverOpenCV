import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# READ THE MODEL WEIGHTS
def initializePredictionModel():
    model = load_model('digit_classifier.keras')
    return model

# 1 - Preprocessing Imagedef preProcess(img):
def preProcess(img):
        # Convert to grayscale
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Apply bilateral filter (better than Gaussian for preserving edges)
        imgBlur = cv.bilateralFilter(imgGray, 11, 17, 17)
        # Adaptive threshold with different parameters
        imgThresh = cv.adaptiveThreshold(imgBlur, 255,
                                          cv.ADAPTIVE_THRESH_MEAN_C,
                                          cv.THRESH_BINARY_INV, 11, 2)
        # Morphological operations to enhance grid lines
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        imgThresh = cv.morphologyEx(imgThresh, cv.MORPH_CLOSE, kernel)

        return imgThresh

# 2 - Reorder points for Warp Perspective
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

# 3 - FINDING THE BIGGEST COUNTOUR ASSUING THAT IS THE SUDUKO PUZZLE
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv.contourArea(i)
        if area > 5000:  # Lowered from your original 50 threshold
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri, True)

            # Show all large contours for debugging
            if area > max_area:
                max_area = area
                biggest = approx

            # If it's a quadrilateral (4 sides), consider it
            if len(approx) == 4:
                return approx, area

    return biggest, max_area
# 4 - TO SPLIT THE IMAGE INTO 81 DIFFRENT IMAGES
def splitBoxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            # Remove borders and resize to 32x32
            box = cv.resize(box, (32, 32))
            box = box[4:-4, 4:-4]  # Remove 4-pixel borders
            box = cv.resize(box, (28, 28))
            box = cv.bitwise_not(box)  # Invert colors (MNIST style)
            boxes.append(box)
    return boxes

def contains_digit(img, pixel_threshold=100, percentage_threshold=0.01):
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Normalize to full 0–255 range
    img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)

    # Invert image if background is dark
    if np.mean(img) < 127:
        img = 255 - img
    # Count number of "dark" pixels
    ink_pixels = np.sum(img < pixel_threshold)
    total_pixels = img.shape[0] * img.shape[1]

    # Return True if ink exceeds percentage threshold
    return (ink_pixels / total_pixels) > percentage_threshold

def prepare_image(image):
    image = image.astype(np.float32)
    # Convert to grayscale if necessary
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Histogram normalization to increase contrast
    image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)

    # Invert image if background is dark
    if np.mean(image) < 127:
        image = 255 - image

    # Pad to make square with white (255) background
    h, w = image.shape
    size_diff = abs(h - w)
    if h > w:
        pad_left = size_diff // 2
        pad_right = size_diff - pad_left
        image = cv.copyMakeBorder(image,0, 0, pad_left, pad_right,
                                   borderType=cv.BORDER_CONSTANT, value=255)
    elif w > h:
        pad_top = size_diff // 2
        pad_bottom = size_diff - pad_top
        image = cv.copyMakeBorder(image, pad_top, pad_bottom, 0, 0,
                                   borderType=cv.BORDER_CONSTANT, value=255)
    # Resize to 28x28
    image = cv.resize(image, (28, 28), interpolation=cv.INTER_AREA)
    return image

# 5 - GET PREDECTIONS ON ALL IMAGES
def getPredection(boxes, model):
    result = []
    for image in boxes:
        if not contains_digit(image):
            result.append(0)
            continue
        # Prepare image for model
        img = prepare_image(image)
        # adding a dimension to the image
        img = tf.reshape(img, (1, 28, 28, 1))

        # Get prediction
        predictions = model(img)
        classIndex = np.argmax(predictions)
        probabilityValue = np.amax(predictions)

        # Apply dynamic threshold
        min_confidence = 0.6  # Lowered from 0.8
        if probabilityValue > min_confidence:
            result.append(classIndex)
        else:
            # Try additional processing for uncertain cases
            result.append(0)
    return result


#### 6 -  TO DISPLAY THE SOLUTION ON THE IMAGE
def displayNumbers(img,numbers,color = (0,255,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                 cv.putText(img, str(numbers[(y*9)+x]),
                               (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv.LINE_AA)
    return img

# 7 - DRAW GRID TO SEE THE WARP PRESPECTIVE EFFICENCY (OPTIONAL)
def drawGrid(img):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range (0,9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv.line(img, pt1, pt2, (255, 255, 0),2)
        cv.line(img, pt3, pt4, (255, 255, 0),2)
    return img


# 8 - TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray,scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    return ver