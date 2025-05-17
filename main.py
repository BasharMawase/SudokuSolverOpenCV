print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utlis import *
import SudokuSolver

pathImage = "/Users/basharmawase/Desktop/Screenshot 2025-05-17 at 22.08.10.png"
heightImg = 450
widthImg = 450
model = initializePredictionModel()  # Load the CNN Model

# 1.Preass the image
img = cv.imread(pathImage)
img = cv.resize(img, (widthImg, heightImg))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
imgThreshold = preProcess(img)

# 2.Find all contours
imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv.findContours(imgThreshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
cv.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # DRAW ALL DETECTED CONTOURS

# 3.Find the biggest countour
biggest, maxArea = biggestContour(contours) # Find the biggest countour
print(biggest)
if biggest.size != 0:
    biggest = reorder(biggest)
    print(biggest)
    cv.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25) # Draw the biggest countour
    pts1 = np.float32(biggest) # Prepare points for Warp
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # Preopare points for Warp
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv.cvtColor(imgWarpColored,cv.COLOR_BGR2GRAY)

# 4. Split the image and find digits from the image
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)
    print(len(boxes))

    numbers = getPredection(boxes, model)
    print(numbers)
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)
    print(posArray)

# 5. Find the solution of the detected sudoku
    board = np.array_split(numbers,9)
    print(board)
    try:
        SudokuSolver.solve(board)
    except:
        pass
    print(board)
    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers =flatList*posArray
    imgSolvedDigits= displayNumbers(imgSolvedDigits,solvedNumbers)

# 6. OVERLAY SOLUTION
    pts2 = np.float32(biggest) # PREPARE POINTS FOR WARP
    pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv.getPerspectiveTransform(pts1, pts2)  # GER
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
    inv_perspective = cv.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    imgDetectedDigits = drawGrid(imgDetectedDigits)
    imgSolvedDigits = drawGrid(imgSolvedDigits)

    imageArray = ([img,imgThreshold,imgContours, imgBigContour],
                  [imgDetectedDigits, imgSolvedDigits,imgInvWarpColored,inv_perspective])
    stackedImage = stackImages(imageArray, 1)
    cv.imshow('Stacked Images', stackedImage)

else:
    print("No Sudoku Found")

cv.waitKey(0)

