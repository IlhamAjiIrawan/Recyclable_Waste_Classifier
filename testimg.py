import os
import cvzone
from cvzone.ClassificationModule import Classifier
import cv2

# Ganti video capture dengan membaca gambar statis
img = cv2.imread('Resources/Waste/7.png')  # Gantilah dengan path gambar yang kamu inginkan
if img is None:
    print("Error: Could not load the image.")
    exit()

print("Image loaded successfully.")

# Load the classifier
classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')
print("Classifier loaded.")

imgArrow = cv2.imread('Resources/arrow.png', cv2.IMREAD_UNCHANGED)
if imgArrow is None:
    print("Error: Arrow image not found.")
    exit()
else:
    print("Arrow image loaded successfully.")

classIDBin = 0

# Import all the waste images
imgWasteList = []
pathFolderWaste = "Resources/Waste"
print(f"Loading waste images from {pathFolderWaste}")
pathList = os.listdir(pathFolderWaste)
for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))

print(f"Loaded {len(imgWasteList)} waste images.")

# Import all the waste bin images
imgBinsList = []
pathFolderBins = "Resources/Bins"
print(f"Loading bin images from {pathFolderBins}")
pathList = os.listdir(pathFolderBins)
for path in pathList:
    imgBinsList.append(cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED))

print(f"Loaded {len(imgBinsList)} bin images.")

# Waste class dictionary
classDic = {0: None,
            1: 0,
            2: 0,
            3: 3,
            4: 3,
            5: 1,
            6: 1,
            7: 2,
            8: 2}

# Resize the image for the background
imgResize = cv2.resize(img, (454, 340))

imgBackground = cv2.imread('Resources/background.png')
if imgBackground is None:
    print("Error: Background image not found.")
    exit()
else:
    print("Background image loaded successfully.")

prediction = classifier.getPrediction(img)
print(f"Prediction: {prediction}")

classID = prediction[1]
print(f"Predicted class ID: {classID}")

if classID != 0:
    if classID - 1 < len(imgWasteList):
        imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID - 1], (909, 127))
        print(f"Overlaid waste image for class ID {classID}.")
    else:
        print(f"Warning: classID {classID} is out of bounds for waste images.")

    imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978, 320))
    print("Overlayed arrow.")

    classIDBin = classDic.get(classID, 0)
    print(f"Bin class ID: {classIDBin}")

imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[classIDBin], (895, 374))
print("Overlayed bin image.")

imgBackground[148:148 + 340, 159:159 + 454] = imgResize
print("Placed resized image on background.")

# Display the final output image
cv2.imshow("Output", imgBackground)

# Wait for user to press any key before closing
cv2.waitKey(0)
cv2.destroyAllWindows()
