import os
import cvzone
from cvzone.ClassificationModule import Classifier
import cv2
import time  # Tambahkan import time

# Start the video capture
cap = cv2.VideoCapture(0)
print("Video capture started.")

if not cap.isOpened():
    print("Error: Could not open video source.")
else:
    print("Video source opened successfully.")

classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')
print("Classifier loaded.")

imgArrow = cv2.imread('Resources/arrow.png', cv2.IMREAD_UNCHANGED)
if imgArrow is None:
    print("Error: Arrow image not found.")
else:
    print("Arrow image loaded successfully.")

classIDBin = 0

# Import waste images
imgWasteList = []
pathFolderWaste = "Resources/Waste"
print(f"Loading waste images from {pathFolderWaste}")
pathList = os.listdir(pathFolderWaste)
for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))
print(f"Loaded {len(imgWasteList)} waste images.")

# Import bin images
imgBinsList = []
pathFolderBins = "Resources/Bins"
print(f"Loading bin images from {pathFolderBins}")
pathList = os.listdir(pathFolderBins)
for path in pathList:
    imgBinsList.append(cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED))
print(f"Loaded {len(imgBinsList)} bin images.")

# Waste class dictionary
classDic = {0: None, 1: 0, 2: 0, 3: 3, 4: 3, 5: 1, 6: 1, 7: 2, 8: 2}

# Variabel untuk hitung FPS
pTime = 0  # previous time

while True:
    success, img = cap.read()

    if not success:
        print("Error: Failed to capture image.")
        break

    imgResize = cv2.resize(img, (454, 340))
    imgBackground = cv2.imread('Resources/background.png')
    if imgBackground is None:
        print("Error: Background image not found.")
        break

    prediction = classifier.getPrediction(img)
    classID = prediction[1]

    if classID != 0:
        if classID - 1 < len(imgWasteList):
            imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID - 1], (909, 127))
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978, 320))
        classIDBin = classDic.get(classID, 0)

    imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[classIDBin], (895, 374))
    imgBackground[148:148 + 340, 159:159 + 454] = imgResize

    # Hitung FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime

    # Tampilkan FPS di layar
    cvzone.putTextRect(imgBackground, f'FPS: {int(fps)}', (50, 50), scale=2, thickness=2, colorT=(255,255,255), colorR=(0,0,0), offset=10)

    # Tampilkan jendela hasil
    cv2.imshow("Output", imgBackground)
    cv2.waitKey(1)

print("Closing the video capture.")
cap.release()
cv2.destroyAllWindows()
