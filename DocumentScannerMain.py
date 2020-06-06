import cv2
import numpy as np
import utlis
import pytesseract

webCam = True
imageLocation = "5.jpeg"
cap = cv2.VideoCapture(1)
cap.set(10,160)
heightImg = 480
widthImg = 480

utlis.initializeTrackbars()
count=0

while True:

    imgBlank = np.zeros((heightImg,widthImg,3),np.uint8)

    if webCam:success, img = cap.read()
    else:img = cv2.imread(imageLocation)
    img = cv2.imread(imageLocation)
    img = cv2.resize(img,(widthImg, heightImg))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convets the image to grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  #converts the grayscale image to blured image
    thres=utlis.valTrackbars() #using this function we get the trackbars which help us adjust the threshold
    imgThreshold = cv2.Canny(imgBlur,thres[0],thres[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION

    ## FIND ALL COUNTOURS
    imgConts = img.copy()
    imgBigConts = img.copy()
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    cv2.drawContours(imgConts, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS


    # FIND THE LARGEST COUNTOUR IN THE FRAME
    big, maxArea = utlis.biggestContour(contours) # FIND THE BIGGEST CONTOUR
    if big.size != 0:
        big=utlis.reorder(big)
        cv2.drawContours(imgBigConts, big, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
        imgBigConts = utlis.drawRectangle(imgBigConts,big,2)
        pts1 = np.float32(big) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        #REMOVE EXTRA UNWANTED PIXELS FROM THE SIDES
        imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))

        # APPLY ADAPTIVE THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)

        # Image Array for Display
        imageArray = ([img,imgGray,imgThreshold,imgConts],
                      [imgBigConts,imgWarpColored, imgWarpGray,imgAdaptiveThre])

    else:
        imageArray = ([img,imgGray,imgThreshold,imgConts],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    # LABELS FOR DISPLAY
    lables = [["Original","Gray","Threshold","Contours"],
              ["Biggest Contour","Warp Prespective","Warp Gray","Adaptive Threshold"]]

    stackedImage = utlis.stackImages(imageArray,0.75,lables)
    cv2.imshow("Result",stackedImage)

    # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage"+str(count)+".jpg",imgWarpColored)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        img = cv2.imread("Scanned/myImage"+str(count)+".jpg")
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        adaptive_threshold = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 111,
                                                   11)

        text = pytesseract.image_to_string(adaptive_threshold)
        print(text)
        # cv2.imshow("grey",grey)
        # cv2.imshow("adaptive_th",adaptive_threshold)
        cv2.waitKey(0)
        cv2.waitKey(300)
        count += 1



