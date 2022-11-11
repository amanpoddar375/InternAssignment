import cv2
import numpy as np
import math

''' -------Function to display 3 windows in each row--------- '''

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


''' -------Function returns dictionary of information of line and rectangle contours--------- '''

def getContours(img, imgContour, contourType):
    if contourType == "line":
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    # print(hierarchy)
    rectangleContours = []
    lineContours = []
    for i in range(0, len(contours)):
        peri = cv2.arcLength(contours[i], True)
        # print(peri)
        approx = cv2.approxPolyDP(contours[i], 0.02 * peri, True)
        # print(len(approx))
        area = cv2.contourArea(contours[i])

        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        # print(box)
        box = np.int0(box)
        # print(box)
        center = rect[0]
        size = rect[1]
        angle = rect[2]
        # print(f"Contour- {i}\nCenter = {center}\nSize = {size}\nAngle = {angle}\nArea = {area}\n")

        '''Trying to Separate Contours that are actually a line and Stored in list of dictionary named as lineContours[]'''

        if contourType == "line":
            if area >= 5:
                if size[0] < 20 or size[1] < 20:
                    print(f"Contour {i} may be line.")
                    cv2.drawContours(imgContour, [box], 0, (0, 0, 255), 2)
                    line_dict = {'contour': i,
                                 'center': center,
                                 'size': size,
                                 'angle': angle,
                                 'area': area,
                                 'box': box}
                    lineContours.append(line_dict)
        else:
            x_start = box[3][0]
            y_start = box[3][1]
            a = box[0][0]
            b = box[0][1]
            c = box[2][0]
            d = box[2][1]
            d1 = math.sqrt((x_start - a) ** 2 + (y_start - b) ** 2)
            d2 = math.sqrt((x_start - c) ** 2 + (y_start - d) ** 2)
            if d1 >= d2:
                x_end = x_start - int(size[0])
                y_end = y_start - int(size[1])
            else:
                x_end = x_start + int(size[1])
                y_end = y_start - int(size[0])

            cv2.drawContours(imgContour, [box], 0, (0, 0, 255), 3)
            rect_dict = {'contour': i,
                         'center': center,
                         'size': size,
                         'angle': angle,
                         'area': area,
                         'box': box,
                         'BBpoints': [x_start, y_start, x_end, y_end]}
            rectangleContours.append(rect_dict)

    if contourType == "line":
        print(len(lineContours))
        similarLineList = []
        while len(lineContours) != 0:
            similarLine = []
            line = lineContours[0]
            similarLine.append(line)
            for j in range(1, len(lineContours)):
                if j < len(lineContours):
                    if abs(line['center'][0] - lineContours[j]['center'][0] < 10 and line['center'][1] -
                           lineContours[j]['center'][1] < 10.0):
                        if abs(line['angle'] - lineContours[j]['angle']) < 1.0:
                            if abs(line['area'] - lineContours[j]['area']) < 175:
                                similarLine.append(lineContours[j])
                                print(f"Contours- {line['contour']} and {lineContours[j]['contour']} are same !!!")
                                del lineContours[j]
                                print(f"Length of lineContours : {len(lineContours)}")
            del lineContours[0]
            similarLineList.extend([similarLine])
        # print(len(similarLine))
        # print(similarLineList[0])
        for similarLines in similarLineList:
            if len(similarLines) > 1:
                if similarLines[0]['area'] < similarLines[1]['area']:
                    lineContours.append(similarLines[0])
                    print(f"Line {similarLines[0]['contour']} is added to Line Contours")
                else:
                    lineContours.append(similarLines[1])
                    print(f"Line {similarLines[1]['contour']} is added to Line Contours")

            else:
                lineContours.append(similarLines[0])
                print(f"Line {similarLines[0]['contour']} is added to Line Contours")
        return lineContours

    else:
        return rectangleContours


img = cv2.imread('shapedetector.jpg')
original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.putText(original_img, "Original RGB Image", (120, 28), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("Original Image", original_img)

blankImg = cv2.imread('blank.jpg')
imgContour = img.copy()

imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

imgCanny = cv2.Canny(imgGray, 20, 23)

lineContours = getContours(imgCanny, imgContour, "line")
# print(len(lineContours))
rectangleContours = getContours(imgCanny, imgContour, "rectangle")

rotatedImgContour = img.copy()

for rectContour in rectangleContours:
    cv2.rectangle(rotatedImgContour, (rectContour['BBpoints'][0], rectContour['BBpoints'][1]), (rectContour['BBpoints'][2], rectContour['BBpoints'][3]), (219, 67, 119), 3)
    cv2.rectangle(blankImg, (rectContour['BBpoints'][0], rectContour['BBpoints'][1]),
                  (rectContour['BBpoints'][2], rectContour['BBpoints'][3]), (0, 0, 0), 2)


for lineContour in lineContours:
    print(lineContour['size'])
    if lineContour['size'][0] > lineContour['size'][1]:
        lineContour['length'] = lineContour['size'][0]
        lineContour['width'] = lineContour['size'][1]
    else:
        lineContour['length'] = lineContour['size'][1]
        lineContour['width'] = lineContour['size'][0]



def findPerpendicularDist(x1, y1, x2, y2, x, y):
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    pd = abs((A * x + B * y + C)) / (math.sqrt(A * A + B * B))
    return pd


for i in range(0, len(lineContours)):
    (linePoint_x, linePoint_y) = lineContours[i]['box'][3]
    (rect_x1, rect_y1) = rectangleContours[i]['box'][3]
    (rect_x2, rect_y2) = rectangleContours[i]['box'][0]
    (rect_x3, rect_y3) = rectangleContours[i]['box'][2]

    pd1 = findPerpendicularDist(rect_x1, rect_y1, rect_x2, rect_y2, linePoint_x, linePoint_y)
    pd2 = findPerpendicularDist(rect_x1, rect_y1, rect_x3, rect_y3, linePoint_x, linePoint_y)
    pd1 = round(pd1, 0)
    pd2 = round(pd2, 0)

    box = rectangleContours[i]['box']
    x_start = box[3][0]
    y_start = box[3][1]
    length = round(lineContours[i]['length'], 0)
    # width = round(lineContours[i]['width'], 0)
    a = box[0][0]
    b = box[0][1]
    c = box[2][0]
    d = box[2][1]
    d1 = math.sqrt((x_start - a) ** 2 + (y_start - b) ** 2)
    d2 = math.sqrt((x_start - c) ** 2 + (y_start - d) ** 2)
    if d1 >= d2:
        x_start = x_start - pd2
        x_end = x_start - length
        y_start = y_start - pd1
        y_end = y_start
    else:
        x_start = x_start + pd1
        x_end = x_start + length
        y_start = y_start - pd2
        y_end = y_start
    # print(int(x_start), y_start, x_end, y_end)

    cv2.rectangle(rotatedImgContour, (int(x_start), int(y_start)), (int(x_end), int(y_end)), (219, 67, 119), 2)

    cv2.putText(rotatedImgContour, "Rectangle Alligned Image", (120, 28), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.rectangle(blankImg, (int(x_start), int(y_start)), (int(x_end), int(y_end)), (0, 0, 0), 2)
    cv2.putText(blankImg, "Rectangle Alligned Image", (120, 28), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Rectangular Allignment Window", blankImg)




cv2.putText(img, "BGR Image", (120, 28),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
cv2.putText(imgBlur, "Blur Image", (120, 28), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
cv2.putText(imgGray, "Gray Image", (120, 28), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
cv2.putText(imgCanny, "Canny Image", (120, 28),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
cv2.putText(imgContour, "Minimum Area Bounded", (120, 28), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

imgStacked = stackImages(0.8, ([img, imgBlur, imgGray],
                               [imgCanny, imgContour, rotatedImgContour]))


cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.imshow("Result", imgStacked)
cv2.waitKey(0)

print("Success!!!")