import cv2
import numpy as np
import math

frameWidth = 640
frameHeight = 480

def empty(a):
    pass

# cv2.namedWindow("Parameters")
# cv2.resizeWindow("Parameters", 640, 240)
# cv2.createTrackbar("Threshold1", "Parameters", 23, 255, empty)
# cv2.createTrackbar("Threshold2", "Parameters", 20, 255, empty)


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

# def crop_rect(img, rect):
#     # get the parameter of the small rectangle
#     center = rect[0]
#     size = rect[1]
#     angle = rect[2]
#     center, size = tuple(map(int, center)), tuple(map(int, size))
#
#     # get row and col num in img
#     height, width = img.shape[0], img.shape[1]
#     print("width: {}, height: {}".format(width, height))
#
#     M = cv2.getRotationMatrix2D(center, angle, 1)
#     img_rot = cv2.warpAffine(img, M, (width, height))
#
#     img_crop = cv2.getRectSubPix(img_rot, size, center)
#
#     return img_crop, img_rot


def getContours(img, imgContour, contourType):
    if contourType == "line":
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(imgContour, contours, -1, (255, 0, 255), 4)
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

            cv2.drawContours(imgContour, [box], 0, (0, 0, 255), 2)
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

        # pic = cv2.drawContours(imgContour, contours[i], -1, (255, 0, 255), 4)
        # picname = f'contour-{i}.png'
        # cv2.imwrite(picname, pic)




        # img_crop, img_rot = crop_rect(imgContour, rect)
        # peri_rot = cv2.arcLength(img_rot, True)
        # approx_rot = cv2.approxPolyDP(img_rot, 0.02 * peri_rot, True)
        # x, y, w, h = cv2.boundingRect(approx_rot)
        # cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)
        '''cv2.rectangle(image, start_point, end_point, color, thickness)'''

        # print(x, y, w, h)
        # cv2.rectangle(imgContour, (x, y), (x+w, y+h), (0, 255, 0), 5)

        '''
           True means Contour is Closed. Peri will have length of the each Contours. We will use this  
           Parameter peri to approximate what type of shape it is. 
           In order to do that, we will use approximation of Poly method and we will input our contour and give resilution 
           and define again that it is closed contour.
           This approx array will have a certain amount of points and based on these, we can determine shape of Polynomials.
        '''
        ## Creating Bounding Boxes...






    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     if area > 10000:
    #         cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 4)



    '''
      RETR_EXTERN method retrieves only the extreme or outer contours.
      Another method called Tree which retrieves all the contours and reconstruct full hierachy.
      CHAIN_APPROX_NONE helps to provide all the stored contour points.
      But if we use other one which is SIMPLE that will compress the values and get lesser no of points only of endpoints.
      
      Also, in order to remove any noisy contours, we can filter out those based on Area of Contours.
    '''




'''This code was for web cam video ...'''

# cap = cv2.VideoCapture(0)
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)
# while True:
#     success, img = cap.read()
#     if success:
#         imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
#         imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
#
#         threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
#         threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
#
#         imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
#
#         imgStacked = stackImages(0.8, ([img, imgGray, imgCanny]))
#
#     cv2.imshow("Result", imgStacked)
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
# cap.release()
# cv2.destroyAllWindows()
# print("Success!!!")


img = cv2.imread('shapedetector.jpg')
imgContour = img.copy()

imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)


imgCanny = cv2.Canny(imgGray, 20, 23)
kernel = np.ones((5, 5))
# imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
lineContours = getContours(imgCanny, imgContour, "line")
rectangleContours = getContours(imgCanny, imgContour, "rectangle")
rotatedImgContour = imgContour.copy()
numberedImgContour = imgContour.copy()
print(len(lineContours))
sorted_index =[]
i=0
for lineContour in lineContours:
    print(lineContour['size'])
    if lineContour['size'][0] > lineContour['size'][1]:
        lineContour['length'] = lineContour['size'][0]
        lineContour['width'] = lineContour['size'][1]
    else:
        lineContour['length'] = lineContour['size'][1]
        lineContour['width'] = lineContour['size'][0]
    dict = {'index': i,
            'length': lineContour['length'],
            'width': lineContour['width']}
    sorted_index.append(dict)
    i = i+1
# print(lineContours)
temp = sorted_index.copy()
# print(f"Before sorting,\n {sorted_index}")
sorted_index = sorted(sorted_index, key = lambda d: d['length'])
# print(f"After sorting in ascending order,\n {sorted_index}")
for rectContour in rectangleContours:
    cv2.rectangle(rotatedImgContour, (rectContour['BBpoints'][0], rectContour['BBpoints'][1]), (rectContour['BBpoints'][2], rectContour['BBpoints'][3]), (0, 255, 0), 5)


'''Assigning Numbers to Rectangles...'''
# numberedImgContour = cv2.cvtColor(numberedImgContour, cv2.COLOR_BGR2RGB)
for i in range(0, len(sorted_index)):
    index = sorted_index[i]['index']
    length = sorted_index[i]['length']
    x = int(rectangleContours[index]['center'][0])
    y = int(rectangleContours[index]['center'][1])
    # print(x, y)
    cv2.putText(numberedImgContour, f"R-{i+1}", (x-100, y+100), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(numberedImgContour, f"Length: {round(length, 2)}", (x - 100, y + 130),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


def findPerpendicularDist(x1, y1, x2, y2, x, y):
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    d = abs((A * x + B * y + C)) / (math.sqrt(A * A + B * B))
    return d
distList = []
for i in range(0, len(lineContours)):
    (linePoint_x, linePoint_y) = lineContours[i]['box'][3]
    (rect_x1, rect_y1) = rectangleContours[i]['box'][3]
    (rect_x2, rect_y2) = rectangleContours[i]['box'][0]
    (rect_x3, rect_y3) = rectangleContours[i]['box'][2]
    pd1 = findPerpendicularDist(rect_x1, rect_y1, rect_x2, rect_y2, linePoint_x, linePoint_y)
    pd2 = findPerpendicularDist(rect_x1, rect_y1, rect_x3, rect_y3, linePoint_x, linePoint_y)
    pd1 = round(pd1, 0)
    pd2 = round(pd2, 0)
    # distList.append((pd1, pd2))

    box = rectangleContours[i]['box']
    x_start = box[3][0]
    y_start = box[3][1]
    length = round(temp[i]['length'], 0)
    width = round(temp[i]['width'], 0)
    # a = box[0][0]
    # b = box[0][1]
    # c = box[2][0]
    # d = box[2][1]
    # d1 = math.sqrt((x_start - a) ** 2 + (y_start - b) ** 2)
    # d2 = math.sqrt((x_start - c) ** 2 + (y_start - d) ** 2)
    if pd1 >= pd2:
        x_start = x_start + pd1
        x_end = x_start + length
        y_start = y_start - pd2
        y_end = y_start - width
    else:
        x_start = x_start - pd1
        x_end = x_start - length
        y_start = y_start - pd2
        y_end = y_start - width
    print(int(x_start), y_start, x_end, y_end)
    cv2.rectangle(rotatedImgContour, (int(x_start), int(y_start)), (int(x_end), int(y_end)), (0, 255, 0), 3)

# print(distList)



imgStacked = stackImages(0.8, ([img, imgGray, imgCanny],
                               [imgContour, numberedImgContour, rotatedImgContour]))

cv2.imshow("Result", imgStacked)
cv2.waitKey(0)


print("Success!!!")