import cv2
import numpy as np
from PIL import Image
import os

def remove_dots(img):
    #img = cv2.imread(file1, 0)
    _, blackAndWhite = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(blackAndWhite, None, None, None, 8, cv2.CV_32S)
    sizes = stats[1:, -1] #get CC_STAT_AREA component
    img2 = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if sizes[i] >= 50:   #filter small dotted regions
            img2[labels == i + 1] = 255

    res = cv2.bitwise_not(img2)
    result = cv2.imwrite(file1, res)
    return result


def merge_images(image1, image2):
    (width1, height1) = image1.size
    (width2, height2) = image2.size

    # result_width = width1 + width2
    result_width = width1 *2
    # result_height = max(height1, height2)

    result_height = height1
    print (height2)
    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(height1,0))
    result = result.resize((512,256), Image.ANTIALIAS)
    return result




def facecrop(image):
    facedata = "haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image)

    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    for f in faces:
        x, y, w, h = [ v for v in f ]
        # cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y-100:y+h+100, x-100:x+w+100]
        return sub_face


# sketch_dir = "result/face_emotions/"
# images = os.listdir(sketch_dir)
# remove_dots("t00002.jpg")
# merge_images("00002.jpg","t00002.jpg")
# image_resize("result/00002.jpg")

# source_dir = "output/"
# dir = os.listdir(source_dir)
# for i in dir:
#     res = remove_dots(source_dir+i)
#     print(res)
