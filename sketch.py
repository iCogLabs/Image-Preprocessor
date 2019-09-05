from PIL import Image, ImageEnhance, ImageFilter

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from pylab import *

from scipy.ndimage import filters

from scipy.misc import imsave

import glob, os
import sys, getopt
import argparse

import cv2
import numpy as np
#import clean

class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)

def merge_images(image1, image2):
    (width1, height1) = image1.size
    (width2, height2) = image2.size

    # result_width = width1 + width2
    result_width = width1 *2
    # result_height = max(height1, height2)

    result_height = height1
    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(height1,0))
    result = result.resize((512,256), Image.ANTIALIAS)
    return result

def remove_dots(image, val):

    gray_im = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    _, blackAndWhite = cv2.threshold(gray_im, 127, 255, cv2.THRESH_BINARY_INV)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(blackAndWhite, None, None, None, 8, cv2.CV_32S)
    sizes = stats[1:, -1] #get CC_STAT_AREA component
    img2 = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if sizes[i] >= val:   #filter small dotted regions
            img2[labels == i + 1] = 255

    res = cv2.bitwise_not(img2)
    #if not os.path.exists("RemoveDot"): os.mkdir("RemoveDot")
    #result = imsave(os.path.join("RemoveDot", 't' + os.path.basename(image)), res)
    return res

def facecrop(image, face_crop):

    facedata = "haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(facedata)
    img = cv2.imread(image)
    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    for f in faces:
        x, y, w, h = [ v for v in f ]
        # cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y-face_crop:y+h+face_crop, x-face_crop:x+w+face_crop] #face_crop = 100
        #if not os.path.exists("FACECROP"): os.mkdir("FACECROP")
        #imsave(os.path.join("FACECROP", 't' + os.path.basename(image)), sub_face)
        imsave(image, sub_face)
        #sub_face.save(os.path.join("FACECROP", 't' + os.path.basename(image))) #t
        return image
        #return "FACECROP", 't' + os.path.basename(image)

def start(mat):
    w, h, c = mat.shape
    ws = int(w * random(1))
    hs = int(h * random(1))
    while (mat[ws, hs, 0] > 254) and (mat[ws, hs, 1] > 254) and (mat[ws, hs, 2] > 254):
        ws = int(w * random(1))
        hs = int(h * random(1))
    return ws, hs


def gen_color_line(mat, dir,max_length,max_dif):
    w, h, c = mat.shape
    ws = int(w * random(1))
    hs = int(h * random(1))
    while (mat[ws, hs, 0] > 254) and (mat[ws, hs, 1] > 254) and (mat[ws, hs, 2] > 254):
        ws = int(w * random(1))
        hs = int(h * random(1))
    if dir == 1:
        wt = ws
        ht = hs
        while (wt < w - 1) and (abs(int(mat[wt, ht, 1]) + int(mat[wt, ht, 2]) + int(mat[wt, ht, 0]) - (
                        int(mat[wt + 1, ht, 1]) + int(mat[wt + 1, ht, 2]) + int(mat[wt + 1, ht, 0]))) < 80):
            wt = wt + 1
    if dir == 2:
        wt = ws
        ht = hs
        while (ht < h - 1) and (abs(int(mat[wt, ht, 1]) + int(mat[wt, ht, 2]) + int(mat[wt, ht, 0]) - (
                        int(mat[wt, ht + 1, 1]) + int(mat[wt, ht + 1, 2]) + int(mat[wt, ht + 1, 0]))) < 3):
            ht = ht + 1
    if dir == 3:
        wt = ws
        ht = hs
        length = 0
        while (length < max_length) and (wt < w-1) and (ht < h-1) and (
            abs(int(mat[wt, ht, 1]) + int(mat[wt, ht, 2]) + int(mat[wt, ht, 0]) - (
                            int(mat[wt + 1, ht + 1, 1]) + int(mat[wt + 1, ht + 1, 2]) + int(
                        mat[wt + 1, ht + 1, 0]))) < max_dif):
            ht += 1
            wt += 1
            length = abs(wt - ws) + abs(ht - hs)
    return ws, hs, wt, ht, length

def save_combined(im, path, filename):

    wsize = 512  # double the resolution 1024
    w, h = im.size
    hsize = int(h * wsize / float(w))
    im_ext = [".jpg", ".jpeg", ".png"]

    #
    if not path.endswith(tuple(im_ext)):
        path = os.path.join(path, filename)

    if hsize * 2 > wsize:  # crop to three
        im = im.resize((wsize, hsize))
        bounds1 = (0, 0, wsize, int(wsize / 2)) #/2
        cropImg1 = im.crop(bounds1)
        # cropImg1.show()
        ###
        #if not os.path.exists(path): os.mkdir(path)
        #print("PATH:::", path)
        cropImg1.save(path)
        bounds2 = (0, hsize - int(wsize / 2), wsize, hsize) #wsize/2

    else:
        #if not os.path.exists(path): os.mkdir(path)
        im = im.resize((wsize // 2, (wsize // 4)))
        ###
        im.save(path)#t

    print('concat image saved')

def sketch(im, color_pic, filename):
    Gamma = 0.97 #0.97
    Phi = 200
    Epsilon = 0.5 #0.5
    k = 2
    Sigma = 1.5

    im = np.array(ImageEnhance.Sharpness(im).enhance(5.0)) #3 neber
    im2 = filters.gaussian_filter(im, Sigma)
    im3 = filters.gaussian_filter(im, Sigma * k)
    differencedIm2 = im2 - (Gamma * im3)
    (x, y) = np.shape(im2)
    for i in range(x):
        for j in range(y):
            if differencedIm2[i, j] < Epsilon:
                differencedIm2[i, j] = 1
            else:
                differencedIm2[i, j] = 250 + np.tanh(Phi * (differencedIm2[i, j]))

    gray_pic = differencedIm2.astype(np.uint8)

    org_pic = np.atleast_2d(color_pic)

    if org_pic.ndim == 2:
        org_pic = np.stack((org_pic, org_pic, org_pic),axis=2)

    if org_pic.ndim == 3:
        w, h, c = org_pic.shape
        if c>0:
            image = color_pic.filter(MyGaussianBlur(radius=5))
            mat = np.atleast_2d(image)

            if gray_pic.ndim == 2:
                gray_pic = np.expand_dims(gray_pic, 2)
                gray_pic = np.tile(gray_pic, [1, 1, c]) # last one 3

            return gray_pic, org_pic

def save_gen(gen, sketch, filename, removedots):

    sketch.save(os.path.join(gen, 't' + filename))
    print('gray image', os.path.join(gen, 't' + filename), " saved")
    return sketch

def save_orgtogen(gray_pic, org_pic, orgtogen, filename, sketch, removedots):

    combined_pic = np.append(org_pic, gray_pic, axis=1)
    concat_img = Image.fromarray(combined_pic)
    save_combined(concat_img, orgtogen, filename)
    return concat_img

def save_gentoorg(gray_pic, org_pic, gentoorg, filename, sketch, removedots):

    combined_pic = np.append(gray_pic, org_pic, axis=1)
    concat_img = Image.fromarray(combined_pic)
    save_combined(concat_img, gentoorg, filename)
    return concat_img

def save_results(im, color_pic, filename, gen, orgtogen, gentoorg, removedots):

    gray_pic, org_pic = sketch(im, color_pic, filename)
    if removedots:
        if not os.path.exists("gray"): os.mkdir("gray")
        gray = imsave(os.path.join("gray", 't' + os.path.basename(filename)), gray_pic)
        gray_pic = remove_dots(os.path.join('gray', 't' + os.path.basename(filename)), removedots)
        gray = imsave(os.path.join("gray", 't' + os.path.basename(filename)), gray_pic)
        gray_img = Image.open(filename)
        org_img = Image.open(os.path.join("gray", 't' + os.path.basename(filename)))
        ###
        im_ext = [".jpg", ".jpeg", ".png"]

        #
        #if not path.endswith(tuple(im_ext)):
        if gen:
            if gen.endswith(tuple(im_ext)):
                imsave(gen, gray_pic)
            else:
                if not os.path.exists(gen): os.mkdir(gen)
                imsave(os.path.join(gen, os.path.basename(filename)), gray_pic)
            print('gray image', os.path.join(gen, os.path.basename(filename)), " saved")
        if orgtogen:
            merged_im = merge_images(gray_img, org_img)
            if orgtogen.endswith(tuple(im_ext)):
                imsave(orgtogen, merged_im)
            else:
                if not os.path.exists(orgtogen): os.mkdir(orgtogen)
                imsave(os.path.join(orgtogen, os.path.basename(filename)), merged_im)
            print('concat (orgtogen) image saved', os.path.join(orgtogen, os.path.basename(filename)))
        if gentoorg:
            merged_im = merge_images(org_img, gray_img)
            if gentoorg.endswith(tuple(im_ext)):
                imsave(gentoorg, merged_im)
            else:
                if not os.path.exists(gentoorg): os.mkdir(gentoorg)
                imsave(os.path.join(gentoorg, os.path.basename(filename)), merged_im)
            print('concat (gentoorg) image saved', os.path.join(gentoorg, os.path.basename(filename)))
    else:
        sketch_pic = Image.fromarray(gray_pic, mode = 'RGB')

        if gen:
            if not os.path.exists(gen): os.mkdir(gen)
            save_gen(gen, sketch_pic, os.path.basename(filename), removedots)
        if orgtogen:
            if not os.path.exists(orgtogen): os.mkdir(orgtogen)
            save_orgtogen(gray_pic, org_pic, orgtogen, os.path.basename(filename), sketch_pic, removedots)
        if gentoorg:
            if not os.path.exists(gentoorg): os.mkdir(gentoorg)
            save_gentoorg(gray_pic, org_pic, gentoorg, os.path.basename(filename), sketch_pic, removedots)

def main(args):

    # args values
    input_dir = args.input_dir
    gen = args.gen
    orgtogen = args.orgtogen
    gentoorg = args.gentoorg
    input_image = args.input_image
    face_crop = args.facecrop
    removedots = args.remove_dots

    #parameter
    max_length=20
    min_length=10
    max_dif=30
    n_point=50
    dir = 3

    if input_image:
        #filepath, filename = os.path.split(files1)
        if not os.path.exists(input_image): os.mkdir(input_image)
        filename = input_image
        cropped_im = facecrop(filename, face_crop)
        #remove_dots(files1)
        im = Image.open(filename).convert('L')
        color_pic = Image.open(cropped_im)

        save_results(im, color_pic, filename, gen, orgtogen, gentoorg, removedots)

    if input_dir:
        input_paths = glob.glob(input_dir+ '/*.jpg')
        input_paths+=(glob.glob(input_dir+ '/*.jpeg'))
        input_paths+=(glob.glob(input_dir + '/*.png'))

        for files1 in input_paths:
            filepath, filename = os.path.split(files1)
            if face_crop:
                cropped_im = facecrop(files1, face_crop)
                print("Face cropped saved:", files1)
            im = Image.open(files1).convert('L')
            color_pic = Image.open(files1)
            #remove_dots(im)
            filename = os.path.join(input_dir, filename)
            save_results(im, color_pic, filename, gen, orgtogen, gentoorg, removedots)

    if not input_dir and not input_image:
         print(parser.print_help(sys.stderr))
         sys.exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--input_image', type=str)
    parser.add_argument('--gen', type=str)#, default="output")#, action="store_true")
    parser.add_argument('--orgtogen', type=str)
    parser.add_argument('--facecrop', type=int)
    parser.add_argument('--gentoorg', type=str)#, nargs='?')
    parser.add_argument('--remove_dots', type=int)#action='store_true')
    args = parser.parse_args()
    main(args)
