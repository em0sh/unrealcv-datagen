# libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2

# modules
import render


def show(img):

	fi, ay = plt.subplots()
	

	#img = render.readPNG(img)

	ay.cla()
	ay.imshow(img)
	ay.set_title("ucv img")

	# plt.pause is for "video" functionality
	#plt.pause(.2)
	plt.show()

def showFlat(img, w):
    
    t = []
    t_ = []
    for k in img:
        for x in k:
            if len(t_) > (w - 1):
                t.append(t_)
                t_ = []
            t_.append(x)
        show(t)
        t = []


def showRaw(img, w):
    
    t = []
    t_ = []
    for x in img:
        if len(t_) > (w - 1):
            t.append(t_)
            t_ = []
        t_.append(x)
    show(t)
    t = []


