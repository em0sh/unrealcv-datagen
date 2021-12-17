# Standard libraries
from io import BytesIO
import math

import plot

# Libraries
import numpy as np
from PIL import Image

# Modules
import io

# TODO: This parameter needs to be tuned for each scene. This should be 
# Distance beyond the object being measured for depth that should be ignored (skybox)
maxDist = 25


def imReadB(im_file):
	''' Read image as a 8-bit numpy array '''
	im = np.asarray(Image.open(im_file))
	return(im)

def readPNG(res):
	''' Read image and return a numpy array '''
	img = Image.open(BytesIO(res))
	return(np.asarray(img))

def get(client, build, w, h):
	''' Retrieve commands from client and returns two arrays
		resImage - light image
		resDepth - depth image
	'''

	# Generate Depth image
	preResDepth = client.request('vget /camera/0/depth npy')
	resDepth = np.load(BytesIO(preResDepth))
    
	# This function "cleans" the depth data and returns it
	resDepth = np.ndarray.flatten(resDepth)
	resDepth = cleanDepth(resDepth)

	resImagePNG = client.request('vget /camera/0/lit png')

	resImageArrayBytes = client.request('vget /camera/0/lit npy')

	resImage = Image.open(io.BytesIO(resImagePNG))
	resImage = resImage.convert('L')
	resImageArray = np.array(resImage)
	resImageArray = np.ndarray.flatten(resImageArray)

	return(resImageArray, resDepth)

def bwImg(payload, w, h):

	plot.show(payload)

	bwImg = payload

	return(bwImg)

def cleanDepth(payload):

	# TODO: Change this from an element wise function to an array (np) function for increase speed

	# Array to hold results before returning to calling function
	tempArray = []

	for i in payload:

		if i > maxDist:
			tempArray.append(0.)
		else:
			tempArray.append(i)
	

	return(tempArray)
