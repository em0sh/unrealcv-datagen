# Standard libraries
#from __future__ import division, absolute_import, print_function
import os, argparse, sys

# Modules
import render, plot, scene

# libraries for opening and manipulating exr files
import numpy as np
import cv2, unrealcv

# ---------------- Parameters
# xRes and yRes set in server client configuration (see response from status for information)
xRes = 100
yRes = 100

# default number of iterations to run
runLength = 360
# Percentage of runLength the test size will be
testSize = .01
# Angle and positional increment for moving items in rendering
inc = 1

# ---------------- Argument handling
# arguments for operation modes
build = False
plotenable = False

# set the ip address as the local machine as the default
ipaddr = '127.0.0.1'

# argparse setup
parser = argparse.ArgumentParser()

# DIAG: Remove no longer used arguments
parser.add_argument("-plot", action='store_true')
parser.add_argument("-ip", action='store', type=str)
parser.add_argument("-object", action='store', type=str)

args = parser.parse_args()

if args.plot:
	plotenable = True

if args.ip:
	ipaddr = args.ip
print('using IP: %s' % ipaddr)

if args.object:
	obj = f'/object/{args.object}'
	print('Manipulating object: %s' % obj)
else:
	print(f'Object to manipulate required - exiting')
	sys.exit(-1)

# ----------------- UnrealCV Client Connection
# Instantiate network connection to UnrealCV for generating images
client = unrealcv.Client((ipaddr, 9000))

# Establish connection with UnrealCV server
client.connect()

# Throw error if connection isn't established
if not client.isconnected():
	print('unrealcv server not found')
	sys.exit(-1)

# Print connection and server information
print(client.request('vget /unrealcv/status'))


# call render module and generate light and depth image
resimage, resdepth = render.get(client, build, xRes, yRes)

# get camera pose information 
pos = client.request('vget /camera/0/location')
rot = client.request('vget /camera/0/rotation')

# Try object - if it does not exist, manipulate the camera (NameError exception) else run the pose manipulation
	# on the specified object
# TODO: This entire implementation of testing for obj isn't ideal, need to rework this
try:
	obj

except NameError:
	print('object not declared, posing camera only')
	# instantiate pose instance based on current camera and object posing
	p1 = scene.pose(pos, rot)

else:
	# Get object pose information
	poso = client.request(f'vget {obj}/location')
	roto = client.request(f'vget {obj}/rotation')

	p1 = scene.pose(poso, roto, obj)



def buildData(num, inc):
	''' Assemble arrays of data for training '''
	# Initialize array to contain training data and labels
	dataListX = []
	dataListY = []

	for i in range(num):


		# move camera
		resImage, resDepth = scene.anim(client, p1, inc, build, xRes, yRes)


		dataListX.append(resImage)
		dataListY.append(resDepth)
			
		loopFraction = int(runLength / 10)
		if i % loopFraction == 0:
			print(f'data build {i/runLength*100}% complete')
			# Use matplotlib to show the image data as it is built
			# DIAG: Commenting this out while working with small training numbers
			# TODO: Multithread this as to not hold up the program
			
	return(dataListX, dataListY)

# Reset pose before continuing
scene.resetPose(p1, client)

# Length of test data for testing network after training
testLength = 1

# Test length computation
if int(testSize*runLength) <= 1:
	testLength = int(testSize*runLength)

# X = Light image, Y = Depth image
trainDataX, trainDataY = buildData(runLength, inc)
testDataX, testDataY = buildData(testLength, inc)
