# UnrealCV-datagen
UnrealCV is a library providing an interface to the Unreal 4 engine through Python.
![Example Engine Image](/data/UCV1.png)
![Example Prediction Image](/data/UCV2.png)

Users can quickly use UnrealCV-datagen to access UE4's 3D environment to generate light and accompanying depth images to train neural networks on generated synthetic data.

UnrealCV-datagen comes with a simple example so that users can quickly integrate UnrealCV into their ML training pipelines.

## Installation
- Install Unreal Engine 4 (Version 4.16 recommended)
- Install UnrealCV
	- See https://github.com/unrealcv/unrealcv
- In the repository directory:
	- pip3 install -r requirements.txt

## Usage
- Use localhost, or the IP of the machine running Unreal Engine 4.
- Import client module to access trainData and testData (numpy arrays)
	

## Example
This example trains a Pytorch single layer neural network to estimate depth from light images.
Input is light images generated in real time in UnrealCV trained against depth images.

- Open the UE4 level packaged with the repository (cubeExample.umap) in a C++ UE4 project
- Play level
- Run
	- python3 net.py -ip 192.168.0.150 -object S1 -plot
	- This example trains a simple single layer network to estimate image depth
	- Will manipulate object 'S1' within UE4 world outliner

## Arguments

### Required:
	-object <object to manipulate>
		This is the name of the object in the  World Outliner in UE4 that will be manipulated and have images generated of

### Others:
	-ip <ip address>
		When not used, value defaults to localmachine. Note that default UnrealCV port of :9000 is always used.
	-plot
		signal to program whether or not to plot training data for debugging

## TODO
Add ability to generate image and depth data and save to disk for later access
render.maxDist is the "rear" of the image that requires tuning for each unique scene. This is not optimal - there is an algorithmic solution. Will add this feature later. It will be based on determining the size of all visible objects, finding their "furthest" orientation pose, and use this distance multiplied by a safety factor to determine render.maxDist.


