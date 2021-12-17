##################### module containing information about scene, posing, and camera
# modules
import render, plot, data


class pose():
	''' creates pose class object '''
	def __init__(self, position, rotation, name = '/camera/0/'):
		self.position = position
		self.rotation = rotation

		# parse string from unrealcv into list and convert to floats
		self.position = self.position.split(' ')
		for j, i in enumerate(self.position):
			self.position[j] = float(i)

		self.rotation = self.rotation.split(' ')
		for j, i in enumerate(self.rotation):
			self.rotation[j] = float(i)

		# set x y and z for camera position
		self.x = self.position[0]
		self.y = self.position[1]
		self.z = self.position[2]

		# set pitch yaw and roll for camera rotation
		self.pitch = self.rotation[0]
		self.yaw = self.rotation[1]
		self.roll = self.rotation[2]
		
		self.obj = name


def resetPose(self, client):

	# Reset object
	self.x = 0
	self.y = 0
	self.z = 0
	self.pitch = 0
	self.yaw = 0
	self.roll = 0
	
	# Reset camera
	client.request('vset /camera/0/location 200 0 50')
	client.request('vset /camera/0/rotation 0 -180 0')
	
def setPose(client, p):
	''' take client connection, and pose containing camera
		information (p) and set new camera pose '''
	
	client.request(f'vset {p.obj}/location {p.x} {p.y} {p.z}')
	client.request(f'vset {p.obj}/rotation {p.pitch} {p.yaw} {p.roll}')

def anim(client, p, increment, build, w, h):
	''' Animate model and return training data for testing '''

	p.pitch += increment
	p.yaw += increment
	p.roll += increment

	setPose(client, p)

	resImage, resDepth = render.get(client, build, w, h)

	return(resImage, resDepth)



