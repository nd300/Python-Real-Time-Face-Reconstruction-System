"""
4D Face Reconstruction System. 
Made as a final year project for the University of Exeter.
This is only to be used for academic purposes.

To start, please run this file with iPython.


Make sure to have a morphable model and a shape predictor in the same directory saved as:
MM -- "all_all_all_norm_v2.mat"
SP -- "shape_predictor_68_face_landmarks.dat"

Please read the README file for more information.
"""

import MMFitting as fit
import ProcessPhotos as PP
import scipy.io as spio
import scipy.spatial as spatial
import numpy as np
import time
import navpy as nv
import cv2

class MainController:
	def __init__(self, shapeChoice):
		#load the morphable model
		self.mat = spio.loadmat('all_all_all_norm_v2.mat', squeeze_me = True)
		self.shapeChoice = shapeChoice
		self.cap = cv2.VideoCapture(0)
		if (self.cap.isOpened()== False): 
			print("Error opening video stream or file")
			
	def main(self, landmarks_first=False):
		#initialize fl and fitting modules
		processPhotos = PP.main(True)
		fitting = fit.MMFitting(self.mat, self.shapeChoice)
		
		#init variables
		shape = []
		curr_landmarks = []
		lastThreeRotationMatrices = []

		#for shape averaging purposes
		startingFrameNum = 1
		midFrameNum = startingFrameNum+25
		
		#for frame tracking purposes
		i = 0

		#main loop
		while(self.cap.isOpened()):
			# capture frame-by-frame
			ret, frame = self.cap.read()
			
			#track frame num
			i+=1
			#check if camera captures
			if (ret == True):
				self.frame = frame
				cv2.imshow('Frame',frame)
				# Press Q on keyboard to  exit
				if cv2.waitKey(25) & 0xFF == ord('q'):
					break
			# Break the loop
			else:
				continue
				
			#make sure to maintain the size of the rotational matrices 
			if(len(lastThreeRotationMatrices)==3):
				lastThreeRotationMatrices.pop(0)
			# extract face landmarks 
			temp_landmarks = processPhotos.processPhotos(self.frame, "Fast")
			#check if face has been found
			if(temp_landmarks != None):
				curr_landmarks = temp_landmarks
			else:
				# invoke the Bulat et al's face-alignment algorithm to try and find a face
				# this works better under heavy occlusion, although it slows the system significantly
				print("Could not find your face. Executing back-up algorithm!")
				temp_landmarks = processPhotos.processPhotos(self.frame, "Slow")
				if(temp_landmarks != []):
					curr_landmarks = temp_landmarks
			#shape training phase
			if (i<midFrameNum):
				#perform fitting
				fitting.secInit(curr_landmarks, i)
				fitting.main()
				currentShape = fitting.identity_shape
				#record shape for averaging purposes
				shape.append (currentShape)
				
				#temporal smoothing	
				#extract current rotational matrix and average it using Euler's triangles
				currentR = fitting.R
				rotAngles = nv.dcm2angle(currentR)
				lastThreeRotationMatrices.append(rotAngles)
				currentR = np.average(np.asarray(lastThreeRotationMatrices).T, axis = 1)
				#output the averaged result back in the fitting module
				fitting.R_Av = currentR
			#average the shapes gathered so far
			elif (i==midFrameNum):
				#perform fitting
				fitting.secInit(curr_landmarks, i)
				fitting.main()
				#record shape for averaging purposes
				currentShape = fitting.identity_shape
				
				shape.append (currentShape)
				
				#start averaging process
				shape = np.asarray(shape).T
				shape=np.average(shape, axis = 2).flatten()
				shape = np.reshape(shape, (len(shape),1))
				#if user has chosen a puppet:
				if (shapeChoice == "2"):
					# subsample vertices for puppet and construct new triangulation list with Delaunay
					shape1 = np.reshape(shape,(len(shape)/3, 3))
					shape1= shape1.T
					puppet_vertices1 = shape1[0][::3]
					puppet_vertices2 = shape1[1][::3]
					puppet_vertices3 = shape1[2][::3]
					puppet_vertices = np.asarray([puppet_vertices1,puppet_vertices2,puppet_vertices3]).T.flatten()
					
					puppet_vertices_forDelaunay = np.asarray([puppet_vertices1,puppet_vertices2])
					puppet_vertices_forDelaunay = puppet_vertices_forDelaunay.T
					from scipy.spatial import Delaunay
					puppet_tri=Delaunay(puppet_vertices_forDelaunay)
					puppet_vertices = np.reshape(puppet_vertices, (len(puppet_vertices),1))	
					
					#output the new shape and triangles list to the fitting and visualisation modules
					fitting.identity_shape = puppet_vertices
					fitting.sampled_triangles = puppet_tri.simplices

				else:				
					fitting.identity_shape = shape
					fitting.sampled_triangles = fitting.faces[0:105600]

						
				#extract current rotational matrix and average it using Euler's triangles
				currentR = fitting.R
				rotAngles = nv.dcm2angle(currentR)
				lastThreeRotationMatrices.append(rotAngles)
				#output the averaged result back in the fitting module
				currentR = np.average(np.asarray(lastThreeRotationMatrices).T, axis = 1)
				fitting.R_Av = currentR
			#continue outputing the averaged shape
			elif(i>midFrameNum):
				fitting.secInit(curr_landmarks, i)
				fitting.main()
					
				#extract current rotational matrix and average it using Euler's triangles
				currentR = fitting.R
				rotAngles = nv.dcm2angle(currentR)
				lastThreeRotationMatrices.append(rotAngles)
				#output the averaged result back in the fitting module
				currentR = np.average(np.asarray(lastThreeRotationMatrices).T, axis = 1)
				fitting.R_Av = currentR
				
consent = raw_input(
"""\n\n\nWelcome to the 4D Face Reconstruction System!	
Built for academic purposes as part of ECM3401 at the University of Exeter.\n
Disclaimer:\n
Please note that your camera will be turned on and you will be recorded 
for the purposes of constructing a face puppet with your image and expressions.
No offense is intended so please have fun with the system!\n
Please confirm that you agree by typing yes: """)
if consent == "yes":
	#ask user whether he wants caricature
	shapeChoice = raw_input(
""" Would you like to output 
[1] Your face shape ?
[2] A caricature face shape ?\n""")
	print "\n\nInitializing the system. Please do not move your head out of the camera frame.\nPress 'q' at any time to exit.\n\n"
	main = MainController(shapeChoice)
	main.main()

