import face_alignment
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2


class main:
	def __init__(self, save=False, shapePredictorDir = "shape_predictor_68_face_landmarks.dat"):
		#initialize 
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(shapePredictorDir)
		self.save = save
		self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=False, flip_input=False)

	def processPhotos(self, frame, faceLandmarkingAlg = "Fast"):
		'''
		this method outputs the facial landmarks of the input frame
		'''
		try:
			if(faceLandmarkingAlg == "Fast"):
				
				# load the input image, resize it, and convert it to grayscale
				# this is done as it's claimed that accuracy is greater
				image = frame
				image = imutils.resize(image, width=500)
				gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

				# detect faces in the grayscale image
				rects = self.detector(gray, 1)

				# loop over the face detections
				for (i, rect) in enumerate(rects):
					# determine the facial landmarks for the face region, then
					# convert the facial landmark (x, y)-coordinates to a NumPy
					# array
					shape = self.predictor(gray, rect)
					preds_list = face_utils.shape_to_np(shape)
					preds = preds_list
					return preds_list.tolist()
			else:
				#run slower fl algorithm
				input = frame
				preds =self.fa.get_landmarks(input)[-1]
				preds_list= preds.tolist()

				return preds_list

		except TypeError:
			print "There has been an error with the current frame"
			return []