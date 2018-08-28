import scipy.io as spio
import numpy as np
import scipy as sp
from mayavi import mlab
import lsqlin
import time
import navpy as nv

class MMFitting:
	def __init__(self, mat=None, shapeChoice = 1):
		print("Initializing variables...")
		if mat == None:
			mat = spio.loadmat('all_all_all_norm_v2.mat', squeeze_me = True)
		#unpack the morphable model
		morphableModel = mat['fmod_norm']
		self.faces = morphableModel['faces'][()]
		self.mean = morphableModel['mean'][()]
		self.mean = np.reshape(self.mean, (len(self.mean),1))

		self.id_mean = morphableModel['id_mean'][()]
		self.id_basis = morphableModel['id_basis'][()]
		self.exp_basis = morphableModel['exp_basis'][()]
		self.IDlands = morphableModel['IDlands'][()]
		self.faces = np.subtract(self.faces, 1) # to account for MATLAB -> Python compatibility

		self.puppet_vertices =[]
		self.puppet_tri =[]

		self.shapeEV, self.shapePC, self.shapeMU =[], [], []
		self.expressionEV, self.expressionPC = [],[]
		self.ndims = len(self.id_basis[0])
		self.exp_ndims=len(self.exp_basis[0])
		#construct the constrained and orthonormal expression bases
		exp_basis_T=np.transpose(self.exp_basis)
		for i in range(self.exp_ndims):
			self.expressionEV.append(np.linalg.norm(exp_basis_T[i]))
		self.expressionEV = np.asarray(self.expressionEV)
		exp_basis_constrained = np.divide(self.exp_basis, self.expressionEV)
		self.expressionPC=exp_basis_constrained
		
		#construct the constrained and orthonormal shape bases
		id_basis_T = np.transpose(self.id_basis)
		for i in range (self.ndims):
			self.shapeEV.append(np.linalg.norm(id_basis_T[i]))			
		self.shapeEV = np.asarray(self.shapeEV)
		id_basis_constrained = np.divide(self.id_basis, self.shapeEV)
		self.shapePC = id_basis_constrained
		
		self.shapeMU = np.reshape(self.mean, (len(self.mean),1))
		self.model_landmarks = self.IDlands
		self.nverts = int(len(self.shapePC)/3)
		
		self.R, self.t = [], []
		self.s=0
		
		self.R_Av = np.asarray([])
		self.vertices = []
		self.identity_shape=[]
		
		#mayavi cannot handle more than 105600 triangle points
		self.sampled_triangles=self.faces[0:105600]
		
		#for shape averaging purposes
		self.startingFrameNum = 1
		self.midFrameNum = self.startingFrameNum+25
		self.shapeChoice = shapeChoice
		
		#reshape model data for estimation purposes
		self.sortedfps = []
		self.normalisedIDBasisForEstimation = []
		self.orthogonalIDBasisForEstimation = []
		self.meanShapeForEstimation = []
		self.normalisedExpBasisForEstimation=[]
		self.orthogonalExpBasisForEstimation=[]
		self.shapeForExpressionEstimation=[]
		self.meanShapeLandmarkPoints=[]
		self.sortedfps = np.arange(1, self.nverts*3+1).reshape(self.nverts,3)
		
		fpssel=[]
		for i in self.model_landmarks:
			fpssel.append(self.sortedfps[i][0])
			fpssel.append(self.sortedfps[i][1])
			fpssel.append(self.sortedfps[i][2])
		self.sortedfps=fpssel			

		self.normalisedIDBasisForEstimation = self.shapeEV[:self.ndims]
		self.normalisedExpBasisForEstimation = self.expressionEV[:self.exp_ndims]

		for i in range(len(self.sortedfps)):
			self.orthogonalIDBasisForEstimation.append(self.shapePC[self.sortedfps[i]-1])
			self.orthogonalExpBasisForEstimation.append(self.expressionPC[self.sortedfps[i]-1])

		for i in self.sortedfps:
			self.meanShapeForEstimation.append(self.shapeMU[i-1])
		
		self.meanShapeLandmarkPoints = np.reshape(self.meanShapeForEstimation, (len(self.orthogonalIDBasisForEstimation)/3,3))
		#migrate in np array form
		self.normalisedIDBasisForEstimation = np.asarray(self.normalisedIDBasisForEstimation)
		self.orthogonalIDBasisForEstimation = np.asarray(self.orthogonalIDBasisForEstimation)
		self.meanShapeForEstimation = np.asarray(self.meanShapeForEstimation)
		self.normalisedExpBasisForEstimation=np.asarray(self.normalisedExpBasisForEstimation)
		self.orthogonalExpBasisForEstimation=np.asarray(self.orthogonalExpBasisForEstimation)
	
	def secInit(self, landmarks=None, curr_frame_num=None):
		''' this is needed to feed in the landmarks at every frame in an elegant way'''
		self.x_landmarks = landmarks
		self.curr_frame_num=curr_frame_num

	def main(self, filename = None):
		"""
		main controller of this file
		"""
		if (filename == None):
			filename='Test Model'
		#extract the vertices from the fitting modules
		vertices1, vertices2, vertices3 = self.fitting()
		#visualise puppet
		self.plot_mesh(vertices1, vertices2,vertices3, self.sampled_triangles,filename, rep="surface")

		
	def fitting(self):
		""" 
		Pose, Shape and Expression estimation module
		"""
		
		if(self.curr_frame_num == self.startingFrameNum):
			# calculate  R, t, s
			# execute this only on first frame
			# t, s are saved here purely for visualisation purposes
			self.R, self.t, self.s = self.POS(self.x_landmarks, self.meanShapeLandmarkPoints)
			tFitting = self.t
			sFitting = self.s			
		else:
			# keep reestimating pose, but only keep R for visualisation purposes
			# tFitting and sFitting are only used to estimate shape and expression
			self.R, tFitting, sFitting = self.POS(self.x_landmarks, self.meanShapeLandmarkPoints)
				
		R=self.R
		t=self.t
		s=self.s
		numsd=3
		
		if (self.curr_frame_num<=self.midFrameNum):
			# only estimate shape for the first 25 frames
			# estimate shape coefficients
			b = self.estimateShape(self.orthogonalIDBasisForEstimation, self.meanShapeForEstimation, self.normalisedIDBasisForEstimation, R, tFitting, sFitting, numsd, self.ndims, self.x_landmarks)
			# for carricature puppet: modify pne shape coefficient directly as that affects the principal components of shape vector as well
			if(self.shapeChoice == "2"):
				b[11] = (b[11]+2)*10 #11*10 6*10 1*20
			#identity_basis*coefficients
			identity = np.dot(self.shapePC, b)
			#add to the mean shape
			self.identity_shape = np.add( identity, self.shapeMU)
			# subselect vertices of the current identity shape for expression estimation
			self.shapeForExpressionEstimation = []
			for i in self.sortedfps:
				self.shapeForExpressionEstimation.append(self.identity_shape[i-1])
			self.shapeForExpressionEstimation = np.asarray(self.shapeForExpressionEstimation)

		#estimate shape coefficients
		e = self.estimateExpression(self.orthogonalExpBasisForEstimation, self.shapeForExpressionEstimation, self.normalisedExpBasisForEstimation, R, tFitting, sFitting, numsd, self.exp_ndims, self.x_landmarks)
		if(self.shapeChoice == "2"):
			#expression_basis*coefficients
			expression_mean = np.dot(self.expressionPC, 1.5*e) # more pronounced expressions if caricature is chosen
		else:
			#expression_basis*coefficients
			expression_mean = np.dot(self.expressionPC, e)


		if(self.curr_frame_num>self.midFrameNum and self.shapeChoice == "2"):
			#subseleact expression to combine with subselected shape
			expression_mean=np.reshape(expression_mean,(len(expression_mean)/3, 3)).T
			expression_mean1 = expression_mean[0][::3]
			expression_mean2 = expression_mean[1][::3]
			expression_mean3 = expression_mean[2][::3]
			expression_mean = np.asarray([expression_mean1, expression_mean2, expression_mean3]).T.flatten()
			expression_mean=np.reshape(expression_mean,(len(expression_mean), 1))
			

		vertices = []
		vertices1, vertices2, vertices3 = [], [], []
		#construct the puppet
		vertices = np.add( self.identity_shape, expression_mean)
		#reshape the vertices matrix to add pose
		vertices = np.reshape(vertices,(len(vertices)/3, 3))
		vertices = vertices.T
		vertices = vertices.tolist()
		vertices.append([1 for x in range (len(vertices[0]))])
		vertices = np.asarray(vertices)
		vertices = vertices.T
		
		#calculate current rotation by averaging it with last 3 known rotations - Temporal smoothing
		if (self.R_Av.size) > 0:
			currRinEuler = nv.dcm2angle(R)
			RinEuler = (currRinEuler + 3*(self.R_Av))/4
			R = nv.angle2dcm(RinEuler[0],RinEuler[1],RinEuler[2])

		#Catching and amending inconsistencies because rotational matrix is distorted by the averaging algorithm.
		#the problem seems patched up for now
		#Note: Phase Unwrapping does not fix this!
		if((R[1][1]<0.85 and R[1][1]>-0.85)):
			Rr=np.negative(self.R)
			Rr[0]=np.negative(Rr[0])
			
		elif(R[1][1]<0):
			Rr=np.negative(R)
			Rr[0]=np.negative(Rr[0])
			#print("Two!")
			
		else:
			Rr=np.negative(self.R)
			Rr[0]=np.negative(Rr[0])
			
		# reshape R, t and s in order to combine with vertices matrix
		Rr = Rr.tolist()
		Rr.append([0,0,0,1])
		Rr[0].append(0)
		Rr[1].append(0)
		Rr[2].append(0)
		Sr = [[s,0,0,0],[0,s,0,0],[0,0,s,0],[0,0,0,s]]
		Tr=[[1,0,0,t[0]],[0,1,0,t[1]],[0,0,1,0],[0,0,0,1]]
		T = np.dot(Sr,Tr)
		T = np.dot(T,Rr)
		M=T[0:3]
		M = np.transpose(M)
		
		#add pose to the current shape
		self.vertices = np.transpose(np.dot(vertices, M))
		

		vertices1=self.vertices[0] #x
		vertices2=self.vertices[1] #y
		vertices3=self.vertices[2] #z
		
		return vertices1, vertices2, vertices3
		    
	def plot_mesh(self, vertices1, vertices2, vertices3, faces, filename ,rep="surface"):
		"""
		plots the mesh defined by the vertices and faces
		"""
		if(self.curr_frame_num==self.startingFrameNum):
			# if this is the first frame, initialize the scene and the figure
			self.fig = mlab.figure()
			mlab.view(0,180, figure = self.fig)

			self.tmesh = mlab.triangular_mesh(vertices1, vertices2, vertices3, faces, representation = rep, figure = self.fig, color = (.7,.7,.7))
			self.tmesh.scene.anti_aliasing_frames = 0
			print("\nShape Trainining in progress!\nTo exit press 'q'")

		elif(self.curr_frame_num <=self.midFrameNum):
			# only update the scene for all frames before shape averaging frame
			self.tmesh.scene.disable_render = True
			self.tmesh.mlab_source.x, self.tmesh.mlab_source.y, self.tmesh.mlab_source.z = vertices1, vertices2, vertices3
			self.tmesh.scene.disable_render = False
		elif(self.curr_frame_num == self.midFrameNum+1):
			# if the shape has been recalculated -> reinitialize the scene and figure
			mlab.close(self.fig)	
			self.fig = mlab.figure()
			mlab.view(0,180, figure = self.fig)

			self.tmesh = mlab.triangular_mesh(vertices1, vertices2, vertices3, faces, representation = rep, figure = self.fig, color = (.7,.7,.7))
			self.tmesh.scene.anti_aliasing_frames = 0
			print("\nShape training Done! Feel free to move around and test the puppet!\nTo exit press 'q'")

		elif(self.curr_frame_num > self.midFrameNum+1):
			#keep updating the scene for all frames 
			self.tmesh.scene.disable_render = True
			self.tmesh.mlab_source.x, self.tmesh.mlab_source.y, self.tmesh.mlab_source.z = vertices1, vertices2, vertices3
			self.tmesh.scene.disable_render = False

		
		
	def POS(self, landmarks, landmarks_model):
		"""
		estimate pose from landmarks
		"""
		npts = len(landmarks)
		b=[]
		#build the matrix of linear equations
		A = [[0 for i in range (8)] for j in range (npts*2)]
		for i in range(len(A)):
			if(i%2==0):
				for j in range(len(A[i])):
					if (j<3):
						A[i][j]=landmarks_model[i/2][j]
					elif(j==3):
						A[i][j]=1
			else:
				for k in range(len(A[i])):
					if (k>=4 and k<=6):
						A[i][k]=landmarks_model[i/2][k-4]
					elif (k==7):
						A[i][k]=1
		#build vector
		for i in range (len(landmarks)):
			b.append(landmarks[i][0])
			b.append(landmarks[i][1])
		
		#solve equations 
		k = np.linalg.lstsq(A, b)[0]
		
		#extract results
		R1 = k[0:3]
		R2 = k[4:7]
		R1Norm = np.linalg.norm(R1,2)
		R2Norm = np.linalg.norm(R2,2)
		sTx = k[3]
		sTy = k[7]
		s = (R1Norm+R2Norm)/2
		
		#construct rotational matrix
		r1=[]
		r2=[]
		for i in R1:
			r1.append(i/R1Norm)
		for i in R2:
			r2.append(i/R2Norm)
		r3 = np.cross(r1,r2)
		r3Temp=[]
		for i in range(len(r3)):
			r3Temp.append(r3[i])
		r3=r3Temp
		
		R=[r1, r2, r3]
		#enforce a valid rotational matrix!
		U, S, V = np.linalg.svd(R)
		R = np.dot(U,V)
		
		#construct translation vector
		t=[sTx/s, sTy/s]
		
		return R,t,s


	def estimateShape(self, shapePC, shapeMU, shapeEV, R, t, s, numsd, ndims, landmarks):
		"""
		estimate shape from identity bases, mean shape and pose params
		"""
		shapePC=shapePC.T
		P=[]
		#reshape bases to suit calculations
		for i in range(len(shapePC)):
			P.append(np.reshape(shapePC[i],(len(shapePC[i])/3,3)))
		mu = np.reshape(shapeMU, (len(shapeMU)/3,3))
		A = np.zeros((2*(len(shapePC[0])/3),ndims))
		h=[]
		P=np.asarray(P)
		#construct system of linear equations
		for i in range (0,(len(shapePC[0])/3)*2-1,2):
				A[i]=np.multiply(s*R[0][0],np.squeeze(P[:,i/2, 0]))
				A[i]+=np.multiply(s*R[0][1],np.squeeze(P[:,i/2, 1]))
				A[i]+=np.multiply(s*R[0][2],np.squeeze(P[:,i/2, 2]))

				A[i+1]=np.multiply(s*R[1][0],np.squeeze(P[:,i/2, 0]))
				A[i+1]+=np.multiply(s*R[1][1],np.squeeze(P[:,i/2, 1]))
				A[i+1]+=np.multiply(s*R[1][2],np.squeeze(P[:,i/2, 2]))
				
			
				h.append(landmarks[i/2][0]-np.dot(s,(np.multiply(R[0][0],mu[i/2][0])+np.multiply(R[0][1],mu[i/2][1])+np.multiply(R[0][2],mu[i/2][2])+t[0])))
				h.append(landmarks[i/2][1]-np.dot(s,(np.multiply(R[1][0],mu[i/2][0])+np.multiply(R[1][1],mu[i/2][1])+np.multiply(R[1][2],mu[i/2][2])+t[1])))
		A=np.asarray(A)
		h=np.asarray(h)
		
		#initialize constraints
		C=np.eye(ndims)
		C=np.append(C,-np.eye(ndims),axis=0)
		d=np.append(np.multiply(numsd, shapeEV), np.multiply(numsd, shapeEV), axis=0)
		C=np.asarray(C)
		d=np.asarray(d)

		#solve constrained system
		b= lsqlin.lsqlin(A, h, 0, C, d, None, None, None, None, None, {'show_progress': False})
		return(np.asarray(b['x']))
	
	def estimateExpression(self, expressionPC, shapeMU, expressionEV, R, t, s, numsd, ndims, landmarks):
		"""
		estimate expression from expression bases, mean shape and pose params
		"""
		expressionPC=expressionPC.T
		P=[]
		#reshape bases to suit calculations

		for i in range(len(expressionPC)):
			P.append(np.reshape(expressionPC[i],(len(expressionPC[i])/3,3)))
		mu = np.reshape(shapeMU, (len(shapeMU)/3,3))
		A = np.zeros((2*(len(expressionPC[0])/3),ndims))
		h=[]
		P=np.asarray(P)

		for i in range (0,(len(expressionPC[0])/3)*2-1,2):
				A[i]=np.multiply(s*R[0][0],np.squeeze(P[:,i/2, 0]))
				A[i]+=np.multiply(s*R[0][1],np.squeeze(P[:,i/2, 1]))
				A[i]+=np.multiply(s*R[0][2],np.squeeze(P[:,i/2, 2]))

				A[i+1]=np.multiply(s*R[1][0],np.squeeze(P[:,i/2, 0]))
				A[i+1]+=np.multiply(s*R[1][1],np.squeeze(P[:,i/2, 1]))
				A[i+1]+=np.multiply(s*R[1][2],np.squeeze(P[:,i/2, 2]))
				
			
				h.append(landmarks[i/2][0]-np.dot(s,(np.multiply(R[0][0],mu[i/2][0])+np.multiply(R[0][1],mu[i/2][1])+np.multiply(R[0][2],mu[i/2][2])+t[0])))
				h.append(landmarks[i/2][1]-np.dot(s,(np.multiply(R[1][0],mu[i/2][0])+np.multiply(R[1][1],mu[i/2][1])+np.multiply(R[1][2],mu[i/2][2])+t[1])))
		A=np.asarray(A)
		h=np.asarray(h)

		#initialize constraints
		C=np.eye(ndims)
		C=np.append(C,-np.eye(ndims),axis=0)
		d=np.append(np.multiply(numsd, expressionEV), np.multiply(numsd, expressionEV), axis=0)
		C=np.asarray(C)
		d=np.asarray(d)

		#solve constrained system
		e= lsqlin.lsqlin(A, h, 0, C, d, None, None, None, None, None, {'show_progress': False})
		return(np.asarray(e['x']))

