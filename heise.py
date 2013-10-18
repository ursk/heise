""" 
Heisenberg: CC mapping with convolutive FFT 
STA using SVD

"""

import numpy as np
#np.__config__.show() # make sure we are on mkl
import sys
import tables
import matplotlib.pyplot as plt
#plt.interactive(1)
#from ipdb import set_trace as trace
#import cProfile # profile the code with cProfile.run('self.learn(stimexp, spike)')

class heisenberg(object):

	def __init__(self):
		self.im=32
		self.win=8
		self.step=3.
		self.frame=np.int(np.floor((self.im-self.win+1)/self.step))
		self.step = np.int(self.step)
		self.channels = 32 

		self.whiten = True
		self.maxiter = 200 # for lbfgs
		self.rdim = 1024 # dimensionality to retain from 32**2=1024 -- [TODO] Empty dimensions might be what leads to funky HF structure in the filters?

		self.bin = 3. # time bins to reduce sampling rate from 150fps to 50 fps
		self.dt = self.bin * .0066 # time step to get rate in spikes/second
		self.Ttrain = 50000
		self.lam = .01 # l2 regularizer -- .01 seems to work, does not improve things though


	def load_data(self, session='tigerp6', movie='duck8'):
		
		path_to_movies = "./movies.h5" # with data in format /movie [frames * x * y] 3d array 
		path_to_spikes = "./spikes.h5" # with data in format /movie/session [channels * frames] 2d array

		h5 = tables.openFile(path_to_movies, mode = "r")
		stim=h5.getNode('/'+movie)[:]
		h5.close()
		
		self.T=stim.shape[0]
		stim = stim.astype('double')
		stim = stim - stim.mean(0)[np.newaxis, :]
		stim = stim / stim.std(0)[np.newaxis, :]

		# compute 3x downsampled stimulus
		if self.bin > 1:
			Tsub = np.floor(self.T/self.bin)
			stim = stim[0:self.bin*Tsub,:,:].transpose(1,2,0).reshape(self.im,self.im,Tsub, self.bin).sum(3).transpose(2,0,1)/3.
			self.T=Tsub

		if self.whiten:
			cov = np.dot(stim.reshape(self.T,32**2).T, stim.reshape(self.T,32**2))/self.T
			D, E = np.linalg.eigh(cov) # h for symmetric matrices
			E = E[:, np.argsort(D)[::-1]]
			D.sort()
			D=D[::-1]
			self.D=D
			wM = np.dot(E[:,0:self.rdim], np.dot(np.diag((D[0:self.rdim]+.1)**-.5), E[:,0:self.rdim].T))
			stim = np.dot(wM, stim.reshape(self.T,32**2).T).T.reshape(self.T,32,32)
			# adding to the diagonal means the output will not be quite unit variance.
			#stim = stim[:,1:31, 1:31] # crop the center 
			#self.im = 30
		
		
		# compute global normalizers: (not compatible with rDIM)
		fft = np.abs(np.fft.fft2(stim[:,self.win:2*self.win,self.win:2*self.win]))
		self.f_mean = np.fft.fftshift(fft.mean(0))
		self.f_std = np.fft.fftshift((fft-fft.mean(0)).std(0))
		
		h5 = tables.openFile(path_to_spikes, mode = "r")
		spikes=h5.getNode('/'+movie+'/'+session)[:]
		h5.close()
		if self.bin > 1:
			spikes = spikes[:,0:Tsub*self.bin].reshape(32, Tsub, self.bin).sum(2)

		return (stim, spikes) # 




	def sta(self, stimexp, spike, plot=False):
		""" Fourier Expansion STA """
		triggers=np.nonzero(spike)[0]
		canvas4d = stimexp[triggers,:].mean(0).reshape(self.win,self.win,self.frame,self.frame)
		self.siglev = np.sqrt(triggers.shape[0]) # noise level, assuming gaussian


		canblock = canvas4d.reshape(self.win**2, self.frame**2)
		U, s, V = np.linalg.svd(canblock) # U is 64 FFT components, V is 525 locations
		n = 1 # still need 2 components, because the first is the funky whitening artifact.
		canrec = np.dot(U[:,0:n],np.dot(np.diag(s[0:n]),V[0:n,:]))
		canmat = canvas4d.transpose((0,2,1,3)).reshape(self.win*self.frame, self.win*self.frame) # square matrix
		canmatrec = canrec.reshape(self.win, self.win, self.frame, self.frame).transpose((0,2,1,3)).reshape(self.win*self.frame, self.win*self.frame)
		
		# flipped sign seems more natural. 
		self.sta_u = -U[:,0:1]
		self.sta_s = s[0]
		self.sta_v = -V[0:1,:]
		#

		#plt.figure(1), plt.clf()
		#plt.subplot(2,1,1)
		#plt.imshow(canblock, interpolation='nearest')
		#plt.subplot(2,1,2)
		#plt.imshow(canrec, interpolation='nearest')
		#plt.colorbar()
		if plot:
			plt.figure(2), plt.clf()
			ax=plt.subplot(2,2,1)
			plt.imshow(canmat, interpolation='nearest'); plt.title('original'); plt.colorbar()
			plt.xticks(np.arange(0,self.win*self.frame,self.frame)-.5, ''); plt.yticks(np.arange(0,self.win*self.frame,self.frame)-.5, '')
			ax.grid(color='k', linestyle='-', linewidth=.5)
			ax=plt.subplot(2,2,2)
			plt.imshow(canmatrec, interpolation='nearest'); plt.title('SVD reconstruction'); plt.colorbar()
			plt.xticks(np.arange(0,self.win*self.frame,self.frame)-.5, ''); plt.yticks(np.arange(0,self.win*self.frame,self.frame)-.5, '')
			ax.grid(color='k', linestyle='-', linewidth=.5)
			plt.subplot(2,2,3)
			plt.imshow(self.sta_u.reshape(self.win,self.win), interpolation='nearest'); plt.title('f-filter'); plt.colorbar()
			plt.subplot(2,2,4)
			plt.imshow(self.sta_v.reshape(self.frame, self.frame), interpolation='nearest'); plt.title('x-filter'); plt.colorbar()
			print "s is", self.sta_s


		return canvas4d


	def cost_fft(self, stim, spike):
		""" Using online FFT to avoid storing the stimulus represetation"""
		pass

	def expand_stim(self, stim):
		""" precompute all FFTs with stride 3 so it fits in memory """
		

		# build the k_x k_f stimulus representation -- move outside, do once
		T = stim.shape[0]
		X = np.zeros((T, self.win**2, self.frame**2)) # work on 4d canvas as much as we can. 
		t=0
		for x in range(self.frame):
			sys.stdout.write(".")
			for y in range(self.frame):
				patch = np.abs(np.fft.fft2(stim[:, self.step*x:self.step*x+self.win, self.step*y:self.step*y+self.win])) 
				# normalizing here is actually fourier whitening. But it leaves spatial inhomogeneities so we pre-whiten and don't normalize here. 
				npatch = ( np.fft.fftshift(patch, axes=(1,2)) - 1*self.f_mean )  / self.f_std 
				X[:,:,t] = npatch.reshape(T,self.win**2)
				t+=1
		

		return X.reshape(T, self.win**2 * self.frame**2)
		
	def expand_time(self, stim):
		
		# the filters
		f1 = np.array((0, 0, 1)) # 40ms delay, 20ms wide
		f2 = np.array((0, 0, 0, 1, 1, 1))/3. # 60ms delay, 120ms wide
		f3 = np.array((0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1))/6. # 120ms delay, 240ms wide
		# np.convolve does not exist, use a loop
		stimt=np.zeros(stim.shape+(3,))
		stimt[:,:,0] = self.my_convolve(stim, f1)
		stimt[:,:,1] = self.my_convolve(stim, f2)
		stimt[:,:,2] = self.my_convolve(stim, f3)

		return stimt
				
	def my_convolve(self, X, f):
		# filtering along first dimension.
		# same shape, pad end with zeros. 
		Xout = np.zeros(X.shape)
		# accumulate weighted sum of shifts. 
		for i in range(f.shape[0]):
			sys.stdout.write("*")
			Xout[0:self.T-f.shape[0], :] += f[i] * X[i:self.T-f.shape[0]+i,:]

		return Xout

	def cost_store(self, x, args, debug=False):
		""" 
		x = u,v,b
		args = spike, stimexp

		the function we are minimizing can be written as
			f = exp(k'x)
			  = exp(sum_i k_i'x)
			  = exp sum_i u_i v'x_i)
		i.e. the large k is broken up into u and v, and i sums over frequency blocks. 
		Cost function
			C = -( n log(f) - f dt )
		Derivatives 
			d/dv log(f) = sum_i u_i x_i = Xu
			d/du log(f) = v'X

		With regularization (on the elements of u and v!), this becomes
			C = -( n log(f) - f dt ) - u'u - v'v
		"""
		
		# unpack parameters
		stimexp, spike = args

		u = x[0:self.win**2]
		v = x[self.win**2:-1]
		b = x[-1]

		k = np.outer(u,v).flatten() # full filter is 64x81
		
		logf= np.dot(stimexp, k) + b # stimexp is 8*8x9*9, outer.flatten is (64, 81)
		f=np.exp(logf)

		#logfp=
		fp= f[:, np.newaxis]

		# cost: negative log-likelihood.
		c = -(spike * logf - f * self.dt).mean(0) + self.lam*(u**2).sum(0) + self.lam*(v**2).sum(0)# maximize spike*logf overlap, minimize f
		if debug: 
			print "cost", c
		
		# gradients
		T = stimexp.shape[0] # if we work on a subset
		dlogfdu = np.dot(v, stimexp.reshape(T,self.win**2,self.frame**2).transpose((0,2,1))) # k-space
		dlogfdv = np.dot(u, stimexp.reshape(T,self.win**2,self.frame**2)) # x-space, dot does last vs. second-to-last
		dlogfdb = 1
		
		dfdu = fp * dlogfdu # k-space, only true for exp nonlinearity
		dfdv = fp * dlogfdv # x-space
		dfdb = f

		cu = (spike[:, np.newaxis] * dlogfdu).mean(0) - (self.dt * dfdu).mean(0) - self.lam*2*u # minus because it's flipped below.
		cv = (spike[:, np.newaxis] * dlogfdv).mean(0) - (self.dt * dfdv).mean(0) - self.lam*2*v # 
		cb = (spike * dlogfdb).mean(0)                - (self.dt * dfdb).mean(0) 
		
		g = -np.hstack((cu, cv, cb)) # 64+81+1 long

		return c,g

	def learnpixel(self, stim, spike, fourier=False):
		from scipy.optimize import fmin_l_bfgs_b as lbfgs
		
		self.im=24 # reduce the stimulus size a little bit. 
		
				
		
		if fourier:
			fft = np.abs(np.fft.fft2(stim[:,4:28,4:28]))
			f_mean = np.fft.fftshift(fft.mean(0))
			f_std = np.fft.fftshift((fft-fft.mean(0)).std(0))
			stim = (fft-f_mean) / f_std 
			stim = stim.reshape(self.T, self.im**2)[:,0:self.im*(self.im/2+1)] # cut off redundant frequencies
		else:
			stim = stim[:,4:28,4:28].reshape(self.T, self.im**2) # subset and flatten
		
		x0 = 0.001 * np.random.randn(stim.shape[1]+1)
		args = (stim[0:self.Ttrain, :], spike[0:self.Ttrain])
		out = lbfgs(self.cost_pixel, x0, fprime=None, args=[args], iprint=-1, maxiter=self.maxiter, disp=1)

		x = out[0]
		k = x[0:-1]
		b = x[-1]

		prediction = np.exp(np.dot(stim, k) + b)
		pixel_rsq = np.corrcoef(prediction[self.Ttrain:self.T], spike[self.Ttrain:self.T])[0,1]

		return pixel_rsq

	def cost_pixel(self, x, args, debug=False):
		""" 
		x = k,b
		args = spike, stimexp

		"""
		
		# unpack parameters
		stim, spike = args
		k = x[0:-1]
		b = x[-1]
		#trace()
		logf= np.dot(stim, k) + b # 
		f=np.exp(logf)
		fp= f[:, np.newaxis]

		# cost: negative log-likelihood.
		c = -(spike * logf - f * self.dt).mean(0) + self.lam*(k**2).sum(0) # maximize spike*logf overlap, minimize f
		if debug: 
			print "cost", c
		
		# gradients
		dlogfdk = stim # k-space
		dlogfdb = 1
		dfdk = fp * dlogfdk # k-space, only true for exp nonlinearity
		dfdb = f
		ck = (spike[:, np.newaxis] * dlogfdk).mean(0) - (self.dt * dfdk).mean(0) - self.lam*2*k # minus because it's flipped below.
		cb = (spike * dlogfdb).mean(0)                - (self.dt * dfdb).mean(0) 
		
		g = -np.hstack((ck, cb)) # 64+81+1 long

		return c,g

	def learn(self, stimexp, spike):
		""" do the learning """	

		from scipy.optimize import fmin_l_bfgs_b as lbfgs
		# initialize at STA
		if 1:
			u=-self.sta_u      # U is 64 FFT components -- the SVD vectors are unit L2 norm! 
			v=-self.sta_v      # V is 525 locations -- sign flip weirdness in the gradient function?
			b=1*np.ones(1) # bias
			x0=np.vstack((u,v.T,b)).flatten() / np.sqrt(self.sta_s) # package parameters
		else:
			x0 = 0.001 * np.random.randn(self.win**2+self.frame**2+1)

		#stimexp = self.expand_stim(stim) # build fourier representation, 3GB
		args = (stimexp[0:self.Ttrain,:], spike[0:self.Ttrain]) # leave a validation set
		
		# numerical sanity check
		if 1:
			epsi = 1e-6
			eps1  =np.zeros(self.win**2 +self.frame**2 +1); eps1[1]=epsi
			eps100=np.zeros(self.win**2 +self.frame**2 +1); eps100[self.win**2+1]=epsi
			eps145=np.zeros(self.win**2 +self.frame**2 +1); eps145[-1]=epsi

			cost, gradient = self.cost_store(x0, args)
			cost1, gradient = self.cost_store(x0+eps1, args)
			cost100, gradient = self.cost_store(x0+eps100, args)
			cost145, gradient = self.cost_store(x0+eps145, args)
			print "Numerical gradient checks:"
			print gradient[1], (cost1-cost)/epsi   # ok
			print gradient[self.win**2+1], (cost100-cost)/epsi # ok
			print gradient[-1], (cost145-cost)/epsi# ok
			

		out = lbfgs(self.cost_store, x0, fprime=None, args=[args], iprint=-1, maxiter=self.maxiter, disp=1)

		x = out[0]
		glm_u = x[0:self.win**2]
		glm_v = x[self.win**2:-1]
		glm_b = x[-1]

		return glm_u, glm_v, glm_b


	def plotkernels(self, glm_u, glm_v, glm_b, stimexp, spike, plot=False):
		k_f = glm_u.reshape(self.win, self.win)
		k_x = glm_v.reshape(self.frame, self.frame)
		k_matrix = np.outer(k_f, k_x) # outer flattens inputs
		k_4d = k_matrix.reshape(self.win, self.win, self.frame, self.frame)
		k_canvas = k_4d.transpose(0,2,1,3).reshape(self.win*self.frame,self.win*self.frame) # requency outside, location inside
		
		prediction = np.exp(np.dot(stimexp, k_matrix.flatten()) + glm_b)
		rsq = np.corrcoef(prediction[self.Ttrain:self.T], spike[self.Ttrain:self.T])[0,1]
		
		if plot:
			plt.figure(3)
			plt.clf()
			ax=plt.subplot(2,3,1); 
			plt.imshow(k_canvas, interpolation='nearest'); plt.title('GLM kernel'); plt.colorbar()
			plt.xticks(np.arange(0,self.win*self.frame,self.frame)-.5, ''); plt.yticks(np.arange(0,self.win*self.frame,self.frame)-.5, '')
			ax.grid(color='k', linestyle='-', linewidth=.5)

			#plt.subplot(2,2,2); plt.imshow(k_matrix, interpolation='nearest'); plt.colorbar()
			plt.subplot(2,3,2); 
			plt.imshow(k_f, interpolation='nearest'); plt.title('k-f'); plt.colorbar()
			plt.subplot(2,3,3); 
			plt.imshow(k_x, interpolation='nearest'); plt.title('k-x'); plt.colorbar()
			plt.subplot(2,1,2);  
			plt.plot(spike[0:1000]/self.dt); plt.plot(prediction[0:1000]);

		print "---------------------------------"
		print "r-squared on training set = %2.2f" %rsq
		print "---------------------------------"

		return rsq

	def runall():
		"""run all sessions in a loop"""
		for session in ['tigerp6', 'beckp1', 'beckp4', 'orangep4', 'orangep5']:
			pass



