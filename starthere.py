""" 
Wrapper file for heise.py

Dependencies: pytables for reading hdf5 data files



"""

import numpy as np
#np.__config__.show() # make sure mkl libraries are used
import sys
import tables
import matplotlib.pyplot as plt
plt.interactive(1)
#from ipdb import set_trace as trace
#import cProfile # profile the code with cProfile.run('self.learn(stimexp, spike)')



if __name__ == '__main__':
	
	# set up the model and get data:
	# data is expected in two files:
	# movies.h5 with data in format /movie [frames * x * y] 3d array 
	# spikes.h5 with data in format /movie/session [channels * frames] 2d array
	
	# edit heise.ph to change parameters in __init__ and give it the data file names in load_data

	# initialize: grab the data
	self = heise.heisenberg()
	session='orangep5'  # 'tigerp6'
	movie='duck30'
	(stim, spikes) = self.load_data(session=session, movie=movie) # 
	stimexp = self.expand_stim(stim)



	# model and visualization 
	rsq=np.zeros(self.channels)
	pixel_rsq=np.zeros(self.channels)
	fft_rsq=np.zeros(self.channels)
	glm_u=np.zeros((self.channels,self.win**2))
	glm_v=np.zeros((self.channels,self.frame**2))
	glm_b=np.zeros(self.channels)
	sta_u=np.zeros((self.channels,self.win**2))
	sta_v=np.zeros((self.channels,self.frame**2))
	for i in range(0,self.channels): # (10,self.channels,7):
		print "estimating heise-GLM for cell", i
		spike = np.double(spikes[i-1,:]>0) # 
		
		# estimate STA
		canvas4d = self.sta(stimexp, spike) # STA sets self.sta_u ...
		fsign = 2*(self.sta_u.mean()>0)-1 # sign to flip
		sta_u[i,:] = fsign*self.sta_u.flatten()
		sta_v[i,:] = fsign*self.sta_v.flatten()
		
		# estimate GLM
		glm_u[i,:], glm_v[i,:], glm_b[i] = self.learn(stimexp, spike)
		fsign = 2*(glm_u[i,:].mean()>0)-1 # sign to flip
		glm_u[i,:] = fsign * glm_u[i,:]
		glm_v[i,:] = fsign * glm_v[i,:]
		rsq[i]=self.plotkernels(glm_u[i,:], glm_v[i,:], glm_b[i], stimexp, spike)

		# estimate pixel GLM to compare performance
	for i in range(0,self.channels):
		print "estimating pixel-GLM for cell", i
		spike = np.double(spikes[i-1,:]>0) # 
		#pixel_rsq[i] = self.learnpixel(stim, spike)
		fft_rsq[i] = self.learnpixel(stim, spike, fourier=True )

	print "explained variances r=", rsq[0:self.channels]**.5
	plt.figure(17); plt.clf() 
	#plt.plot(rsq); #plt.bar(range(0,self.channels), rsq)
	#plt.plot(pixel_rsq);
	plt.bar(range(0,self.channels), rsq)
	plt.bar(range(0,self.channels), pixel_rsq, color='g')
	plt.bar(range(0,self.channels), fft_rsq, color='r')
	plt.xlim((0, self.channels)); plt.ylim((-.05, .25))
	plt.legend({'heise', 'pixel'})
	plt.savefig('rsq_bar_'+session+'_'+movie+'.pdf')

	# save data to disk
	np.savez('outfile_'+session+'_'+movie+'step'+str(self.step), glm_u, glm_v, sta_u, sta_v, rsq, pixel_rsq)
	
	# plot the kernels
	plt.figure(5); plt.clf(); plt.jet()
	for i in range(0,self.channels): # (10,self.channels,7):
		sys.stdout.write(str(i)+' '); sys.stdout.flush()
		plt.axes([0.1, 0.05+i/34., .1, 1/37.]); plt.xticks(()); plt.yticks(()); vm = np.abs(glm_u[i,:]).max()
		plt.imshow( glm_u[i,:].reshape(self.win, self.win), interpolation='nearest', vmin=-vm, vmax=vm )
		plt.ylabel(str(i+1), {'rotation' : 'horizontal'})
		plt.axes([0.2, 0.05+i/34., .1, 1/37.]); plt.xticks(()); plt.yticks(()); vm = np.abs(glm_v[i,:]).max()
		plt.imshow( glm_v[i,:].reshape(self.frame, self.frame), interpolation='nearest', vmin=-vm, vmax=vm )
		plt.axes([0.3, 0.05+i/34., .1, 1/37.]); plt.xticks(()); plt.yticks(()); vm = np.abs(sta_u[i,:]).max()
		plt.imshow( sta_u[i,:].reshape(self.win, self.win), interpolation='nearest', vmin=-0, vmax=vm )
		plt.axes([0.4, 0.05+i/34., .1, 1/37.]); plt.xticks(()); plt.yticks(()); vm = np.abs(sta_v[i,:]).max()
		plt.imshow( sta_v[i,:].reshape(self.frame, self.frame), interpolation='nearest', vmin=-vm, vmax=vm )
	# save image to disk. 
	plt.savefig('glm_'+session+'_'+movie+'.pdf')
	



	# individual example plots:
	i=24 # sample cell
	spike = np.double(spikes[i-1,:]>0) 
	rsq[i]=self.plotkernels(glm_u[i,:], glm_v[i,:], glm_b[i], stimexp, spike, plot=True)
	plt.savefig('glm_example_'+session+'_'+movie+'.pdf')
	canvas4d = self.sta(stimexp, spike, plot=True)
	plt.savefig('sta_example_'+session+'_'+movie+'.pdf')
	
	# run Pixel STA and STC for comparison plots:
	plt.clf()
	for i in range(0,self.channels):
		sys.stdout.write(str(i)+' '); sys.stdout.flush()
		spike = np.double(spikes[i-1,:]>0)
		triggers = np.nonzero(spike)[0]
		n = triggers.shape[0]
		frames = stim[triggers,:,:]
		sta = frames.mean(0)
		cov = np.cov(frames.reshape(n, self.im**2).T)
		D, E = np.linalg.eigh(cov)

		plt.axes([0.05, 0.05+i/34., .103, 1/37.]); plt.xticks(()); plt.yticks(()) #  [left, bottom, width, height] 
		plt.imshow(sta*np.sqrt(n), interpolation='nearest', vmin=-7, vmax=7) # scale to 5 std on new noise
		plt.ylabel(str(i+1), {'rotation' : 'horizontal'})
		for k in range(1,6+1):
			plt.axes([0.05+k/11., 0.05+i/34., .1, 1/37.]); plt.xticks(()); plt.yticks(())
			plt.imshow( E[:,-k].reshape(self.im, self.im) )
	# save image to disk. 
	plt.savefig('sta_stc_'+session+'_'+movie+'.pdf')



	# load file and reuse:
	stuff = np.load('outfile_'+session+'_'+movie+'.npz')
	glm_u = stuff.items()[1][1] # 64 freq
	glm_v = stuff.items()[0][1]
	sta_u = stuff.items()[2][1] # 64 freq
	sta_v = stuff.items()[3][1]
	rsq   = stuff.items()[4][1]
	pixel_rsq=stuff.items()[5][1]


