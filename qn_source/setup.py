from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import os, socket, sys
import getpass

# set paths based on hostname
hostname = socket.gethostname()
home = os.path.expanduser('~')
user = getpass.getuser()

apple_cblas = '/Developer/SDKs/MacOSX10.6.sdk/System/Library/Frameworks/vecLib.framework/Versions/A/Headers'
apple_cblas = '/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/'
apple_cblas = '/System/Library/Frameworks/vecLib.framework/Versions/A/Headers/'

if 'darwin' in sys.platform:
    cblas_include = apple_cblas
elif 'ln' in hostname or 'n0' in hostname:
    # neuro cluster
    cblas_include = '/global/software/centos-5.x86_64/modules/mkl/10.0.4.023/include'
elif 'linux' in sys.platform:
    cblas_include = '/usr/include'
    #cblas_include = '/usr/lib/python2.6/dist-packages/scipy/lib/blas/'
else:
    raise NotImplementedError('Not sure where you are building me')


include_dirs = [numpy.get_include(), cblas_include]
library_dirs = []

ext_modules = [
        Extension('quasinewton',
                  sources=['quasinewton.pyx'],
                  depends=['quasinewton.pxd'],
                  include_dirs=[numpy.get_include(), cblas_include],
                  library_dirs=library_dirs,
                  libraries=[]),
        Extension("tokyo",
                  sources=["tokyo.pyx"],
                  depends=['tokyo.pxd'],
                  libraries=['cblas'], # , 'atlas' # tokyo needs atlas. 
                  library_dirs=library_dirs,
                  include_dirs=include_dirs),        
        ]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
    )

