#! /usr/bin/env python

"""
Low-rank plus sparse trajectory decomposition for direct exoplanet imaging
     
"""

__author__ = 'Simon Vary, Hazan Daglayan'


import numpy as np

from pynopt.nopt.constraints import FixedRank, PositiveSparsity
from pynopt.nopt.problems import LinearProblemSum
from pynopt.nopt.solvers import NAHT

from pynopt.nopt.transforms import EntryWise, CompositeTransform
from pynopt.nopt.transforms.identity import Identity


import vip_hci as vip
import pylops
import scipy



class Trajectorlet(pylops.LinearOperator):
    def __init__(self, shape, kernel, angles, dtype=None, *args, **kwargs):
        '''
            Params:
                - shape (the dimensions of the data cube)
                - kernel (the psf kernel for convolution)
                - angles (list of rotated angles)
        '''
        self._shape_cube = shape
        self.shape = (np.prod(self._shape_cube), np.prod(self._shape_cube[-2:]))
        self.dtype = dtype
        super().__init__(dtype, self.shape, *args, **kwargs)
        self._kernel = kernel
        self._angles = angles
        self._conv2d = pylops.signalprocessing.Convolve2D(
            N = np.prod(self._shape_cube[-2:]),
            dims=self._shape_cube[-2:],
            h=self._kernel,
            offset = (self._kernel.shape[0]//2+1, self._kernel.shape[1]//2+1)
        )

    def _matvec(self, x): # Forward operation \Psi(P)
        y = self._conv2d.matvec(x)
        cube_out = np.tile(y, (self._shape_cube[0], 1))
        cube_der = vip.preproc.cube_derotate(cube_out.reshape(self._shape_cube),
                                             -self._angles, imlib='skimage')
        return cube_der.flatten()

    def _rmatvec(self, y): # Adjoint operation \Psi^T(y)
        der_cube = vip.preproc.cube_derotate(y.reshape(self._shape_cube), 
                                             self._angles, imlib='skimage')
        x = self._conv2d.rmatvec(der_cube.sum(axis=0).flatten())
        return x






def create_annular_regions(cube_shape, r_in, r_out, r_by, center):
    '''
        Params:
            - cube_shape (shape of the data cube)
            - r_in (inner radius)
            - r_out (outer radius)
            - r_by (annular size)
            - center (center of the frame)
    
    '''
    cx, cy = center
    x = np.arange(0, cube_shape[2])
    y = np.arange(0, cube_shape[1])
    annular_regions_frame = []
    annular_regions_cube = []
    for r in range(r_in, r_out, r_by):
        mask_out = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 >= r**2
        mask_in = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < (r+r_by)**2
        annular_regions_frame.append((mask_in*mask_out).flatten())
        annular_regions_cube.append(np.tile(mask_in*mask_out,(cube_shape[0],1,1)).flatten())
    return annular_regions_frame, annular_regions_cube

def exoplanet_lrpt_annular(cube, angle_list,  fwhm, inner_rad=0, outer_rad=10, 
                           asize=10, r=20, s=10, MAX_ITER=30, psfn = None, 
                           normalize = True, lsqr_init=False, verbosity=2, maxtime=120):
    '''
        Params:
            - cube (data cube)
            - angle_list (list of rotated angles)
            - fwhm (FWHM of teleschope)
            - inner_rad (inner radius of the annulus)
            - outer_rad (outer radius of the annulus)
            - asize (annular size)
            - r (rank)
            - s (sparsity)
            - MAX_ITER (maximum iteration number for NAHT)
            - psfn (normalized PSF)
            - normalize (boolean for normalization)
            - lsqr_init (boolean for initialization with lsqr)
            - verbosity (0 : No output,
                         1 : Only the final result of each minimization problem,
                         2 : Results on each step of minimization problem)
    '''

    cy, cx = vip.var.frame_center(cube)
    
    
    if normalize:
        stdev_frame = cube.std(axis=0)
        cube_in = cube / stdev_frame
    else:
        cube_in = cube
    
    # Divide frame and cube to annuli
    annular_regions_frame, annular_regions_cube = create_annular_regions(cube_in.shape, 
                                                    inner_rad, outer_rad, asize, 
                                                    (cx, cy))
    all_pixels = sum(annular_regions_cube).reshape(cube_in.shape)
    kernel = psfn / np.linalg.norm(psfn)

    
    cube_planet_ = np.zeros_like(cube_in)
    cube_background = np.zeros_like(cube_planet_)
    x_frame = np.zeros(np.prod(cube_in[0].shape))
    opt_log = []
    
    
    # Apply algorithm for each frame
    for i in range(len(annular_regions_cube)):
        # Indices 
        outp_ind = np.where(annular_regions_cube[i]==1)[0]
        outp_ind_frame = np.where(annular_regions_frame[i]==1)[0]
        
        t, m, n = cube_in.shape
        
        
        P_omega = EntryWise(outp_ind, m*n*t) # Decide omega - the indices in the annulus -  
        b = P_omega.matvec(cube_in.flatten()) # Cube values in the indices
        
        # Constraints
        HTr = FixedRank(r, (t,int(len(b)/t)), randomized=False)
        HTs = PositiveSparsity(s)
        constraints = (HTr, HTs)
        trajectorlet = Trajectorlet(cube_in.shape, kernel/t**(1/2), angle_list)
        comp_tra = CompositeTransform((trajectorlet,P_omega)) 
        As = (Identity(), comp_tra) 
        
        # Define the problem
        problem = LinearProblemSum(As, b, constraints)
        solver = NAHT(logverbosity = 2, maxiter = MAX_ITER, 
                      verbosity = verbosity, maxtime=maxtime, minreldecrease = 1-1e6)

        # Initialization with LSQR
        if lsqr_init:
            x0 = [None, None]
            _, x0[0] = constraints[0].project(b)
            w1 = As[1]._As[1].rmatvec(x0[0] - b)
            w2 = scipy.sparse.linalg.lsqr(As[1]._As[0], w1, show=False, iter_lim=50)[0]
            _,x0[1] = constraints[1].project(w2)
        else:
            x0 = None

        # Solve the problem
        x, opt_log_ann = solver.solve(problem, x0=x0) 
        
        # Assign solution to the frames and cubes
        x_frame[outp_ind_frame] = x[1][outp_ind_frame]
        ann_3D = annular_regions_cube[i].reshape(cube_in.shape)
        cube_background[ann_3D] = x[0]
        
        # Return to nonnormalized values 
        if normalize:
            cube_background[ann_3D] = (cube_background*stdev_frame)[ann_3D]
        
        opt_log.append(opt_log_ann)
        

    # Convolution
    y = trajectorlet._matvec(x_frame)
    cube_planet_ = y.reshape(cube_in.shape)
    cube_planet = vip.preproc.cube_derotate(cube_planet_, angle_list, imlib='skimage')
    

    return (cube_planet[0], cube_background, all_pixels, opt_log)
    
