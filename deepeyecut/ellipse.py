# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:29:28 2019

@author: rylanl
"""

#functions for fitting points to an elipse.

import numpy as np



class LSqEllipse:
    #code used under MIT license from Ben Hammel
    #https://github.com/bdhammel/least-squares-ellipse-fitting
    #Least square fitting of an elipse based on 
    # Halir, R., Flusser, J.: 'Numerically Stable Direct Least Squares Fitting of Ellipses'

    def fit(self, data, weights):
        """Lest Squares fitting algorithm 
        Theory taken from (*)
        Solving equation Sa=lCa. with a = |a b c d f g> and a1 = |a b c> 
            a2 = |d f g>
        Args
        ----
        data (list:list:float): list of two lists containing the x and y data of the
            ellipse. of the form [[x1, x2, ..., xi],[y1, y2, ..., yi]]
        Returns
        ------
        coef (list): list of the coefficients describing an ellipse
           [a,b,c,d,f,g] corresponding to ax**2+2bxy+cy**2+2dx+2fy+g
        """
        x, y = np.asarray(data, dtype=float)
        #PL introduced weights!
#         print(np.vstack([x**2, x*y, y**2]))
#         print(np.array(weights))
        
        #Quadratic part of design matrix [eqn. 15] from (*)
        D1 = np.mat(np.vstack([x**2, x*y, y**2])*np.array(weights)[np.newaxis,:]).T
        
        #Linear part of design matrix [eqn. 16] from (*)
        D2 = np.mat(np.vstack([x, y, np.ones(len(x))])*np.array(weights)[np.newaxis,:]).T
        
        #forming scatter matrix [eqn. 17] from (*)
        S1 = D1.T*D1
        S2 = D1.T*D2
        S3 = D2.T*D2  
        
        #Constraint matrix [eqn. 18]
        C1 = np.mat('0. 0. 2.; 0. -1. 0.; 2. 0. 0.')

        #Reduced scatter matrix [eqn. 29]
        M=C1.I*(S1-S2*S3.I*S2.T)

        #M*|a b c >=l|a b c >. Find eigenvalues and eigenvectors from this equation [eqn. 28]
        eval, evec = np.linalg.eig(M) 

        # eigenvector must meet constraint 4ac - b^2 to be valid.
        cond = 4*np.multiply(evec[0, :], evec[2, :]) - np.power(evec[1, :], 2)
        a1 = evec[:, np.nonzero(cond.A > 0)[1]]
        
        #|d f g> = -S3^(-1)*S2^(T)*|a b c> [eqn. 24]
        a2 = -S3.I*S2.T*a1
        
        # eigenvectors |a b c d f g> 
        self.coef = np.vstack([a1, a2])
        self._save_parameters()
            
    def _save_parameters(self):
        """finds the important parameters of the fitted ellipse
        
        Theory taken form http://mathworld.wolfram
        Args
        -----
        coef (list): list of the coefficients describing an ellipse
           [a,b,c,d,f,g] corresponding to ax**2+2bxy+cy**2+2dx+2fy+g
        Returns
        _______
        center (List): of the form [x0, y0]
        width (float): major axis 
        height (float): minor axis
        phi (float): rotation of major axis form the x-axis in radians 
        """

        #eigenvectors are the coefficients of an ellipse in general form
        #a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0 [eqn. 15) from (**) or (***)
        a = self.coef[0,0]
        b = self.coef[1,0]/2.
        c = self.coef[2,0]
        d = self.coef[3,0]/2.
        f = self.coef[4,0]/2.
        g = self.coef[5,0]
        
        #finding center of ellipse [eqn.19 and 20] from (**)
        x0 = (c*d-b*f)/(b**2.-a*c)
        y0 = (a*f-b*d)/(b**2.-a*c)
        
        #Find the semi-axes lengths [eqn. 21 and 22] from (**)
        numerator = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
        denominator1 = (b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        denominator2 = (b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        width = np.sqrt(numerator/denominator1)
        height = np.sqrt(numerator/denominator2)

        # angle of counterclockwise rotation of major-axis of ellipse to x-axis [eqn. 23] from (**)
        # or [eqn. 26] from (***).
        phi = .5*np.arctan((2.*b)/(a-c))

        self._center = [x0, y0]
        self._width = width
        self._height = height
        self._phi = phi

    @property
    def center(self):
        return self._center

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def phi(self):
        """angle of counterclockwise rotation of major-axis of ellipse to x-axis 
        [eqn. 23] from (**)
        """
        return self._phi

    def parameters(self):
        return self.center, self.width, self.height, self.phi

def fit_elipse(row,likelihood_thresh=0.25,threshold_pass_points=5):
    """fits x and y points from deeplabcut to an elipse, with thresholding option to avoid fitting
    points with low likelyhood (Rylan)
        Args
        -----
        row (pandas dataframe row): row from a pandas dataframe containing "x", "y", and "likelihood" for
        each frame
        
        likelihood_thresh (float): likelyhood threshold. Used in calculating whether enough points are available
        for a fit. If at least X (threshold_pass_points) number of points is not greater than this, the fit is 
        not performed on  this frame
        
        Returns
        _______
        dictionary of 
        center (x,y) coordinates
        width of ellipse (width/diameter)
        height
        angle
        area
        
        """
    xs=row.loc[:, 'x']
    ys=row.loc[:, 'y']
    ls=row.loc[:, 'likelihood']
    
    #calculate the number of points that pass the threshold
    points_pass_likelihood=sum(i > likelihood_thresh for i in ls)
    
    #if the number of points passing the likelihood threshold is less than 'threshold_pass_points,
    #set the values equal to NaN (no fit)
    if points_pass_likelihood<threshold_pass_points:
        center, width, height, phi,area=[np.nan,np.nan],np.nan,np.nan,np.nan,np.nan
        
    else:
    
        lsqe = LSqEllipse() 
        lsqe.fit([xs, ys],ls)
        center, width, height, phi = lsqe.parameters()
   
        area = np.pi*width*height
    
    return {'center':center, 'width':width, 'height':height,'angle':phi,'area':area}
