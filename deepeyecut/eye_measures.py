# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 18:13:52 2019

@author: rylanl
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

def remove_pupil_outliers (pupils_df,columns, percentile):
    df=pupils_df.copy()
    
    for column in columns:
        
        values = np.asarray(df[column])
        threshold = np.percentile(values[np.isfinite(values)], percentile)
        print('Setting values greater than ' + str(np.percentile(values[np.isfinite(values)], percentile)) + ' to NaN for column '+ column)
        outlier_index = values > threshold
        df[column][outlier_index] = np.nan
        
    return df

def post_process_pupil(pupils_df, columns, percentile,outlier_columns, replace_nan=False, smooth_filter_sigma=0):
    '''Filter pupil parameters and replace outliers with nan.

    :Param pupil_df 
    :Param pupil_percentile: percentile for thresholding outliers. Pupil area values higher than this value are set to NaN.
    :Param replace_nan: Optional, Boolean, whether to replace NaN values (outliers) with the last non-NaN, good value. 
    :Param smooth_filter_sigma: Optional, whether to guassian filter the data and which sigma to use. Recommended is a multiple of the sample rate of the signal (eg 0.05*sample_freq).
                                If zero, the trace is not filtered.
    
    Return: pupil parameters with outliers replaced with nan and optionally guassian filtered
    '''
    #first calculate the pupil area and set values greater than the threshold using NaN 
    
    pupils_df=remove_pupil_outliers(pupils_df,columns=outlier_columns, percentile=percentile)
    
    for column in columns:
        
        #optionally interpolate across NaN values
        if replace_nan:       
            pupils_df[column]=pupils_df[column].interpolate(method='nearest')
    
        #optionally guassian filter
        if smooth_filter_sigma!=0:
            pupils_df[column]=gaussian_filter1d(pupils_df[column].values, int(smooth_filter_sigma))
        
    return pupils_df

def eye_pos_degrees(pupils_df,pupil_cols,cr_cols,eye_radius=1.7,pixel_scale=.001, relative_to_mean=False):
    '''
    Calculates azimuth and altitude positions for the eye in terms of units of degrees.
    
    :param pupil: x and y axis measurements for the centroid of the pupil
    :param cr: x and y axis measurements for the centroid of the corneal reflection
    :param eye_radius: assumed radius of the eye, in mm. For mouse 1.7 based on Remtulla & Hallett, 1985 
    :param pixel_scale: scaling value for number of mm per pixel from the eye monitoring movie in terms of mm per pixel. Requires calibration based on camera magnification, resolution, etc
    :param relative_to_mean: whether to return values that are relative (subtracted from) to the mean pupil/CR for the X and Y axes. If False, returns values not relative to the mean.
    
    :return: azimuth and altiude eye position in measurements of degrees
    
    Based on methods described in Zhuang et al, 2017 and Denman et al, 2017
    
    for azimuth negative should equal nasal (worth checking)
    '''

    #calculate x and y positions as the difference between the pupil and corneal reflection centroids
    #note that the measurments need to have the same units, so multiply by pixel scale to get in terms of mm (same as eye radius)
    print ('using pupil col name ' + pupil_cols[0] + ' for x coordinate calculation')
    x=(pupils_df[pupil_cols[0]]-pupils_df[cr_cols[0]])*pixel_scale
    print ('using pupil col name ' + pupil_cols[1] + ' for y coordinate calculation')
    y=(pupils_df[pupil_cols[1]]-pupils_df[cr_cols[1]])*pixel_scale

    #if relative to the mean, calculate the mean position and subtract measurements from it
    if relative_to_mean==True:
        #np.arctan returns in units of [-pi/2, pi/2], multiply by 180 over pi to get into degrees (alternatively use math.degrees)
        delta_x=np.subtract(x,np.nanmean(x))
        delta_y=np.subtract(y,np.nanmean(y))

    
        azi=np.arctan(np.divide(delta_x,eye_radius))*(180./np.pi)
        alt=np.arctan(np.divide(delta_y,eye_radius))*(180./np.pi)

    else:
        #np.arctan returns in units of [-pi/2, pi/2], multiply by 180 over pi to get to degrees
        azi=np.arctan(np.divide(x,eye_radius))*(180./np.pi)
        alt=np.arctan(np.divide(y,eye_radius))*(180./np.pi)
    
   
    return azi,alt