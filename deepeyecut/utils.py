# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:42:07 2019

@author: rylanl
"""
import os
import numpy as np
import h5py
import pandas as pd
from multiprocessing import Pool
from functools import partial



#utilities


def full_file_path(directory, file_type='.tif',prefix='', exclusion='Null', case_insens=True):
    """Returns a list of files (the full path) in a directory that contain the characters in "prefix"
    with the given extension, while excluding files containing the "exclusion" characters
    
    Args
        -----
        directory (str): path to directory to search in
        file_type (str): extension of file to find ('.tif')
        prefix (str): string that must be in the filename to be found
        exclusion (str): string to used to exclude various files that would normally be found based on being in
                        directory
        case_insens(bool): whether to be concerned with the case of the filename when searching/excluding
                        
        Returns
        _______
        (list) full file paths for all files that are in the directory matchign the criteria specified
    
    """

    file_ext_length=len(file_type)
    for dirpath,_,filenames in os.walk(directory):
      if case_insens==True:
        return [os.path.abspath(os.path.join(dirpath, filename)) for filename in filenames if prefix.lower() in filename.lower() and exclusion.lower() not in filename.lower() and filename[-file_ext_length:] == file_type ]
      
      elif case_insens==False:
        return [os.path.abspath(os.path.join(dirpath, filename)) for filename in filenames if prefix in filename and exclusion not in filename and filename[-file_ext_length:] == file_type ]

def parallelize(data, func, num_of_processes):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)

def parallelize_on_rows(data, func, num_of_processes=16):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)


def pd_df_to_hdf5_dset(df,hdf_dfile,df_columns=[],hdf_names=[],grp='',datatype='f'):
    #note best to pass in hdf_dfile that is opened with mode "a" rather than "w" to avoid replacing data
    if grp and grp not in hdf_dfile:
        main = hdf_dfile.create_group(grp)
    else:
        main=hdf_dfile

    #if columns aren't defined, assume all from the pandas dataframe
    if not df_columns:
        df_columns=list(df.columns)
    #ensure number of hdf names matches number of columns to insert into the hdf5
    if not hdf_names:
        hdf_names=df_columns
    if len(hdf_names)!=len(df_columns):
        raise ('the number of names specified for the HDF5 does not match the number of columns provided')

    for xx,col in enumerate(df_columns):
        #test if the column actually contains a list of multiple values (the centroid values are packaged thsi way)
        if np.shape(df[col][0]):
            xs=[df[col][ll][0] for ll in range(len(df[col]))]
            ys=[df[col][kk][1] for kk in range(len(df[col]))]
            #we split them into individual columns
            dset_x=main.create_dataset(str(hdf_names[xx])+'_x',np.shape(xs))
            dset_x[:]=np.asarray(xs,dtype=np.float32)
            
            dset_y=main.create_dataset(str(hdf_names[xx])+'_y',np.shape(ys))
            dset_y[:]=np.asarray(ys,dtype=np.float32)
            
            #set attribute that describes the number of frames failed to fit for pupil
    
            failed_fit_indexes=df[df.isnull().any(axis=1)][col].index
            dset_x.attrs['failed_fits'] = len(failed_fit_indexes)
            dset_y.attrs['failed_fits'] = len(failed_fit_indexes)
            
        #if not a list of multiple values, make array and insert in HDF5
        else:

            data=np.array(df[col],dtype=np.float32)
            dset=main.create_dataset(str(hdf_names[xx]),np.shape(data))
            dset[:]=data
            
            failed_fit_indexes=df[df.isnull().any(axis=1)][col].index
            dset.attrs['failed_fits'] = len(failed_fit_indexes)
            
            #np.array([np.array(xi) for xi in x])
        

def save_raw_points_h5(df,dfile,pupil_names,cr_names,extra_names,h5_grp="raw_points",close=False):
    idx = pd.IndexSlice
 
    print('saving hdf5 of point fits')
    #---------------------------------------------

    #save raw points
    grp = dfile.create_group(h5_grp)
    
    #pupil 
    dset_p=grp.create_dataset('pupil_x', np.shape(np.array(df[pupil_names].loc[:, idx[:, ['x']]])), dtype='f')
    dset_p[:,:]=np.asarray(df[pupil_names].loc[:, idx[:, ['x']]])
    
    dset_s=grp.create_dataset('pupil_y', np.shape(np.array(df[pupil_names].loc[:, idx[:, ['y']]])), dtype='f')
    dset_s[:,:]=np.asarray(df[pupil_names].loc[:, idx[:, ['y']]])
    
    dset_t=grp.create_dataset('pupil_likelihoods', np.shape(np.array(df[pupil_names].loc[:, idx[:, ['likelihood']]])), dtype='f')
    dset_t[:,:]=np.asarray(df[pupil_names].loc[:, idx[:, ['likelihood']]])
    
    #corneal reflection
    dset_r=grp.create_dataset('cr_x', np.shape(np.array(df[cr_names].loc[:, idx[:, ['x']]])), dtype='f')
    dset_r[:,:]=np.asarray(np.array(df[cr_names].loc[:, idx[:, ['x']]]))   
    
    dset_l=grp.create_dataset('cr_y', np.shape(np.array(df[cr_names].loc[:, idx[:, ['y']]])), dtype='f')
    dset_l[:,:]=np.asarray(np.array(df[cr_names].loc[:, idx[:, ['y']]]))  
    
    dset_m=grp.create_dataset('cr_likelihoods', np.shape(np.array(df[cr_names].loc[:, idx[:, ['likelihood']]])), dtype='f')
    dset_m[:,:]=np.asarray(np.array(df[cr_names].loc[:, idx[:, ['likelihood']]]))  
    
    #extra_names
    
    dset_r=grp.create_dataset('extra_x', np.shape(np.array(df[extra_names].loc[:, idx[:, ['x']]])), dtype='f')
    dset_r[:,:]=np.asarray(np.array(df[extra_names].loc[:, idx[:, ['x']]]))   
    
    dset_l=grp.create_dataset('extra_y', np.shape(np.array(df[extra_names].loc[:, idx[:, ['y']]])), dtype='f')
    dset_l[:,:]=np.asarray(np.array(df[extra_names].loc[:, idx[:, ['y']]]))  
    
    dset_m=grp.create_dataset('extra_likelihoods', np.shape(np.array(df[extra_names].loc[:, idx[:, ['likelihood']]])), dtype='f')
    dset_m[:,:]=np.asarray(np.array(df[extra_names].loc[:, idx[:, ['likelihood']]]))  
    

    if close:
        dfile.close()
        
        
def measure_error_from_annotation (pupils_df,pup_column, annotations_df,annotate_column,error='euclidean'):
    """Given two dataframes, 1 of fit pupil parameters and one of ground-truth annotation values, calculates the error
    between the user-defined columns (X,Y coordinates) using either the euclidean distance between the two or the 
    directed hausdorff distance.
    
    Args
    -----
    pupils_df (pandas df): the fit data to be compared to ground truth
    pup_column (str): the column name of the XY coordinates from the pupils_df
    prefix (str): string that must be in the filename to be found
    exclusion (str): string to used to exclude various files that would normally be found based on being in
                    directory
    case_insens(bool): whether to be concerned with the case of the filename when searching/excluding
                    
    Returns
    _______
    (list) full file paths for all files that are in the directory matchign the criteria specified
    
    """
    failed_fit_indexes=pupils_df[pupils_df.isnull().any(axis=1)][pup_column].index
    
    if error == 'euclidean':
        from scipy.spatial.distance import euclidean
    
        error=[]
        for index, row in pupils_df.iterrows():
            if any(failed_fit_indexes==index):
                error.append(float('NaN'))
            else:
                error.append(euclidean(pupils_df[pup_column].iloc[index],annotations_df[annotate_column].iloc[index]))
            
        return error
    
    elif error=='hausdorff':
        from scipy.spatial.distance import directed_hausdorff as dh
    
        return dh(list(pupils_df[pup_column].values),list(annotations_df[annotate_column].values))