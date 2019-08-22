# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:48:45 2019

@author: rylanl
"""
from deeplabcut import analyze_videos
import pandas as pd
import os
import h5py


from deepeyecut.ellipse import fit_elipse
from deepeyecut.utils import full_file_path,pd_df_to_hdf5_dset,save_raw_points_h5, parallelize_on_rows
from deepeyecut.video_annotation import annotate_movie

from functools import partial


fit_extra_to_elipse=False

likelihood_pup_thresh=0.9
likelihood_oth_thresh=0.8


#define the columns that correspond to pupil, corneal reflection (CR), and any extra points (for example, eyelid)
#to determine this open the config.yaml file dfined in the path_config_file below

pupil_cols=(6,10) #for peterL's model (1,8)
cr_cols=(0,1) #for peterL's model (0,1)
extra_cols=(10,18)  #for peterL's model (8,16)

#path to deeplab cut configuration to use that includes trained network weights
path_config_file = r"D:\Analysis\deep_eye_human\Human_Eye_track-Rylan_Larsen-2019-07-23\config.yaml"
#peter path="\\allen\programs\braintv\workgroups\cortexmodels\peterl\visual_behavior\DLC_models\universal_eye_tracking-peterl-2019-04-25\config.yaml"

#movies to analyze
videofile_path = [r"D:\LPW\LPW\8\2.avi"] #Enter the list of videos to analyze.



#step 1) - run model on data
#------------------------------------------------------------------------------------------------
analyze_videos(path_config_file,videofile_path)

#step 2) - extract the fitted points from the deeplabcut HDf5 file, fit these to an elipse for the pupil and potentially other 'extra' points 
#----------------------------------------------------------------------------------------------------


#get full path to the extracted points from the H5
file_base=os.path.basename(videofile_path[0])[:-4]+'DeepCut_resnet50'
h5_path=full_file_path(os.path.dirname(videofile_path[0]), prefix=file_base,file_type='.h5', exclusion='ellipse')


#get the super column name (the title of the model)
df = pd.read_hdf(h5_path[0])

#the root column is named after the model, we drop this to make indexing easier
df.columns = df.columns.droplevel(0)

#because the user can define the names for the labels, we have to figure out what those are based on the passed in tuple indexes
pupil_names=[df.columns[cols][0] for cols in range(pupil_cols[0],pupil_cols[1]*3,3)]
cr_names=[df.columns[cols][0] for cols in range(cr_cols[0]*3,cr_cols[1]*3,3)]
extra_names=[df.columns[cols][0] for cols in range(extra_cols[0]*3,extra_cols[1]*3,3)]

idx = pd.IndexSlice

pupil_info=df[pupil_names].loc[:, idx[:, ['x','y','likelihood']]]

h5_save_name=file_base+'_eye_params.hdf5'
video_file_dir=os.path.dirname(videofile_path[0])

save_fn=os.path.join(video_file_dir,h5_save_name)

if os.path.isfile(save_fn):
        print ('H5 movie File EXISTS: Overwriting previous version with current')
        os.remove(save_fn)

dfile = h5py.File(save_fn)

save_raw_points_h5(df=df,dfile=dfile,pupil_names=pupil_names,cr_names=cr_names,extra_names=extra_names,h5_grp="raw_points",close=False)

print('\n Fitting elipses to pupil....')
pupilz = partial(fit_elipse,likelihood_thresh=likelihood_pup_thresh)
pupil_params=pupil_info.apply(pupilz,axis=1)

#pupil_params=parallelize_on_rows(data=pupil_info,func=pupilz,num_of_processes=4)

pupils_df=pd.DataFrame(list(pupil_params))
pupils_df.columns={"pup_ang":'angle',"pup_area":'area',"pup_center":'center',"pup_height":'height',"pup_width":'width'}

cr=df[cr_names].loc[:, idx[:, ['x','y']]]
cr.columns=cr.columns.droplevel(0)
cr.columns=["CR_x","CR_y"]

if fit_extra_to_elipse==True:
    extra_df=df[extra_names].loc[:, idx[:, ['x','y','likelihood']]]
       
    print('\n Fitting elipses to additional points...')
    extraz = partial(fit_elipse,likelihood_thresh=likelihood_oth_thresh)

    extra_params=extra_df.apply(extraz,axis=1)
    extra=pd.DataFrame(list(extra_params))
    extra.columns={"ext_ang":'angle',"ext_area":'area',"ext_center":'center',"ext_height":'height',"ext_width":'width'}

#if the other points aren't fit to the elipse, then just save the individual points  
else:
    
    extra=df[extra_names].loc[:, idx[:, ['x','y']]]
    extra.columns= ['%s%s' % (a, '_%s' % b if b else '') for a, b in extra.columns]

pupils_df=pd.concat([pupils_df,cr,extra],axis=1)
   

pd_df_to_hdf5_dset(df=pupils_df,df_columns=["pup_ang","pup_area","pup_center","pup_height","pup_width"],
                   hdf_dfile=dfile,hdf_names=[],grp='pupil')

pd_df_to_hdf5_dset(df=pupils_df,df_columns=list(cr.columns),
                   hdf_dfile=dfile,hdf_names=[],grp='cr')

pd_df_to_hdf5_dset(df=pupils_df,df_columns=list(extra.columns),
                   hdf_dfile=dfile,hdf_names=[],grp='extra')

dfile.close()

#step 3 produce annotated movie
annotate_movie(videofile_path=videofile_path[0],pupils_df=pupils_df,fit_extra_to_elipse=fit_extra_to_elipse,extras_points_plot=extra.columns)

