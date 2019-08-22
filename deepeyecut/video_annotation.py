# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:58:53 2019

@author: rylanl
"""
import cv2
import os



def annotate_movie(videofile_path,pupils_df,fit_extra_to_elipse=True, extras_points_plot=[],print_error_col='',display=True):
    
    file_base=os.path.basename(videofile_path)[:-4]+'_DeepCut_resnet50'
        
    cap = cv2.VideoCapture(videofile_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    save_mov_name=os.path.join(os.path.dirname(os.path.realpath(videofile_path)),(file_base+'_annotated_.avi'))
                               
    out = cv2.VideoWriter(save_mov_name,fourcc, cap.get(5), (int(cap.get(3)),int(cap.get(4))))
    
    color              = (0,0,255)
    lineType               = cv2.LINE_AA
    thickness=2
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    curr_frame=0
    print ('\n annotating movie: ')
    while(curr_frame<total):
    
        # read the frames
        _,frame = cap.read()
    
        #get the pupil params
        if any(pupils_df[['pup_height','pup_width','pup_center','pupil_x','pupil_y']].iloc[curr_frame].isnull()) or pupils_df['pup_center'][curr_frame][0]!=pupils_df['pup_center'][curr_frame][0]:
            print('NaN: skipping pupil fit for frame ' + str(curr_frame))
        else:
            width=int(pupils_df['pup_width'][curr_frame])
            height=int(pupils_df['pup_height'][curr_frame])
            angle=pupils_df['pup_ang'][curr_frame]
        
            #plot pupil
            cv2.ellipse(frame,(int(pupils_df['pup_center'][curr_frame][0]),int(pupils_df['pup_center'][curr_frame][1])),(width,height),angle,0,360,255,lineType=lineType,thickness=1)
        
        #draw CV circle for corneal reflection
        cv2.circle(frame,(int(pupils_df['CR_x'][curr_frame]),int(pupils_df['CR_y'][curr_frame])),radius=5,color=color,thickness=thickness,lineType=lineType)                                                                      
        
        if fit_extra_to_elipse==True:
            #check for nan in fits:
            if pupils_df['ext_width'][curr_frame] !=pupils_df['ext_width'][curr_frame]:
                print('NaN: skipping extra fit for frame ' + str(curr_frame))
                #if not nan, then plot the fit
                
            else:
                ext_width=int(pupils_df['ext_width'][curr_frame])
                ext_height=int(pupils_df['ext_height'][curr_frame])
                ext_ang=int(pupils_df['ext_ang'][curr_frame])
            
                cv2.ellipse(frame,(int(pupils_df['ext_center'][curr_frame][0]),int(pupils_df['ext_center'][curr_frame][1])),(ext_width,ext_height),ext_ang,0,360,color=(0,255,0),lineType=lineType,thickness=1)
        else:
            #plot extra points if ellipse fitting not performed
            for kk in range(0,int(len(extras_points_plot)),2):
    
                x=int(pupils_df[extras_points_plot].loc[curr_frame][kk])
                y=int(pupils_df[extras_points_plot].loc[curr_frame][kk+1])
            
                cv2.circle(frame,(x,y),radius=4,color=(0,255,0),thickness=-1,lineType=-1)
                #cv2.putText(frame,'X', (x,y), font, fontScale, fontColor, lineType)
          
        if print_error_col:
            err=round(pupils_df[print_error_col][curr_frame], 1)
            err_font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,str(err),(10,50), err_font, 1,(255,255,255),2)
            
                                                                                                                                                      
        out.write(frame)
        curr_frame+=1
        #if key pressed is 'Esc', exit the loop
        if display:
            cv2.imshow('frame',frame)
        if curr_frame%1000==0:
            print ('finished annotating frame number : ' + str(curr_frame) + ' of ' + str(total))
    
        if cv2.waitKey(33)== 27:
            break
            
    
    out.release()
    
    cv2.destroyAllWindows()