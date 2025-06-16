import os,sys,time
import h5py,cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import rbf,griddata


def median_filter(matrix:np.array,filter_N:int=3):
    median_matrix=cv2.medianBlur(matrix, filter_N)
    return median_matrix

def read_h5(filepath:str,main_key:str="FPGA control board"):
    h5_file=h5py.File(filepath,'r')
    print(h5_file.filename)
    for key in h5_file[main_key].keys():
        print(f'key:{key}')
    h5_data=h5_file.get(main_key)
    matrix_counts = np.array(h5_data['PMT counter'],dtype=np.float32)
    row_n,col_n=matrix_counts.shape
    print(f'Matrix with shape:{matrix_counts.shape}')
    matrix_ref=np.array(h5_data['PMT ref'],dtype=np.float32)
    corrected_counts=matrix_counts
    # corrected the  unnormal counts by topup mode
    #method 1 -correct backround by division
    # corrected_counts=(matrix_counts/matrix_ref)*matrix_ref.mean()
    
    #method 2 -median_filter
    #corrected_counts=median_filter(matrix_counts,5)
    
    #method 3 correct backround by substraction and then median_filter
    #corrected_counts=matrix_counts+matrix_ref.mean()-matrix_ref
    #corrected_counts=median_filter(corrected_counts,3)
    
    print(corrected_counts)
    pos_x=np.array(h5_data['positon 1 data'],dtype=np.float32)
    pos_y=np.array(h5_data['positon 2 data'],dtype=np.float32)
    idx_x=np.argsort(pos_x)
    #print(pos_x,pos_y)
    correct_x=np.array(list(map(lambda x, y: y[x], idx_x, pos_x)))
    correct_counts=np.array(list(map(lambda x, y: y[x], idx_x, matrix_counts)))# wrong pos
    # print(correct_counts)
    # print(matrix_counts)
    #interpolate 
    #interp_counts=interpolate.Rbf(pos_x,pos_y,matrix_counts,function='multiquadric')
    max_x,min_x=pos_x.max(),pos_x.min()
    max_y,min_y=pos_y.max(),pos_y.min()
    newpos_x=np.linspace(min_x,max_x,col_n)
    newpos_y=np.linspace(min_y,max_y,row_n)
    grid_X,grid_Y=np.meshgrid(newpos_x,newpos_y)
    #print(newpos_x,newpos_y)
    #new_counts=interp_counts(newpos_x,newpos_y)
    new_counts=griddata((pos_x.flatten(),pos_y.flatten()),corrected_counts.flatten(),(grid_X,grid_Y),method='cubic',fill_value=corrected_counts.mean())
    print(new_counts)
    print(new_counts.shape)
    #new_counts=matrix_counts
    return h5_file[main_key],correct_x,correct_counts,new_counts

def read_hdf5(filepath:str):
    h5_file=h5py.File(filepath,'r')
    print(h5_file.filename)
    for key in h5_file.keys():
        print(f'key:{key}')
    
    return h5_file

def read_STXM08U_h5(filepath:str,main_key:str="FPGA control board"):
    h5_file=h5py.File(filepath,'r')
    print(h5_file.filename)
    
    h5_data=h5_file.get(main_key)
    matrix_counts = np.array(h5_data['PMT counter'],dtype=np.float32)
    row_n,col_n=matrix_counts.shape
    print(f'Matrix with shape:{matrix_counts.shape}')
    matrix_ref=np.array(h5_data['PMT ref']).astype(np.float32)
    pos_x=np.array(h5_data['positon 1 data'],dtype=np.float32)
    pos_y=np.array(h5_data['positon 2 data'],dtype=np.float32)
    return pos_x-pos_x.min(),pos_y-pos_y.min(),matrix_counts,matrix_ref

if __name__=="__main__":
    preE_h5file=os.path.abspath("h5_data\\STXM_RAW\\SF20250508175217.h5")
    mainE_h5file=os.path.abspath("h5_data\\STXM_RAW\\SF20250508175618.h5")
    #preE_h5file=os.path.abspath("h5_data\\STXM_RAW\\SF20250508191020.h5")
    #mainE_h5file=os.path.abspath("h5_data\\STXM_RAW\\SF20250508191145.h5")
    title="STXM08U_10um"
    Engery_str=["525eV","540eV"]
    preE_data=read_STXM08U_h5(preE_h5file)
    mainE_data=read_STXM08U_h5(mainE_h5file)
    fig,axes=plt.subplots(1,2,figsize=(18,6))
    # pre edge plot
    vlim=(821,927)
    im1=axes[0].scatter(x=preE_data[0],y=preE_data[1],c=preE_data[2],s=8,cmap=cm.Spectral)
    axes[0].set_xlabel("X(um)",fontsize=16)
    axes[0].set_ylabel("Y(um)",fontsize=16)
    axes[0].text(0.85, 0.1, Engery_str[0], horizontalalignment='center',
     verticalalignment='center', transform=axes[0].transAxes, fontsize=18,color='black')
    # main edge plot
    im2=axes[1].scatter(x=mainE_data[0],y=mainE_data[1],c=mainE_data[2],s=8,cmap=cm.Spectral)
    axes[1].set_xlabel("X(um)",fontsize=16)
    axes[1].set_ylabel("Y(um)",fontsize=16)
    axes[1].text(0.85, 0.1, Engery_str[1], horizontalalignment='center',
     verticalalignment='center', transform=axes[1].transAxes, fontsize=18,color='black')
    # add colorbar
    cbar=fig.colorbar(im1,ax=axes,orientation='vertical', fraction=0.1)
    # save figure
    save_fig=os.path.join(os.path.dirname(preE_h5file),f"STXM_{Engery_str[0]}_{Engery_str[1]}.png")
    plt.savefig(save_fig)

    plt.show()
