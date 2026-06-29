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

def plot_STXM08U_h5(filepath:str,I_lim:tuple=(0,500),label:str="",c_map=cm.Spectral):
    if os.path.exists(filepath) and filepath.endswith('.h5'):
        E_data=read_STXM08U_h5(filepath)
        print(np.min(E_data[2]),np.max(E_data[2]))
        # pre edge plot
        fig,ax=plt.subplots(1,1,figsize=(9,8),dpi=150)
        im1=ax.scatter(x=E_data[0],y=E_data[1],c=E_data[2],s=18,cmap=c_map,clim=I_lim)
        ax.set_xlabel("X(um)",fontsize=12)
        ax.set_ylabel("Y(um)",fontsize=12)
        ax.text(0.5, 0.1, label, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontsize=20,color='white')
        # add colorbar
        cbar1=fig.colorbar(im1,ax=ax,orientation='vertical', fraction=0.1)
        # save figure
        filename,ext=os.path.splitext(os.path.basename(filepath))
        save_fig=os.path.join(os.path.dirname(filepath),f"STXM08U_{filename}_{label}.png")
        plt.show()
        fig.savefig(save_fig)
        

if __name__=="__main__":
    data_folder=r'G:\Cryo_STXM\20241227'
    E_h5file=os.path.join(data_folder,"SF20241227043355.h5")
    vlim=(-20,40)
    plot_STXM08U_h5(E_h5file,I_lim=vlim,label="",c_map=cm.Spectral)