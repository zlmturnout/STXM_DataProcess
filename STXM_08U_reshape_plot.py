import os,sys,time
import h5py,cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import rbf,griddata


def reshape_h5(filepath:str,main_key:str="FPGA control board"):
    h5_file=h5py.File(filepath,'r')
    print(h5_file.filename)
    # for key in h5_file[main_key].keys():
    #     print(f'key:{key}')
    h5_data=h5_file.get(main_key)
    matrix_counts = np.array(h5_data['PMT counter'],dtype=np.float32)
    row_n,col_n=matrix_counts.shape
    print(f'Matrix with shape:{matrix_counts.shape}')
    matrix_ref=np.array(h5_data['PMT ref']).astype(np.float32)
    pos_x=np.array(h5_data['positon 1 data'],dtype=np.float32)
    pos_y=np.array(h5_data['positon 2 data'],dtype=np.float32)
    # plot historm
    #plt.hist(matrix_counts)
    #interpolate 
    #interp_counts=interpolate.Rbf(pos_x,pos_y,matrix_counts,function='multiquadric')
    max_x,min_x=pos_x.max(),pos_x.min()
    max_y,min_y=pos_y.max(),pos_y.min()
    newpos_x=np.linspace(min_x,max_x,col_n)
    newpos_y=np.linspace(min_y,max_y,row_n)
    grid_X,grid_Y=np.meshgrid(newpos_x,newpos_y)
    # interpolate into new_data with shape(row_n,col_n) in a ordered position
    orderPMT_counts=griddata((pos_x.flatten(),pos_y.flatten()),matrix_counts.flatten(),(grid_X,grid_Y),method='cubic',fill_value=matrix_counts.mean())
    orderPMT_ref=griddata((pos_x.flatten(),pos_y.flatten()),matrix_ref.flatten(),(grid_X,grid_Y),method='cubic',fill_value=matrix_counts.mean())
    return h5_file[main_key],orderPMT_counts,orderPMT_ref

def plot_STXM08U_reshape_h5(filepath:str,I_lim:tuple=(0,500),label:str="",c_map=cm.Spectral):
    if os.path.exists(filepath) and filepath.endswith('.h5'):
        h5_data,orderPMT_counts,orderPMT_ref=reshape_h5(filepath)
        print(np.min(orderPMT_counts),np.max(orderPMT_counts))
        scan_range=np.array(h5_data['range(um)'])
        print(f'Scan range(um):{scan_range}')
        # imshow
        fig,ax=plt.subplots(1,1,figsize=(9,8),dpi=150)
        im1= ax.imshow(orderPMT_counts[10:-10,10:-10], interpolation='bilinear', cmap="RdYlBu",
               origin='lower', extent=[0, scan_range[1], 0, scan_range[0]],
               vmax=I_lim[0], vmin=I_lim[1])
        ax.set_xlabel("X(um)",fontsize=12)
        ax.set_ylabel("Y(um)",fontsize=12)
        #ax.text(0.5, 0.1, label, horizontalalignment='center',
        #verticalalignment='center', transform=ax.transAxes, fontsize=20,color='white')
        # add colorbar
        cbar1=fig.colorbar(im1,ax=ax,orientation='vertical', fraction=0.1)
        # save figure
        filename,ext=os.path.splitext(os.path.basename(filepath))
        save_fig=os.path.join(os.path.dirname(filepath),f"STXM08U_reshape_{filename}_{label}.png")
        plt.show()
        fig.savefig(save_fig)
        

if __name__=="__main__":
    data_folder=r'G:\Cryo_STXM\20241227'
    E_h5file=os.path.join(data_folder,"SF20241227043355.h5")
    vlim=(-10,50)
    plot_STXM08U_reshape_h5(E_h5file,I_lim=vlim,label="",c_map=cm.Spectral)