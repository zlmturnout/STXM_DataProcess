import os,sys,time
import h5py,cv2,math
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

def plot_STXM08U_reshape_h5(filepath:str,I_coef:tuple=(0.2,0.8),label:str="",c_map=cm.Spectral,
                            I_manual:bool=False,I_counts:tuple=(0,50)):
    if os.path.exists(filepath) and filepath.endswith('.h5'):
        h5_data,orderPMT_counts,orderPMT_ref=reshape_h5(filepath)
        min_I=np.percentile(orderPMT_counts,15) if not I_manual else I_counts[0]
        max_I=np.percentile(orderPMT_counts,95) if not I_manual else I_counts[1]
        print(min_I,max_I)
        scan_range=np.array(h5_data['range(um)'])
        Energy=np.array(h5_data['energy(eV)'])
        print(f'Scan range(um):{scan_range}\nEnergy(eV):{Energy[0]}eV')
        # imshow
        fig,ax=plt.subplots(1,1,figsize=(9,8),dpi=150)
        im1= ax.imshow(orderPMT_counts[10:-10,10:-10], interpolation='bilinear', cmap=c_map,
               origin='lower', extent=[0, scan_range[1], 0, scan_range[0]],
               vmax=min_I+I_coef[1]*(max_I-min_I), vmin=min_I+I_coef[0]*(max_I-min_I))
        ax.set_xlabel("X(um)",fontsize=12)
        ax.set_ylabel("Y(um)",fontsize=12)
        #ax.text(0.5, 0.1, label, horizontalalignment='center',
        #verticalalignment='center', transform=ax.transAxes, fontsize=20,color='white')
        # add colorbar
        cbar1=fig.colorbar(im1,ax=ax,orientation='vertical', fraction=0.1)
        # save figure
        filename,ext=os.path.splitext(os.path.basename(filepath))
        save_png=os.path.join(os.path.dirname(filepath),f"{filename}_STXM08U_{Energy[0]}eV_{label}.png")
        save_jpg=os.path.join(os.path.dirname(filepath),f"{filename}_STXM08U_{Energy[0]}eV_{label}.jpg")
        #plt.show()
        fig.savefig(save_png,dpi=300)
        fig.savefig(save_jpg,dpi=300)

def plot_STXM08U_combine_h5(folderpath:str,label:str="Combined",c_map=cm.Spectral,I_counts:tuple=(0,50)):
    # read data from folder
    Energy_data=[]
    Energy_list=[]
    for h5file in os.listdir(folderpath):
        if h5file.endswith('.h5'):
            h5file=os.path.join(folderpath,h5file)
            h5_data,orderPMT_counts,orderPMT_ref=reshape_h5(h5file)
            Energy=np.array(h5_data['energy(eV)'])[0]
            scan_range=np.array(h5_data['range(um)'])
            print(f'Scan range(um):{scan_range}\nEnergy(eV):{Energy}eV')
            Energy_data.append(orderPMT_counts)
            Energy_list.append(Energy)
    # plot by energy in order
    #Energy_list.sort()
    E_num=len(Energy_list)
    print(f'Energy number:{E_num}')
    row_n=2
    fig, axs = plt.subplots(math.ceil(E_num/row_n),row_n, figsize=(12, 9), dpi=150)
    fig.suptitle('cryo-STXM images of EVs at different energies')
    # Adjust the layout
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    full_images=[]
    for i,Energy in enumerate(Energy_list):
        ax=axs[i//row_n,i%row_n]
        im= ax.imshow(Energy_data[i][10:-10,10:-10], interpolation='bilinear', cmap=c_map,
               origin='lower', extent=[0, scan_range[1], 0, scan_range[0]], 
               vmax=I_counts[1], vmin=I_counts[0])
        ax.set_xlabel("X(um)",fontsize=12)
        ax.set_ylabel("Y(um)",fontsize=12)
        ax.set_title(f"{Energy}eV")
        #ax.text(0.5, 0.1, f"{Energy}eV", horizontalalignment='center',
        #verticalalignment='center', transform=ax.transAxes, fontsize=12,color='white')
        full_images.append(im)
        # add seperate colorbar
        # cbar1=fig.colorbar(im,ax=ax,orientation='vertical', fraction=0.1)
    # add colorbar in one
    cbar1=fig.colorbar(full_images[0],ax=axs.ravel().tolist(),orientation='vertical', fraction=0.1)
    # save figure
    save_png=os.path.join(folderpath,f"STXM08U_EnergyScan_{label}.png")
    save_jpg=os.path.join(folderpath,f"STXM08U_EnergyScan_{label}.jpg")
    fig.savefig(save_png,dpi=300)
    fig.savefig(save_jpg,dpi=300)
    plt.show()


if __name__=="__main__":
    data_folder=r'J:\Projects_interested\Imaging_EVs_Cell\STXM_EVs_methods\cryo-STXM20241227'
    #multiple files in folder
    for h5file in os.listdir(data_folder):
        if h5file.endswith('.h5'):
            h5file=os.path.join(data_folder,h5file)
            plot_STXM08U_reshape_h5(h5file,I_coef=(-0.2,1.8),label="EVs_terrain",c_map=cm.terrain)
    # for single file
    E_h5file=os.path.join(data_folder,"SF20241227051113.h5")
    #plot_STXM08U_reshape_h5(E_h5file,I_coef=(-0.2,1.5),label="EVs",c_map=cm.pink)
    # plot_STXM08U_reshape_h5(E_h5file,I_coef=(0.1,1.9),label="EVs",c_map=cm.pink,
    #                         I_manual=False,I_counts=(0,60))
    # plt.show()

    #plot combined h5 files
    h5_folder=r'J:\Projects_interested\Imaging_EVs_Cell\STXM_EVs_methods\cryo-STXM20241227\EnergyScanSingle'
    #plot_STXM08U_combine_h5(h5_folder,label="EVs_terrain",c_map=cm.terrain,I_counts=(20,140))
    #plot_STXM08U_combine_h5(h5_folder,label="EVs_terrain",c_map=cm.terrain,I_counts=(0,70))