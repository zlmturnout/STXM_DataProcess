import os,sys,time
import h5py,cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import rbf,griddata
from Img_process.hist_equalization import numpy_hist_equalization,adaptive_hist_equalization

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

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift

def complex_enhancement(image, alpha=1.05, gamma=1.3, beta=2.0):
    """
    混合相位-振幅增强算法
    输入: 
        image - 输入图像(实值或复值)
        alpha - 振幅缩放因子
        gamma - 振幅非线性增强指数
        beta - 相位增强系数
    返回:
        enhanced_image - 增强后的实部图像
    """
    # 1. 构造复波函数
    if np.iscomplexobj(image):
        psi = image  # 已包含相位信息
    else:
        psi = image.astype(complex)  # 实部为图像，虚部为零
    
    # 2. 振幅增强 (非线性能量调节)
    magnitude = np.abs(psi) ** gamma
    amplitude_factor = alpha * (magnitude + 1e-6)  # 防止除以零
    
    # 3. 相位增强 (强化相位梯度)
    phase = np.angle(psi) * beta
    
    # 4. 重构复波函数
    psi_enh = amplitude_factor * np.exp(1j * phase)
    
    # 5. 转换为增强图像
    enh_image = np.real(psi_enh)
    
    return enh_image

def freq_domain_enhancement(img, alpha=1.1, gamma=1.2, beta=2.5, sigma=0.05):
    """
    带频域滤波器的增强模型
    sigma: 高通滤波器阈值 (0-1)
    """
    # 傅里叶变换
    F = fftshift(fft2(img))
    
    # 振幅谱增强
    mag = np.abs(F)**gamma
    mag_enh = alpha * mag
    
    # 相位谱增强
    phase = np.angle(F)
    phase_enh = beta * phase
    
    # 高通滤波器抑制低频背景
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows,cols))
    radius = sigma * min(rows, cols)
    mask = 1 - np.exp(-(np.square(np.arange(cols)-ccol) + 
                        np.square(np.arange(rows)-crow)[:,np.newaxis])/(2*radius**2))
    
    # 重构频谱
    F_enh = mag_enh * np.exp(1j * phase_enh) * mask
    enh_img = np.abs(ifft2(fftshift(F_enh)))
    
    return enh_img

def calc_contrast(img):
    """计算对比度指数 (物理有效性验证)"""
    h, w = img.shape
    roi1 = img[h//4:h//2, w//4:w//2].mean()  # 样品区
    roi2 = img[h//8*7:, w//8*7:].mean()      # 背景区
    return (roi1 - roi2) / (roi1 + roi2 + 1e-6)
    
if __name__=="__main__":
    data_folder=r'J:\Projects_interested\Imaging_EVs_Cell\review\Figures\STXM_RAW'
    # preE_h5file=os.path.join(data_folder,"SF20250508174335.h5")
    # mainE_h5file=os.path.join(data_folder,"SF20250508174504.h5")
    preE_h5file=os.path.join(data_folder,"SF20250508182442.h5")
    mainE_h5file=os.path.join(data_folder,"SF20250508182622.h5")
    title="EVs_mag_ad"
    #Engery_str=["525eV","540eV"]
    Engery_str=["",""]
    preE_data=read_STXM08U_h5(preE_h5file)
    mainE_data=read_STXM08U_h5(mainE_h5file)
    fig,axes=plt.subplots(2,2,figsize=(12,12))
    # img enchancement
    fre_preE=complex_enhancement(preE_data[2])
    
    fre_mainE=complex_enhancement(mainE_data[2])
    # 基础均衡化
    bit_depth=12
    eq_preE, mapping_preE = numpy_hist_equalization(preE_data[2], bit_depth)
    ad_preE = adaptive_hist_equalization(preE_data[2], grid_size=2, clip_limit=0.01, bit_depth=bit_depth)
    eq_mainE, mapping_mainE = numpy_hist_equalization(mainE_data[2], bit_depth)
    ad_mainE = adaptive_hist_equalization(mainE_data[2], grid_size=2, clip_limit=0.01, bit_depth=bit_depth)


    print(f"原始对比度: {calc_contrast(preE_data[2]):.4f}")
    print(f"均衡对比度: {calc_contrast(eq_preE):.4f}")
    print(f"自适应对比度: {calc_contrast(ad_preE):.4f}")
    print(f"原始对比度: {calc_contrast(mainE_data[2]):.4f}")
    print(f"均衡对比度: {calc_contrast(eq_mainE):.4f}")
    print(f"自适应对比度: {calc_contrast(ad_mainE):.4f}")
    

    # pre edge plot
    vlim=(860,960)
    im1=axes[0][0].scatter(x=preE_data[0],y=preE_data[1],c=preE_data[2],s=8,cmap=cm.Spectral)
    im3=axes[1][0].scatter(x=preE_data[0],y=preE_data[1],c=eq_preE,s=8,cmap=cm.Spectral)
    axes[0][0].set_xlabel("X(um)",fontsize=16)
    axes[1][0].set_xlabel("X(um)",fontsize=16)
    axes[0][0].set_ylabel("Y(um)",fontsize=16)
    axes[1][0].set_ylabel("Y(um)",fontsize=16)
    axes[0][0].text(0.85, 0.1, Engery_str[0], horizontalalignment='center',
     verticalalignment='center', transform=axes[0][0].transAxes, fontsize=18,color='black')
    axes[1][0].text(0.85, 0.1, Engery_str[0], horizontalalignment='center',
     verticalalignment='center', transform=axes[1][0].transAxes, fontsize=18,color='black')
    
    # main edge plot
    im2=axes[0][1].scatter(x=mainE_data[0],y=mainE_data[1],c=mainE_data[2],s=8,cmap=cm.Spectral)
    im4=axes[1][1].scatter(x=mainE_data[0],y=mainE_data[1],c=eq_mainE,s=8,cmap=cm.Spectral)

    axes[0][1].set_xlabel("X(um)",fontsize=16)
    axes[1][1].set_xlabel("X(um)",fontsize=16)
    axes[0][1].set_ylabel("Y(um)",fontsize=16)
    axes[1][1].set_ylabel("Y(um)",fontsize=16)
    axes[0][1].text(0.85, 0.1, Engery_str[1], horizontalalignment='center',
     verticalalignment='center', transform=axes[0][1].transAxes, fontsize=18,color='black')
    axes[1][1].text(0.85, 0.1, Engery_str[1], horizontalalignment='center',
     verticalalignment='center', transform=axes[1][1].transAxes, fontsize=18,color='black')
    # add colorbar
    cbar1=fig.colorbar(im1,ax=axes[0][0],orientation='vertical', fraction=0.1)
    cbar2=fig.colorbar(im2,ax=axes[0][1],orientation='vertical', fraction=0.1)
    cbar3=fig.colorbar(im3,ax=axes[1][0],orientation='vertical', fraction=0.1)
    cbar4=fig.colorbar(im4,ax=axes[1][1],orientation='vertical', fraction=0.1)
    # save figure
    save_fig=os.path.join(os.path.dirname(preE_h5file),f"STXM_{title}_{Engery_str[0]}_{Engery_str[1]}.png")
    plt.savefig(save_fig)

    plt.show()
