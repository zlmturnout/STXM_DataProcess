import os,sys,time
import h5py,cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import rbf,griddata


def median_filter(matrix:np.array([]),filter_N:int=3):
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
    corrected_counts=matrix_counts+matrix_ref.mean()-matrix_ref
    #corrected_counts=median_filter(corrected_counts,3)
    
    print(corrected_counts)
    pos_x=np.array(h5_data['positon 1 data'],dtype=np.float32)
    pos_y=np.array(h5_data['positon 2 data'],dtype=np.float32)
    idx_x=np.argsort(pos_x)
    #print(pos_x,pos_y)
    correct_x=np.array(list(map(lambda x, y: y[x], idx_x, pos_x)))
    correct_counts=np.array(list(map(lambda x, y: y[x], idx_x, matrix_counts)))
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


if __name__=="__main__":
    #test_h5=os.path.abspath("h5_data\\SSF20221202080751interp.hdf5")
    
    #test_h5=os.path.abspath("h5_data\\SF20221202061732.h5")
    test_h5=os.path.abspath("h5_data\\SF20221205224716.h5")
    
    h5_data,correct_x,correct_counts,new_counts=read_h5(test_h5)
    Image_count=h5_data['PMT counter'][:]
    #print(f'original:\n{Image_count}')
    row_n,col_n=Image_count.shape
    
    # for i in range(row_n):
    #     if i%2==1:
    #         line=Image_count[i].copy()
    #         for j in range(col_n):
    #             Image_count[i][j]=line[col_n-j-1]
    #             #new_line.append(line[row_n-j-1])
    # print(f'new:\n{Image_count}')
    Image_ref=h5_data['PMT ref']
    Position_1=h5_data['positon 1 data']
    Position_2=h5_data['positon 2 data']
    matrix_count = np.array(Image_count,dtype=np.float32)
    matrix_ref = np.array(Image_ref,dtype=np.float32)
    matrix_pos1 = np.array(Position_1,dtype=np.float32)
    matrix_pos2 = np.array(Position_2,dtype=np.float32)
    fig1 = plt.figure(figsize =(16, 9))
    fig1.canvas.manager.window.setWindowTitle("Visualize raw image")
    
    #plt.subplot(2,2,1),plt.imshow(correct_counts[1::2],cmap=cm.rainbow,vmin=10,vmax=400)
    plt.subplot(2,2,1),plt.scatter(x=matrix_pos1,y=matrix_pos2,c=Image_count,s=10,cmap=cm.rainbow)
    plt.colorbar(location='right', fraction=0.1),plt.title("PMT counter")
    
    plt.subplot(2,2,2),plt.imshow(new_counts,cmap=cm.rainbow),plt.title("new counts")
    #plt.subplot(2,2,2),plt.scatter(x=matrix_pos1,y=matrix_pos2,c=Image_ref,s=10,cmap=cm.rainbow),plt.title("PMT ref")
    plt.colorbar(location='right', fraction=0.1)
    plt.subplot(2,2,3),plt.imshow(correct_x[1::2],cmap=cm.rainbow),plt.title("Pos X_odd")
    plt.colorbar(location='right', fraction=0.1)
    plt.subplot(2,2,4),plt.imshow(correct_x[::2],cmap=cm.rainbow),plt.title("Pos X_even")
    #plt.subplot(2,2,4),plt.imshow(matrix_pos2,cmap=cm.rainbow),plt.title("PMT Y")
    plt.colorbar(location='right', fraction=0.1)
    # #figure pos line
    # fig2 = plt.figure(figsize =(16, 9))
    # fig2.canvas.manager.window.setWindowTitle("Visualize raw image")
    # x_list=[i for i in range(col_n)]
    # plt.subplot(2,2,1),plt.plot(x_list,matrix_pos1[0])
    # plt.subplot(2,2,2),plt.plot(x_list,matrix_pos1[2])
    # plt.subplot(2,2,3),plt.plot(x_list,matrix_pos1[4])
    # plt.subplot(2,2,4),plt.plot(x_list,matrix_pos1[6])
    fig3 = plt.figure(figsize =(16, 9))
    fig3.canvas.manager.window.setWindowTitle("Visualize raw image")
    #plt.subplot(1,1,1),plt.scatter(x=matrix_pos1,y=matrix_pos2,c=Image_count,s=10,cmap=cm.rainbow)
    plt.subplot(1,1,1),plt.imshow(new_counts,cmap=cm.Greys),plt.title("new counts")
    plt.colorbar(location='right', fraction=0.1),plt.title("all counts")
    plt.show()

    print("OK")
