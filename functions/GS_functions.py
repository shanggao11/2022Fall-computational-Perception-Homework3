'''@Author: Shang Gao  * @Date: 2022-09-28 18:31:20  * @Last Modified by:   Shang Gao  * @Last Modified time: 2022-09-28 18:31:20 '''
import os
import sys
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch
import shutil
# from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
# import Ranger
# from audtorch.metrics.functional import pearsonr
# from pytictoc import TicToc
# import imshowtools 
# sys.path.append(where you put this file),from GS_functions import GF
# sys.path.append('/user_data/shanggao/tang/'),from GS_functions import GF
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
# t = TicToc()
import numpy as np
import scipy
import cv2


class GF:
    def get_VE(pred,real):
        pred=pred.flatten()
        real=real.flatten()
        assert pred.shape==real.shape
        VE=1 - np.var(pred - real) / (np.var(real))
        return VE
    def get_model_rsp(model,img_subp_mat,batch_size=512,device='cpu',norm_1=0):
        '''
        img_subp_mat shape: (batchnum, 1, subcropsize, subcropsize), e.g. (44540, 1, 50, 50)
        '''
        if norm_1==1:
            img_subp_mat=GF.norm_to_1(img_subp_mat)
        img_subp_mat=torch.tensor(img_subp_mat,dtype=torch.float)
        assert len(img_subp_mat.shape)==4
        all_rsp=[]
        valpics=GF.ImageDataset_cphw3(img_subp_mat=img_subp_mat)
        val_loader=DataLoader(valpics,batch_size=batch_size,shuffle=False)
        for num,batch_pics in enumerate(val_loader):
            with torch.no_grad():
                print(num)
                model=model.to(device)
                batch_pics=batch_pics.to(device)
                rsp=model(batch_pics)
                all_rsp.append(rsp.detach().cpu().numpy())
        all_rsp=np.vstack(all_rsp)
        return all_rsp

    class ImageDataset_cphw3(Dataset):
        def __init__(self,img_subp_mat ):
            """
            cell_start: start from 1
            mode=num_neurons/startend
            """
            self.data=img_subp_mat 
        def __len__(self):
            return self.data.shape[0]  # number of images
        def __getitem__(self, index):
            img = self.data[index]
            return img
    def crop_img_subparts(img,crop_size=50,stride=1):
        '''
        Input: img->grayscale; crop_size->crop size of subpart image you want; stride-> cropping stride. 
        crop_size is the model inpute size
        Output: return (num_of_subparts,crop_size,crop_size)
        Note: cropping is from: topleft -> topright -> bottomleft -> bottomright
        '''
        assert len(img.shape)==2
        H,W=img.shape
        assert (H,W)>(100,100)
        # if stride&1 != H&1:
        #     img=img[:-1,:]
        # if stride&1 != W&1:
        #     img=img[:,:-1]
        H,W=img.shape
        print('Image shape (H,W):',(H,W))
        H_num,W_num=GF.compute_num_ofsubpart((H,W),crop_size,stride)
        print('Number of blocks:',(H_num,W_num))
        H_stridelist=list(range(0,H,stride))
        W_stridelist=list(range(0,W,stride))
        # print(H_stridelist)
        img_subp_mat=[]
        for i in range(H_num):
            for j in range(W_num):
                # print(i)
                x0,x1=H_stridelist[i], H_stridelist[i]+crop_size
                y0,y1=W_stridelist[j], W_stridelist[j]+crop_size
                # print(y0) 
                img_R=img[x0:x1, y0:y1]
                # print(img_R.shape)
                img_subp_mat.append(img_R)
        img_subp_mat=np.stack(img_subp_mat)
        Number_of_blocks=(H_num,W_num)
        return img_subp_mat,Number_of_blocks

    def compute_num_ofsubpart(imgshape,crop_size,stride):
        assert len(imgshape)==2
        assert isinstance(imgshape,tuple)
        H,W=imgshape[0],imgshape[1]
        H_num=int((H-crop_size)/stride+1)
        W_num=int((W-crop_size)/stride+1)
        return H_num,W_num
    def slice_max(oneDarray,slice_num):
        '''
        In: oneDarray, slice_num=2
        Example: a=[1,2,54,5,6,7], slice_num=2, Out=[0,2,54,0,0,7]
        '''
        oneDarray=oneDarray.flatten()
        oneDarray=oneDarray.reshape((-1,slice_num))
        print('oneDarray reshape:',oneDarray,'\n')
        print('oneDarray shape',oneDarray.shape,'\n')
        slice_max_v=np.max(oneDarray,axis=1)
        NewArray=np.zeros(oneDarray.shape)
        for i in range(len(slice_max_v)):
            max_1row=slice_max_v[i]
            Array_1row=oneDarray[i,:]
            bool_idx=(Array_1row>=max_1row)+0
            bool_idx_final=bool_idx
            if len(np.where(bool_idx==1)[0])>=2:
                print('exist same value, choose the first one...')
                First_idx=np.where(bool_idx==1)[0][0]
                bool_idx2=np.zeros(bool_idx.shape)
                bool_idx2[First_idx]=1
                bool_idx_final=bool_idx2
            NewArray[i,:]=bool_idx_final*max_1row
            
        print('NewArray',NewArray)
        return NewArray.flatten()
        
    def make_mask(ic, jc,RFsize, img_size=(50,50),criteria='zero',gaussian_radius=11,gaussian_sigma=2.2):
        '''
        img_size:tuple
        criteria='zero'/'gaussian'/
        '''
        # set up a mask
        m = 0
        mask = np.zeros(img_size, dtype=np.float32)
        temp = np.zeros(img_size).astype(np.float32)
        for i in range(img_size[0]):
            for j in range(img_size[1]):
                if (i - ic) ** 2 + (j - jc) ** 2 < (RFsize + 1) ** 2:
                    temp[i, j] = 1
        if criteria.lower()=='gaussian':
            mask= cv2.GaussianBlur(1 - temp, (gaussian_radius, gaussian_radius), gaussian_sigma)
        elif criteria.lower()=='zero':
            mask=1-temp
        return mask
    def conv2(img,kernel,mode='same'):
        """
        From: https://www.codegrepper.com/code-examples/python/conv2+python
        Emulate the function conv2 from Mathworks.
        Usage:
        z = conv2(img,kernel,mode='same')
        - Support other modes than 'same' (see conv2.m)
        """
        img=np.array(img,dtype=np.float)
        kernel=np.array(kernel,dtype=np.float)
        if not(mode == 'same'):
            raise Exception("Mode not supported")

        # Add singleton dimensions
        if (len(img.shape) < len(kernel.shape)):
            dim = img.shape
            for i in range(len(img.shape),len(kernel.shape)):
                dim = (1,) + dim
            img = img.reshape(dim)
        elif (len(kernel.shape) < len(img.shape)):
            dim = kernel.shape
            for i in range(len(kernel.shape),len(img.shape)):
                dim = (1,) + dim
            kernel = kernel.reshape(dim)

        origin = ()

        # Apparently, the origin must be set in a special way to reproduce
        # the results of scipy.signal.convolve and Matlab
        for i in range(len(img.shape)):
            if ( (img.shape[i] - kernel.shape[i]) % 2 == 0 and
                img.shape[i] > 1 and
                kernel.shape[i] > 1):
                origin = origin + (-1,)
            else:
                origin = origin + (0,)

        z = scipy.ndimage.filters.convolve(img,np.flip(kernel), mode='constant', origin=origin)

        return z

    def pie_chart(labels,sizes,shadow=0,startangle=90):
        '''
        labels:list or np array of strings.
        sizes: list or np array of numbers.
        '''
        assert len(labels) 
        fig1, ax1 = plt.subplots(dpi=100)
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                shadow=shadow, startangle=startangle)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')  
        plt.tight_layout()

    def img2matrix(imgmainpath,suffix='.png'):
        '''
        make sure all imgs are the same size
        '''
        pathlist=GF.filelist_suffix(imgmainpath,suffix)
        im_shape=np.array(PIL.Image.open(f"{imgmainpath}/{pathlist[0]}").convert('L')).shape[0]
        im_matrix=np.zeros((len(pathlist),im_shape,im_shape))
        for i in range(len(pathlist)):
            # print('Change img:',i+1)
            im_matrix[i,:,:]= np.array(PIL.Image.open(f"{imgmainpath}/{pathlist[i]}").convert('L'))
        
        return im_matrix
            
    def show_imgs_in1Page(img_matrix,cmap='gray',showsize=(10,10),columns=None,rows=None,padding=False,title=None):
        '''
        shape: (numbers,H,W,C) or (numbers,H,W)
        '''

        # assert len(img_matrix.shape)==3
        assert isinstance(showsize,tuple)
        imshowtools.imshow(*img_matrix,cmap=cmap,size=showsize,columns=columns,rows=rows,padding=padding,title=title)



    # def gen_range(start, stop, step):
    #     """Generate list"""
    #     current = start
    #     while current < stop:
    #         next_current = current + step
    #         if next_current < stop:
    #             yield (current, next_current)
    #         else:
    #             yield (current, stop)
    #         current = next_current

    def norm_to_1(imagemat):
        """
        In: Input shape should be 4[BHWC or BCHW] or 3[CHW or HWC] or 2[HW] or 1[vector], tensor or numpy arrary.
        Out: Norm to 1 version , Batch and Channel seperate
        """
        assert (
            len(imagemat.shape) == 2
            or len(imagemat.shape) == 1
            or len(imagemat.shape) == 4,
            len(imagemat.shape) == 3,
        ), "Input shape should be 4[BHWC or BCHW] or 3[CHW or HWC] or 2[HW] or 1[vector]"
        assert isinstance(imagemat, torch.Tensor) or isinstance(
            imagemat, np.ndarray
        ), "input data should be torch tensor or numpy array"
        grad_mode = None
        if isinstance(imagemat, torch.Tensor):
            if imagemat.requires_grad:
                grad_mode = "True"
                GG = imagemat.grad
            else:
                grad_mode = "False"
                GG = None

            imagemat_new = imagemat.detach().clone()  # .detach().clone()
            print("---------------------------")
            imagemat_new = torch.tensor(imagemat_new, dtype=torch.float)  ### new line
        else:
            imagemat_new = imagemat.copy()
            imagemat_new = np.array(imagemat_new, dtype=np.float32)  ### new line
        if len(imagemat_new.shape) == 3:
            C, H, W = imagemat_new.shape
            assert (
                C == 1 or C == 3 or W == 1 or W == 3
            ), "Input should be CHW or HWC, and channel can only be 1 or 3"
            if C == 1 or C == 3:
                new_img = GF.channel_norm1(imagemat_new, mode="CHW")
            elif W == 1 or W == 3:
                new_img = GF.channel_norm1(imagemat_new, mode="HWC")
            else:
                raise RuntimeError("Check input")

        if len(imagemat_new.shape) == 2 or len(imagemat_new.shape) == 1:
            if imagemat_new.max() == imagemat_new.min():
                new_img = imagemat_new
            else:
                new_img = (imagemat_new - imagemat_new.min()) / (
                    imagemat_new.max() - imagemat_new.min()
                )

        if len(imagemat_new.shape) == 4:
            B, H, W, C = imagemat_new.shape
            assert H == 1 or H == 3 or C == 1 or C == 3, "Input should be BHWC or BCHW"
            if C == 1 or C == 3:
                mode = "HWC"
                for i in range(B):
                    imagemat_new[i, :, :, :] = GF.channel_norm1(
                        imagemat_new[i, :, :, :], mode=mode
                    )
                new_img = imagemat_new
            elif H == 1 or H == 3:
                mode = "CHW"
                for i in range(B):

                    imagemat_new[i, :, :, :] = GF.channel_norm1(
                        imagemat_new[i, :, :, :], mode=mode
                    )
                new_img = imagemat_new
            else:
                assert False == True, "Check whether your image channel is 1 or 3"
        if grad_mode == "True":
            new_img.requires_grad = True
            new_img.grad = GG
            # new_img = torch.tensor(new_img, requires_grad=True)
        return new_img

    def channel_norm1(mat, mode="CHW(HWC)"):
        if isinstance(mat, torch.Tensor):
            mat_new = mat.clone()
        else:
            mat_new = mat.copy()
        assert len(mat_new.shape) == 3, "Input shape should be 3D(CHW or HWC)"
        assert isinstance(mat_new, np.ndarray) or isinstance(
            mat_new, torch.Tensor
        ), "input should be numpy or torch tensor"
        if mode == "CHW(HWC)" or mode == "CHW":
            for i in range(mat_new.shape[0]):
                mat_new[i, :, :] = (mat_new[i, :, :] - mat_new[i, :, :].min()) / (
                    mat_new[i, :, :].max() - mat_new[i, :, :].min()
                )
            F_mat = mat_new
        elif mode == "HWC":
            for i in range(mat_new.shape[2]):
                mat_new[:, :, i] = (mat_new[:, :, i] - mat_new[:, :, i].min()) / (
                    mat_new[:, :, i].max() - mat_new[:, :, i].min()
                )
            F_mat = mat_new
        else:
            assert False == True, "Input mode: CHW or HWC"
        return F_mat

    def npy2mat(varname, npyfilepath, matsavepath):
        gg = np.load(npyfilepath)
        io.savemat(matsavepath, {varname: gg})
    def mat2npy(matfilename,varname):
        '''
        this method only works for matlab > v7 file.
        '''
        mat = io.loadmat(matfilename)
        rsp=mat[varname]
        return rsp
    def copy_allfiles(src, dest):
        """
        src:folder path
        dest: folder path
        this will not keep moving the folder to another folder
        this is moving the files in that folder to another folder
        """
        src_files = os.listdir(src)
        for file_name in src_files:
            full_file_name = os.path.join(src, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dest)

    def mkdir(mainpath, foldername):
        """
        mainpath: path you want to create folders
        foldername: foldername, str, list or tuple
        Return: the path you generate.
        """
        assert isinstance(foldername, (str, tuple, list))
        if isinstance(foldername, str):
            pathname = GF.mkdir0(mainpath, foldername)
        if isinstance(foldername, (list, tuple)):
            pathname = []
            for i in foldername:
                pathname0 = GF.mkdir0(mainpath, i)
                pathname.append(pathname0)
        return pathname

    def mkdir0(mainpath, foldername):
        if mainpath[-1] == "/" or mainpath[-1] == "\\":
            pathname = mainpath + foldername + "/"
            folder = os.path.exists(mainpath + foldername + "/")
            if not folder:
                os.makedirs(f"{mainpath}/{foldername}")
                print("Create folders ing")
                print("done !")
            else:
                print("folder existed")
        else:
            pathname = mainpath + "/" + foldername + "/"
            folder = os.path.exists(mainpath + "/" + foldername + "/")
            if not folder:
                os.makedirs(f"{mainpath}/{foldername}")
                print("Create folders ing")
                print("done !")
            else:
                print("folder already existed")
        return pathname

    def sortTC(vector, sort_mode="Top_down"):
        """
        sort_mode: Top_down/Bottom_up(default:Top_down)

        """
        
        sort_mode=sort_mode.lower()
        if sort_mode not in ("top_down", "bottom_up"):
            raise RuntimeError(
                "sort_mode args incorrect:\nPlease input:\n1.Top_down\n2.Bottom_up"
            )

        if sort_mode == "top_down":
            value = np.sort(vector)[::-1]
            index = np.argsort(vector)[::-1]
        elif sort_mode == "bottom_up":
            value = np.sort(vector)
            index = np.argsort(vector)
        return value, index

    def save_mat_file(filename, var, varname="data"):
        io.savemat(filename + ".mat", {varname: var})

# .....↓deleted