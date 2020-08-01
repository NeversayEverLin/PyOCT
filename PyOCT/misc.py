import os
from h5py._hl.files import File 
import numpy as np 
import xml.etree.ElementTree as ET
import time
from scipy.linalg import dft
import numpy.matlib 
import matplotlib.pyplot as plt 
import matplotlib 
from PyOCT import CAO 
import re 
import h5py
from scipy.linalg.misc import norm 
from scipy.signal import fftconvolve
import matplotlib.patches as patches

def find_all_dataset(root_dir,saveFolder, saveOption='out'):
    """
    Looking for all datasets under root_dir and create a saveFolder under root_dir for data save. 
    : root_dir: root directory of all data files
    : saveFolder: name of folder where the data should be saved 
    Return:
    : NumOfFile: total of raw data files 
    : RawDataFileID: sorted raw data file ID
    : SettingsFileID: sorted settings file ID of corresponding raw data file 
    : BkgndFileID: background data file 
    : save_path: the path to save data 
    : saveOption: 'in' or 'out', indicating save the processed files into current root directory with folder name as saveFolder ('in')
    :               or save the processed files into an independent directory with saveFolder as a full directory path. 
    """
    if saveOption.lower() == 'in':
        save_path = os.path.join(root_dir,saveFolder) 
    elif saveOption.lower() == 'out':
        save_path = saveFolder 
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    subfolders = os.listdir(root_dir)
    SettingsFileID = [] 
    RawDataFileID = []
    BkgndFileID = []
    for item in subfolders:
        if item.endswith('_settings.xml'):
            SettingsFileID.append(item) 
        if item.endswith('_raw.bin'):
            if 'bkgnd' not in item:
                RawDataFileID.append(item)
            else:
                BkgndFileID.append(item) 
    # sort file name by numerical order
    pattern = re.compile(r'_\d+_') 
    pattern2 = re.compile(r'\d+')
    RawDataFileID = sorted(RawDataFileID, key=lambda x:int(pattern2.findall(pattern.findall(x)[0])[0]))
    SettingsFileID = sorted(SettingsFileID , key=lambda x:int(pattern2.findall(pattern.findall(x)[0])[0]))
    NumOfFile = len(RawDataFileID) 
    return NumOfFile, RawDataFileID, BkgndFileID, SettingsFileID, save_path


def SaveData(save_path,FileName,inData,datatype='data',varName = 'OCTData'):
    """
    Save data in the format of .hdf5
    : save_path: directory path where the data will be saved. 
    : FileName: name of file name. Therefore, the file will be FileName.hdf5 
    : inData: input data. This should be an ndarray or Settings file. 

    """
    if datatype.lower() == 'data':
        if np.iscomplexobj(inData):
            DataFileSave = h5py.File(save_path+'/'+FileName+'.hdf5','w')
            DataFileSave.create_dataset(varName+'_real',shape=np.shape(inData),data=np.real(inData),compression="gzip")
            DataFileSave.create_dataset(varName+'_imag',shape=np.shape(inData),data=np.imag(inData),compression="gzip")
            DataFileSave.close()
        else:
            DataFileSave = h5py.File(save_path+'/'+FileName+'.hdf5','w')
            DataFileSave.create_dataset(varName,shape=np.shape(inData),data=inData,compression="gzip")
            DataFileSave.close()
    elif datatype.lower() == 'settings':
        SettingsFile = h5py.File(save_path+'/'+FileName+'.hdf5','w')
        for k, v in inData.items():
            SettingsFile.create_dataset(k,data=v)
        SettingsFile.close()         
    else:
        raise ValueError("Wrong data type!")    

def LoadSettings(path,FileName):
    """
    Loading Settings file. 
    path should NOT end with "/" or "\\".
    """
    Settings = dict.fromkeys([], []) 
    fid = h5py.File(path+'/'+FileName,'r')
    for key in fid.keys():
        Settings[key] = fid[key][()]
    return Settings 


def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    """Calculating correlation coefficient between two input 2D array
    with same definition to corr2() in MATLAB
    """
    a = a - mean2(a)
    b = b - mean2(b)
    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum())
    return np.abs(r)


def normxcorr2(template, image, mode="full"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs. 
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)
    
    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    
    return out

def Max2d(inData):
    return np.amax(inData), np.unravel_index(inData.argmax(),inData.shape) 

def patternMatch(template,rootImage,cropIndex = None, showFit = False):
    """Compare template image to rootImage and find the translation index required 
    for makeing template image matched with rootImage. That's by moving (transX,transY) to ensure 
    template image as much similar as to rootImage. Using normxcorr2() method which requires a small image region cropped from template.
    Therefore, cropIndex means the subimage of template used to deconvlve with rootImage. If cropIndex is None, then directly using template image as subimage.
    : template: to be compared, 2d numpy array as real. if cropIndex is None, both dimensions of template image must be smaller than rootImage. Using cropIndex must result in a smaller dimension of subimage compared to rootImage. 
    : rootImage: basic image, 2d nump.array as real. It is best template and rootImage has the same 
    : cropIndex: None as default, or (4,) list/array with [xmin,xmax,ymin,ymax]. 
    : showFit: present fit results 
    """
    if cropIndex == None:
        CropImage = template 
        centerofCropInTemplate = (0,0)
    else:
        cropIndex = np.asarray(cropIndex) 
        CropImage = template[cropIndex[0]:cropIndex[1],cropIndex[2]:cropIndex[3]]
        centerofCropInTemplate = (int(np.ceil((cropIndex[1]+cropIndex[0])/2)), int(np.ceil((cropIndex[2]+cropIndex[3])/2)))
    cTmp = normxcorr2(CropImage,rootImage,mode='same') 
    cMax, cPos = Max2d(cTmp) 
    transX, transY = (cPos[0] - centerofCropInTemplate[0],  cPos[1] - centerofCropInTemplate[1])
    if showFit:
        figC = plt.figure(figsize=(14,4))
        ax00 = plt.subplot2grid((1,3),(0,0),rowspan=1,colspan=1) 
        ax01 = plt.subplot2grid((1,3),(0,1),rowspan=1,colspan=1) 
        ax02 = plt.subplot2grid((1,3),(0,2),rowspan=1,colspan=1) 
        imax0 = ax00.imshow(cTmp,aspect='equal') 
        ax01.imshow(rootImage,aspect='equal',cmap='gray')
        ax02.imshow(template,aspect='equal',cmap='gray') 
        figC.colorbar(imax0,ax=ax00,orientation='vertical',fraction=0.05,aspect=50)

        figT = plt.figure(figsize=(5,5))
        axT = plt.subplot2grid((1,2),(0,0),rowspan=1,colspan=1) 
        axT2 = plt.subplot2grid((1,2),(0,1),rowspan=1,colspan=1) 
        axT.imshow(rootImage,cmap='gray',aspect='equal',interpolation = 'none')
        rect = patches.Rectangle((cPos[1]-np.shape(CropImage)[1]/2,cPos[0]-np.shape(CropImage)[0]/2), np.shape(CropImage)[1], np.shape(CropImage)[0], fill=False,linestyle='--',linewidth=2,edgecolor='tab:red')
        axT.add_patch(rect)
        axT2.imshow(CropImage,cmap='gray',aspect='equal',interpolation='none')

    return cTmp, cMax, transX, transY 


def filter_bilateral( img_in, sigma_s, sigma_v, reg_constant=1e-8 ):
    """Simple bilateral filtering of an input image

    Performs standard bilateral filtering of an input image. If padding is desired,
    img_in should be padded prior to calling

    Args:
        img_in       (ndarray) monochrome input image
        sigma_s      (float)   spatial gaussian std. dev.
        sigma_v      (float)   value gaussian std. dev.
        reg_constant (float)   optional regularization constant for pathalogical cases

    Returns:
        result       (ndarray) output bilateral-filtered image

    Raises: 
        ValueError whenever img_in is not a 2D float32 valued numpy.ndarray
    """

    # check the input
    if not isinstance( img_in, numpy.ndarray ) or img_in.dtype != 'float32' or img_in.ndim != 2:
        raise ValueError('Expected a 2D numpy.ndarray with float32 elements')

    # make a simple Gaussian function taking the squared radius
    gaussian = lambda r2, sigma: (numpy.exp( -0.5*r2/sigma**2 )*3).astype(int)*1.0/3.0

    # define the window width to be the 3 time the spatial std. dev. to 
    # be sure that most of the spatial kernel is actually captured
    win_width = int( 3*sigma_s+1 )

    # initialize the results and sum of weights to very small values for
    # numerical stability. not strictly necessary but helpful to avoid
    # wild values with pathological choices of parameters
    wgt_sum = numpy.ones( img_in.shape )*reg_constant
    result  = img_in*reg_constant

    # accumulate the result by circularly shifting the image across the
    # window in the horizontal and vertical directions. within the inner
    # loop, calculate the two weights and accumulate the weight sum and 
    # the unnormalized result image
    for shft_x in range(-win_width,win_width+1):
        for shft_y in range(-win_width,win_width+1):
            # compute the spatial weight
            w = gaussian( shft_x**2+shft_y**2, sigma_s )

            # shift by the offsets
            off = numpy.roll(img_in, [shft_y, shft_x], axis=[0,1] )

            # compute the value weight
            tw = w*gaussian( (off-img_in)**2, sigma_v )

            # accumulate the results
            result += off*tw
            wgt_sum += tw

    # normalize the result and return
    return result/wgt_sum