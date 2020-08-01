# PF-OCE imaging reconstruction and processing 
import os
from matplotlib.pyplot import figure, step 
import numpy as np 
from PyOCT import PyOCTRecon 
from PyOCT import misc
from PyOCT import CAO 
import matplotlib.pyplot as plt 
import matplotlib 
import re 
import h5py 
import time
from progress.bar import Bar

time01 = time.time()

# set font of plot 
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica']
font = {'weight': 'normal',
        'size'   : 8}
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def Run_test():
    root_dir = os.path.join(os.path.realpath(__file__)[:-24],'example_dataset')  # root_dir where all dataset has been located 
    save_folder = "OCTImage" # folder to save processed OCT image dataset, under root_dir
    TruncProcess = False # using Bactch processing to trunc data set into small segmentation to be processed 

    # find all dataset and settings 
    OCT_NumOfFile,OCT_RawDataFileID, OCT_BkgndFileID, OCT_SettingsFileID, OCT_savePath = misc.find_all_dataset(root_dir,save_folder)

    # setting parameters 
    Ascans = 256
    Frames = 256 
    depths = np.arange(0,1024,step=1) + 1024 #all depths to reconstruct

    alpha2 = -42 # alpha dispersion correction coefficient 2
    alpha3 = 40 # alpha dispersion correction coefficient 3


    # start processing all dataset under root_dir
    bar = Bar(' OCT Imaging Processing', max=OCT_NumOfFile)
    for i in range(OCT_NumOfFile):
        bar.next()
        print(" ")
        SampleID = OCT_RawDataFileID[i] 
        BkgndID = OCT_BkgndFileID[0] 
        if TruncProcess:
            data = PyOCTRecon.Batch_OCTProcessing(Ascans=Ascans,Frames=Frames,ChunkSize=128,root_dir = root_dir,sampleID=SampleID,bkgndID =BkgndID,saveOption=False,saveFolder='OCEProcessedData',RorC='complex',verbose=False,frames=Frames,alpha2=alpha2,alpha3=alpha3,depths=depths,ReconstructionMethods='CAO') #camera_matrix=camera_matrix    else:
        else:
            OCTVoltmp = PyOCTRecon.OCTImagingProcessing(root_dir = root_dir,sampleID=SampleID[0:-8],bkgndID=BkgndID,saveOption=False,saveFolder='OCTProcessedData',RorC='complex',verbose=False,frames=Frames,alpha2=alpha2,alpha3=alpha3,depths=depths,ReconstructionMethods='CAO')
            data = np.reshape(OCTVoltmp.data,(np.size(depths),Ascans,Frames))
            if i == 0:
                OCTVoltmp.ShowXZ(np.abs(data))
                Settings = OCTVoltmp.Settings 


    # option could also save data into anther folder 
    misc.SaveData(OCT_savePath,"ProcessedtestData",data,datatype='data',varName='OCTData') 
    misc.SaveData(OCT_savePath,"Settings",Settings,datatype='settings',varName='Settings')


    # loading dataset could be a trivial work by using h5py 
    fid = h5py.File(OCT_savePath+'/ProcessedtestData.hdf5', 'r') 
    Settings = misc.LoadSettings(OCT_savePath,"Settings.hdf5") 
    data = fid['OCTData_real'][()] + 1j*fid["OCTData_imag"][()] # for real datatype processing, only using fid['OCTData'][()]
    print(np.shape(data)) 
    fid.close()


    # CAO processing 
    print("S1: Coherence gate remove ...")
    data, Settings = CAO.CoherenceGateRemove(data,Settings,verbose=True,proctype='oct')
    #print("S2: Focal plane registration...") 
    # # since we perform coherenceGateRemove over the whole volume data, it is non-necessary to perform focal plane curvature removal. 
    print("S2: Phase Registration ...")
    data, Settings = CAO.PhaseRegistration(data,Settings)
    print("S3: Defocusing ...")
    data, Settings = CAO.ViolentDefocus(data,Settings,showFitResults=True,proctype='oct')

    figCAO = plt.figure(constrained_layout=False,figsize=(10,5))
    #plt.imshow(np.squeeze(np.amax(np.abs(data)**0.4,axis=2)),cmap='gray')
    OCTVoltmp.ShowXZ(np.abs(data),figHandle=figCAO) 

    figFace = plt.figure(figsize=(4,4))
    dataFace = np.squeeze(np.abs(data[int(Settings['zf']/Settings['zPixSize']),:,:]))
    plt.imshow(dataFace,cmap='rainbow') 
    figFace.suptitle("OCT en-face at focal plane")

    time02 = time.time()
    print("Total Reconstruction time is {} sec".format(time02-time01)) 
    plt.show() 



