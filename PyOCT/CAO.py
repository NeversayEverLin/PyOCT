# PF-OCE imaging reconstruction and processing 
import os 
import numpy as np 
from PyOCT import PyOCTRecon 
import matplotlib.pyplot as plt 
import matplotlib 
import re 
import scipy
import h5py 
from scipy import ndimage
# set font of plot 
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica']
font = {'weight': 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# define directory info
#root_dir = 'Z:/home/yl3248/data/Feb24_2020_Collagen/LuProtocol_P1' 

# import data 
#FullVolumDataFile = h5py.File(root_dir+'/TestDataSet/FullVolumOCTData_Complex.hdf5', 'r') 
#VolumeOCT = FullVolumDataFile['OCTData_real'][()] + 1j*FullVolumDataFile['OCTData_imag'][()] 
#FullVolumDataFile.close()

#import settings
#SettingsFile = h5py.File(root_dir+'/OCTProcessedData/Settings.hdf5','r')
#Settings = {}
#for keys in SettingsFile.keys():
#        #print(keys)
#        Settings[keys] = SettingsFile[keys][()]

def QuadraticFit2d(inData,Settings, verbose=True):
        """
        Quadratic 2D fit with coefficients as 
        a0 + a1*x + a2*y + a3*x**2 + a4*x**2*y + a5*x**2*y**2 + a6*y**2 +  a7*x*y**2 + a8*x*y
        Here only considering one single time point fit. 
        : inData: [Z,X,Y] data, complex or abs.  
        : Settings: settings dictionary 
        """
        [sizeZ,sizeX,sizeY] = np.shape(inData)
        x = np.arange(0,sizeX,step=1) 
        y = np.arange(0,sizeY,step=1)  
        surfData = np.squeeze(np.argmax(np.abs(inData),axis=0)) 
        X,Y = np.meshgrid(x,y,copy=False) 
        X = X.flatten()
        Y = Y.flatten()
        A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
        B = surfData.flatten()
        coeff, r, rank, _ = np.linalg.lstsq(A, B,rcond=-1) # rcond = -1, using machine precision as to define zeros. rcond = None means that values smaller than machine precision * Max of A will be considered as zeros. 
        if verbose:
                print("QuadraticFit2d results:")
                print("coeff0: {}".format(coeff[0]))
                print("r : {}".format(r))
        zxy = coeff[1]*X+coeff[2]*Y+coeff[3]*X**2 + coeff[4]*X**2*Y+coeff[5]*X**2*Y**2+coeff[6]*Y**2+coeff[7]*X*Y**2+coeff[8]*X*Y
        zxy = np.reshape(zxy,(sizeX,sizeY))
        print("rank: {}".format(rank))
        #  print("s: {}".format(s)) 
        return coeff[0], zxy  


def SearchingCoverGlass(inData,Settings,start_index = 5, end_index = 150, verbose = True):
        from scipy.signal import find_peaks
        if inData.ndim == 3:
                tempData = np.squeeze(np.amax(np.amax(np.abs(inData[start_index:end_index,:,:]),axis=1,keepdims=True),axis=2,keepdims=True))
        elif inData.ndim == 2: 
                tempData = np.squeeze(np.amax(np.abs(inData[start_index:end_index,:]),axis=1,keepdims=True))
        elif inData.ndim == 1:
                tempData = np.abs(inData)
        else:
                raise ValueError("Wrong input data for SearchingCoverGlass !")
        # Method I: based on local maximum 
        tempData = ndimage.median_filter(tempData,size=3) 
        mid_index = int((start_index+end_index)/2)
        peaks = np.zeros((2,),dtype=np.int)
        peaks[0] = int(np.argmax(tempData[start_index:mid_index]) + start_index)
        peaks[1] = int(np.argmax(tempData[mid_index:end_index])+mid_index) 
        if verbose:
                print("Glass position is {} and {} pixel".format(peaks[0],peaks[1])) 
        figSG = plt.figure(figsize=(5,4))
        axSG = plt.subplot2grid((1,1),(0,0))
        axSG.plot(tempData) 
        axSG.scatter(peaks[0],tempData[peaks[0]],s=80,facecolors='none', edgecolors='tab:red',linewidths=2,linestyles='dotted')
        axSG.scatter(peaks[1],tempData[peaks[1]],s=80,facecolors='none', edgecolors='tab:red',linewidths=2,linestyles='dotted')
        axSG.set_yscale('log') 
        Settings['CoverSlip_Positions'] = peaks 
        return peaks, Settings

def PhaseRegistration(inData, Settings, dzc = 5,start_index = 5, end_index = 150):
        """
        Phase registration w.r.t coverslip phase as mitigation of phase instability 
        Pre-assumption: coverslip is flat and level without any strong scatters around
        : inData: raw complex OCT reconstructed signal, input OCT(x,y,z) data after normal imaging reconstruction.
        : Settings: dictionary of OCT settings 
        : dzc: number of pixels over which the coverglass phase will be averaged based on OCT-intensity weights
        : start_index, end_index: range of index to search position of coverslip which should cover both two flat surface of coverslip
        """
        print("Phase Registration ...")
        if "RefGlassPos" not in Settings.keys():  # if not, means not coherence gate curvature removal implemented. Then we just search the position of surf glass        
                if 'CoverSlip_Positions' not in Settings.keys():
                        peaks, Settings = SearchingCoverGlass(inData,Settings,start_index = 5, end_index = 150)
                pos_coverglass = peaks[1] 
        else:
                pos_coverglass = Settings["RefGlassPos"]
        surface_phi = np.average(inData[pos_coverglass-dzc:pos_coverglass+dzc,:,:],axis=0,weights=np.abs(inData[pos_coverglass-dzc:pos_coverglass+dzc,:,:]))
        surface_phi_2d = np.exp(-1j*np.angle(surface_phi)) #np.conj(surface_phi)
        S = np.multiply(inData,np.repeat(surface_phi_2d[np.newaxis,:,:],np.shape(inData)[0],axis=0))
        return S, Settings    

        
## S2: Bulk demodulation
def Gauss(x, amp, cen, wid, bis):
        return bis  + amp * np.exp(-(x-cen)**2 /wid)

def Gauss2( x, amp1, cen1, wid1, amp2, cen2, wid2):
    #(c1, mu1, sigma1, c2, mu2, sigma2) = params
    return amp1 * np.exp( - (x - cen1)**2.0 / (2.0 * wid1**2.0) ) + amp2 * np.exp( - (x - cen2)**2.0 / (2.0 * wid2**2.0) )

def BulkDemodulation(inData, Settings,showFitResults=False,proctype='oce'):
        """
        inData must be a full volume OCT image 
        """
        from scipy import ndimage  
        from lmfit import Model
        [Z,X,Y] = np.shape(inData)
        tempD = np.fft.fftshift(np.sum(np.abs(np.fft.fftn(np.fft.ifftshift(inData))),axis=0))
        LineX = ndimage.median_filter(np.squeeze(np.sum(tempD,axis=1)),size=3)#line profile for x-direciton
        LineY = ndimage.median_filter(np.squeeze(np.sum(tempD,axis=0)),size=3) # line profile for y-direction 
        x_X = np.arange(0,np.size(LineX),step=1)
        x_Y = np.arange(0,np.size(LineY),step=1)

        gmodel = Model(Gauss,independent_vars='x',param_names=['amp','cen','wid','bis']) 
        LineXFit = gmodel.fit(LineX, x=x_X,amp=np.amax(LineX),cen=(X+1)/2, wid=10,bis=np.amin(LineX))
        LineYFit = gmodel.fit(LineY, x=x_Y,amp=np.amax(LineY),cen=(Y+1)/2, wid=10,bis=np.amin(LineY))
        if showFitResults:
               # print("Line X Fit Results are:")
               # print(LineXFit.fit_report())
               # print("Line X Fit Results are:")
               # print(LineYFit.fit_report())
                figBulkDem = plt.figure(constrained_layout=False,figsize=(5,4)) 
                plt.subplots_adjust(wspace=0.4,hspace=0.5) 
                plt.tight_layout()
                figBulkDem.suptitle("Fit Gaussian along XY in FFT domain",fontsize=8)
                axFigBulkDem0_res = plt.subplot2grid((3,2),(2,0),rowspan=1,colspan=1)
                axFigBulkDem0_fit = plt.subplot2grid((3,2),(0,0),rowspan=2,colspan=1,sharex=axFigBulkDem0_res)                
                axFigBulkDem1_res = plt.subplot2grid((3,2),(2,1),rowspan=1,colspan=1)
                axFigBulkDem1_fit = plt.subplot2grid((3,2),(0,1),rowspan=2,colspan=1,sharex=axFigBulkDem1_res)
                
                LineXFit.plot_fit(ax=axFigBulkDem0_fit)
                LineXFit.plot_residuals(ax=axFigBulkDem0_res) 
                dely_LineX = LineXFit.eval_uncertainty(sigma=3)
                axFigBulkDem0_fit.fill_between(x_X, LineXFit.best_fit-dely_LineX, LineXFit.best_fit+dely_LineX, color="#ABABAB",label='3-$\sigma$ uncertainty band')
                axFigBulkDem0_fit.legend(fontsize=4)
                axFigBulkDem0_res.legend(fontsize=4)
                axFigBulkDem0_res.set_xlabel('X (pixel)',fontsize=10,labelpad=-1)
                axFigBulkDem0_res.set_ylabel('Residuals',fontsize=10,labelpad=-5)
                axFigBulkDem0_fit.set_ylabel('Fit Amp',fontsize=10,labelpad=0) 
                axFigBulkDem0_fit.set_xlabel('')
                axFigBulkDem0_fit.get_lines()[0].set_markerfacecolor('w')
                axFigBulkDem0_fit.get_lines()[0].set_markersize(3)
                axFigBulkDem0_fit.get_lines()[0].set_markeredgecolor('tab:red')
                axFigBulkDem0_fit.get_lines()[1].set_color('tab:orange')
                axFigBulkDem0_res.get_lines()[1].set_markerfacecolor('w')
                axFigBulkDem0_res.get_lines()[1].set_markersize(3)
                axFigBulkDem0_res.get_lines()[1].set_markeredgecolor('tab:red')
                axFigBulkDem0_res.get_lines()[0].set_color('tab:orange')
                plt.setp(axFigBulkDem0_fit.get_xticklabels(), visible=False)
                plt.setp(axFigBulkDem0_fit.get_yticklabels(), fontsize=8)
                plt.setp(axFigBulkDem0_res.get_yticklabels(), fontsize=8)
                plt.setp(axFigBulkDem0_res.get_xticklabels(), fontsize=8)
                axFigBulkDem0_fit.set_title('')
                axFigBulkDem0_res.set_title('')

                LineYFit.plot_fit(ax=axFigBulkDem1_fit)
                LineYFit.plot_residuals(ax=axFigBulkDem1_res) 
                dely_LineY = LineYFit.eval_uncertainty(sigma=3)
                axFigBulkDem1_fit.fill_between(x_Y, LineYFit.best_fit-dely_LineY, LineYFit.best_fit+dely_LineY, color="#ABABAB",label='3-$\sigma$ uncertainty band')
                axFigBulkDem1_fit.legend(fontsize=4)
                axFigBulkDem1_res.legend(fontsize=4)
                axFigBulkDem1_res.set_xlabel('X (pixel)',fontsize=10,labelpad=-1)
                axFigBulkDem1_res.set_ylabel('Residuals',fontsize=10,labelpad=-5)
                axFigBulkDem1_fit.set_ylabel('Fit Amp',fontsize=10,labelpad=0) 
                axFigBulkDem1_fit.set_xlabel('')
                axFigBulkDem1_fit.get_lines()[0].set_markerfacecolor('w')
                axFigBulkDem1_fit.get_lines()[0].set_markersize(3)
                axFigBulkDem1_fit.get_lines()[0].set_markeredgecolor('tab:red')
                axFigBulkDem1_fit.get_lines()[1].set_color('tab:orange')
                axFigBulkDem1_res.get_lines()[1].set_markerfacecolor('w')
                axFigBulkDem1_res.get_lines()[1].set_markersize(3)
                axFigBulkDem1_res.get_lines()[1].set_markeredgecolor('tab:red')
                axFigBulkDem1_res.get_lines()[0].set_color('tab:orange')
                plt.setp(axFigBulkDem1_fit.get_xticklabels(), visible=False)
                plt.setp(axFigBulkDem1_fit.get_yticklabels(), fontsize=8)
                plt.setp(axFigBulkDem1_res.get_yticklabels(), fontsize=8)
                plt.setp(axFigBulkDem1_res.get_xticklabels(), fontsize=8)
                axFigBulkDem1_fit.set_title('')
                axFigBulkDem1_res.set_title('')              
        Xpeak = LineXFit.best_values['cen']
        Ypeak = LineYFit.best_values['cen'] 
        dx = Settings['xPixSize']
        dy = Settings['yPixSize']
        Xshift = (Xpeak - np.floor((X-1)/2)) * (1/(dx*X))
        Yshift = (Ypeak - np.floor((Y-1)/2)) * (1/(dy*Y))# note the difference between Python and MATLAB when indexing median value
        x = (2*np.pi)*(1/X)*np.arange(0,X,1)*dx 
        y = (2*np.pi)*(1/Y)*np.arange(0,Y,1)*dy
        [xm,ym] = np.asarray(np.meshgrid(x,y))
        if proctype.lower()=='oce':
                phase = np.transpose(xm)*Xshift +  np.transpose(ym)*Yshift*0
        elif proctype.lower() == 'oct':
                phase = np.transpose(xm)*Xshift +  np.transpose(ym)*Yshift
        phase = phase - np.mean(phase) 
        demodulator_2d = np.exp(-1j*phase) 
        demodulator_3d = np.repeat(demodulator_2d[np.newaxis,:,:],Z,axis=0)
        Settings['qx_shift'] = Xshift # 1/um in qx domain that is shifted from center 
        Settings['qy_shift'] = Yshift # 1/um in qy domain that is shifted from center 

        return np.multiply(inData,demodulator_3d), Settings

def ObtainGridMesh(inData,Settings):
        """Find grid mesh coordinates corrsponding to spatial and frequency domain as the same dimension of input data. 
        return x,y,qx,qy in Settings. 
        return:
        inData, Settings. 
        where Settings include:
        x,y: x and y coordinates corresponding to two axis in enface of OCT image. 
        qx,qy: frequency coordinates with center pixles as 0. 
        xm,ym: 2D x,y cooridnates as same dimension of enface in OCT image. 
        """
        [Z,X,Y] = np.shape(inData) 
        dx = Settings['xPixSize']
        dy = Settings['yPixSize']
        x = (2*np.pi)*(1/X)*np.arange(0,X,1)*dx 
        y = (2*np.pi)*(1/Y)*np.arange(0,Y,1)*dy
        Settings['x'] = x
        Settings['y'] = y # in unit of um 

        # setting qx and qy array for future use 
        qx = 2*np.pi/(Settings['xPixSize']*1e-6)*(1/X)*np.arange(0,X,1) 
        qx = qx - qx[int(np.floor((X-1)/2))] 

        qy = 2*np.pi/(Settings['yPixSize']*1e-6)*(1/Y)*np.arange(0,Y,1)
        qy = qy - qy[int(np.floor((X-1)/2))] 
        Settings['qx'] = qx 
        Settings['qy'] = qy # in unit of 1/meter. 
        qxm,qym = np.meshgrid(qx,qy)
        qxm = np.transpose(qxm)
        qym = np.transpose(qym) 
        Settings['qxm'] = qxm 
        Settings['qym'] = qym 
        [xm,ym] = np.asarray(np.meshgrid(x,y))
        Settings['xm'] = np.transpose(xm)
        Settings['ym'] = np.transpose(ym)
        return Settings 


## S3: Defocus 
def SearchingFocalPlane(inData,Settings,start_bias = 50, extend_num_pixels = 240,showFitResults=True):
        """
        Search focal plane by fitting Guassian profile 
        Initially it will search from the position start_bias pixels from coverslip and extend extend_num_pixels pixels
        The pixels used to locate focal plane is: [CoverSlip_Positions + start_bias: CoverSlip_Positions + start_bias+extend_num_pixels]
        : start_bias: int, number of pixels the starting position away from covergalss slip position 
        : extend_num_pixels:int, number of pixels extended from starting position 
        Return:
        : zf:  position of focal plane in um as OPL, w.r.t zero path. 
        : Settings
        """
        from scipy import ndimage  
        from lmfit import Model
        [Z,X,Y] = np.shape(inData)
        #if 'CoverSlip_Positions' not in Settings.keys():
        #        _, Settings = SearchingCoverGlass(inData,Settings,start_index = 5, end_index = 150)
        #peaks = Settings['CoverSlip_Positions']
        #pos_cover = peaks[1] 
        if "RefGlassPos" not in Settings.keys():  # if not, means not coherence gate curvature removal implemented. Then we just search the position of surf glass        
                if 'CoverSlip_Positions' not in Settings.keys():
                        peaks, Settings = SearchingCoverGlass(inData,Settings,start_index = 5, end_index = 150)
                pos_cover = peaks[1] 
        else:
                pos_cover = Settings["RefGlassPos"]
        
        # method II using quadratic fit
        zf, _  = QuadraticFit2d(inData[(pos_cover+start_bias):(pos_cover+start_bias+extend_num_pixels),:,:],Settings,verbose=showFitResults)  
        zf = int(zf + pos_cover+ start_bias) # now this will serve as the reference position of cover glass 
        print("Focal position found at {} pixel".format(zf)) 
        zf = zf*Settings['zPixSize']
        """
        # method I using Gaussian fit 
        tmpData = np.squeeze(np.amax(np.amax(np.abs(inData[(pos_cover+start_bias):(pos_cover+start_bias+extend_num_pixels),:,:]),axis=1,keepdims=True),axis=2,keepdims=True))
        tmpData = np.log10(tmpData)
        tmpData = ndimage.median_filter(tmpData,size=3)
        init_cen = np.argmax(tmpData)
        x_z = np.arange(pos_cover+start_bias,pos_cover+start_bias+np.size(tmpData),step=1)
        # 1 Gaussian fit 
        gmodel = Model(Gauss,independent_vars='x',param_names=['amp','cen','wid','bis']) 
        LineZFit = gmodel.fit(tmpData, x=x_z,amp=0.5*np.amax(tmpData),cen=x_z[init_cen], wid=40,bis=np.amin(tmpData))
        if showFitResults:
                figFP = plt.figure(constrained_layout=False,figsize=(4,3)) 
                #plt.subplots_adjust(wspace=0.4,hspace=0.5) 
                plt.tight_layout()
                figFP.suptitle("Fit Focal plane",fontsize=8)
                axFigFP_res = plt.subplot2grid((3,1),(2,0),rowspan=1,colspan=1)
                axFigFP_fit = plt.subplot2grid((3,1),(0,0),rowspan=2,colspan=1,sharex=axFigFP_res)                
                
                LineZFit.plot_fit(ax=axFigFP_fit)
                LineZFit.plot_residuals(ax=axFigFP_res) 
                dely_LineZ = LineZFit.eval_uncertainty(sigma=3)
                axFigFP_fit.fill_between(x_z, LineZFit.best_fit-dely_LineZ, LineZFit.best_fit+dely_LineZ, color="#ABABAB",label='3-$\sigma$ uncertainty band')
                axFigFP_fit.legend(fontsize=4)
                axFigFP_res.legend(fontsize=4)
                axFigFP_res.set_xlabel('X (pixel)',fontsize=10,labelpad=-1)
                axFigFP_res.set_ylabel('Residuals',fontsize=10,labelpad=-5)
                axFigFP_fit.set_ylabel('Fit Amp',fontsize=10,labelpad=0) 
                axFigFP_fit.set_xlabel('')
                axFigFP_fit.get_lines()[0].set_markerfacecolor('w')
                axFigFP_fit.get_lines()[0].set_markersize(3)
                axFigFP_fit.get_lines()[0].set_markeredgecolor('tab:red')
                axFigFP_fit.get_lines()[1].set_color('tab:orange')
                axFigFP_res.get_lines()[1].set_markerfacecolor('w')
                axFigFP_res.get_lines()[1].set_markersize(3)
                axFigFP_res.get_lines()[1].set_markeredgecolor('tab:red')
                axFigFP_res.get_lines()[0].set_color('tab:orange')
                plt.setp(axFigFP_fit.get_xticklabels(), visible=False)
                plt.setp(axFigFP_fit.get_yticklabels(), fontsize=8)
                plt.setp(axFigFP_res.get_yticklabels(), fontsize=8)
                plt.setp(axFigFP_res.get_xticklabels(), fontsize=8)
                axFigFP_fit.set_title('')
                axFigFP_res.set_title('')
        zf = (LineZFit.best_values['cen']) *Settings['zPixSize'] # in um
        print(" Focal plane position is at {} pixel".format(int(LineZFit.best_values['cen'])))
        """
        Settings['zf'] = zf # in um
        return zf, Settings


def ViolentDefocus(inData,Settings,showFitResults=False,proctype='oce'):
        print("Defocusing...")
        # searching focal plane 
        if 'zf' not in Settings.keys():
                _,Settings = SearchingFocalPlane(inData, Settings,showFitResults=showFitResults) 
        zf = Settings['zf']

        [Z,X,Y] = np.shape(inData)
        k_m = Settings['k'] * 1e9 # here k is in the unit of 1/nm, convert to 1/m
        N = np.size(k_m) 
        kc = np. mean(k_m) 
        dz = Settings['zPixSize']*1e-6 
        if 'qxm' not in Settings.keys(): 
                Settings = ObtainGridMesh(inData,Settings)
        qxm = Settings['qxm']
        if proctype.lower() == 'oce':
                qym = Settings['qym']*0 
        elif proctype.lower() == 'oct':
                qym = Settings['qym']
        qr = np.fft.ifftshift(np.sqrt(qxm**2+qym**2)) #optimize out fftshift

        #aperture to prevent imaginary #'s as low pass filter 
        aperture = (qr <= 2*Settings['refractive_index']*kc)
        #aperture = np.ones(np.shape(qr))
        #defocus kernel 
        phase = np.sqrt(np.multiply(aperture,(2*Settings['refractive_index']*kc)**2-qr**2)) 


        #PERFORM CAO
        output = inData
        for i in range(np.shape(inData)[0]):
                planeFD = np.fft.fft2(np.squeeze(inData[i,:,:]))
                correction = np.multiply(aperture,np.exp(-1j*(dz*i-zf*1e-6)*phase))
                plane = np.fft.ifft2(np.multiply(planeFD,correction))
                output[i,:,:] = plane 

        return output, Settings


def CoherenceGateRemove(inData,Settings,verbose=True,proctype='OCE'):
        """
        Coherence gate curvature removal
        """
        if 'CoverSlip_Positions' not in Settings.keys():
                _, Settings = SearchingCoverGlass(inData,Settings,start_index = 5, end_index = 150)
        peaks = Settings['CoverSlip_Positions']
        pos_cover = peaks[1]  
        zc0, zxy = QuadraticFit2d(inData[pos_cover-20:pos_cover+20,:,:],Settings,verbose=verbose)  
        zc0 = int(zc0 + pos_cover-20) # now this will serve as the reference position of cover glass 
        print("Reference Coverglass position now will be always at {} pixel".format(zc0)) 
        Settings["RefGlassPos"] = zc0 
        [Z,X,Y] = np.shape(inData)
        k_m = Settings['k'] * 1e9 # here k is in the unit of 1/nm, convert to 1/m
        N = np.size(k_m) 
        kc = np. mean(k_m) 
        dz = Settings['zPixSize']*1e-6 
        if 'qxm' not in Settings.keys(): 
                Settings = ObtainGridMesh(inData,Settings)
        qxm = Settings['qxm']
        if proctype.lower() == 'oce':
                qym = Settings['qym']*0 
        elif proctype.lower() == 'oct':
                qym = Settings['qym']
        qr = np.fft.ifftshift(np.sqrt(qxm**2+qym**2)) #optimize out fftshift
        qz = np.sqrt((2*Settings['refractive_index']*kc)**2-qr**2)
        inData = np.fft.fft(inData,axis=0) 
        for i in range(np.shape(inData)[0]):
                inData[i,:,:] = inData[i,:,:] * np.exp(1j*qz*zxy)         
        return np.fft.ifft(inData,axis=0), Settings

