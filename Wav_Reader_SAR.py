# -*- coding: utf-8 -*-
"""
Created on Sat May 14 11:22:52 2022

@author: Will Watson

Based on MIT IAP Radar Course 2011
Resource: Build a Small Radar System Capable of Sensing Range, Doppler, 
and Synthetic Aperture Radar Imaging 
By Gregory L. Charvat

The RMA SAR algorithm was written by Gregory L. Charvat as part of his dissertation:
G. L. Charvat, ``A Low-Power Radar Imaging System," Ph.D. dissertation,
Dept. of Electrical and Computer Engineering, Michigan State University, East Lansing, MI, 2007.

Please cite appropriately.  

This algorithm was based on:
Range Migration Algorithm from ch 10 of Spotlight Synthetic Aperture Radar
Signal Processing Algorithms, Carrara, Goodman, and Majewski

"""


from scipy.io import wavfile
import numpy as np
import math



import matplotlib.pylab as plt
from statistics import mean

def dbv(volt_in):
    temp = np.absolute(volt_in)
    volt_out = np.multiply(20,np.log10(temp))
    
    return volt_out

plt.rcParams.update({'font.size': 12})
plt.rcParams["font.family"] = "Times New Roman"

FS, Y = wavfile.read('May_20_SAR_truck_to_street.wav')


print(FS)

c = 3E8
Tp = 20E-3 #Pulse Time
Trp = 0.25 # (s) min range profile time duratio
N = int(Tp*FS) #Samples per Pulse

fstart = 2400E6 #(Hz) LFM start frequency for ISM band
fstop = 2483E6 #(Hz) LFM stop frequency for ISM band
fc = (fstart+fstop)/2

BW = fstop-fstart #Transmit bandwidth


# Seperate the channels
s = Y[:,0]
trig = Y[:,1]




# parse data for when it is on

rpstart = abs(trig)>mean(abs(trig))



count = 0
Nrp = int(Trp*FS) # min  samples between range profiles

index_list = []
for ii in range (Nrp,len(rpstart)-Nrp+1):
    x = rpstart[ii]
    y = not (any(rpstart[ii-Nrp:ii]))
    if (x) and (y):
        print(count)
        count = count + 1;
        index_list.append(ii)

# index_removal=12
# index_list=index_list[:-index_removal]
RP=np.empty((count,Nrp))
RPtrig=np.empty((count,Nrp))
count = 0
for index in index_list:
    RP[count]=s[index:index+Nrp]
    RPtrig[count]=trig[index:index+Nrp]
    count = count + 1;

# RP=RP/32768    
        
print ("seperated")
#parse data by pulse
count = 0;
thresh = 0.08;

sif=np.empty((len(RP),int(N/2)),dtype = 'complex_')
for jj in range(0,len(RP)):
    SIF = np.zeros(N)
    start = (RPtrig[jj]> thresh)
    count = 0
    print(jj)
    for ii in range (11,(len(start)-2*N)):
        I =  np.argmax(RPtrig[jj][ii:ii+2*N]) # index of max (I)
        if (not ( not (start[ii-10:ii-2]).all) and  (I == 0)):
            count = count + 1;
            SIF = RP[jj][ii:ii+N]+ SIF
            # print(SIF)

    #hilbert transform
    q = np.fft.ifft(SIF/count)
    # print(len(q)/2)
    sif[jj] =  np.fft.fft(q[int(math.ceil(len(q)/2)):len(q)+1])

print("SAR Data ready")



for i in range(len(sif)):
    sif[i] = sif[i]-np.mean(sif,axis=0)

print("DC Removed")

fc = (2590E6 - 2260E6)/2 + 2260E6; #(Hz) center radar frequency
BW = (2590E6 - 2260E6); #(hz) bandwidth

# VERY IMPORTANT, change Rs to distance to cal target
# Rs = (12+9/12)*.3048; (m) y coordinate to scene center (down range), make this value equal to distance to cal target
Rs = 0
Xa = 0 # (m) beginning of new aperture length
delta_x = 2*(1/12)*0.3048 #(m) 2 inch antenna spacing
L = delta_x*(len(sif)) #(m) aperture length
Xa = np.linspace(-L/2, L/2, int(L/delta_x)) #(m) cross range position of radar on aperture L
Za = 0
Ya = Rs #THIS IS VERY IMPORTANT, SEE GEOMETRY FIGURE 10.6
t = np.linspace(0, Tp, len(sif[0])) #(s) fast time, CHECK SAMPLE RATE
Kr = np.linspace(((4*math.pi/c)*(fc - BW/2)), ((4*math.pi/c)*(fc + BW/2)), (len(t)))


"""
Begin update of SBAND_RMA_IFP.m

for 
figcount = 1;
close_as_you_go = 0;
do_all_plots = 0;

"""

# apply hanning window to data first
N = len(sif[0]);
H=np.empty(N)
for ii in range(N):
    H[ii] = 0.5 + 0.5*math.cos(2*math.pi*((ii+1)-N/2)/N)

sif_h=np.empty(np.shape(sif),dtype = 'complex_')
for ii in range(len(sif)):
    sif_h[ii] = sif[ii]*H

sif = sif_h

# along track FFT (in the slow time domain)
# first, symetrically cross range zero pad so that the radar can squint
zpad = 2048 #cross range symetrical zero pad
szeros = np.zeros((zpad, N),dtype = 'complex_')

for ii in range(N):
    index = round((zpad - len(sif))/2)
    szeros[index:(index + len(sif))+1][ii] = sif[:][ii] # symetrical zero pad

sif = szeros;

#Problem point
S =  np.fft.fftshift(np.fft.fft(sif,axis=0))

Kx = np.linspace((-math.pi/delta_x), (math.pi/delta_x), len(S))

# matched filter

phi_mf = np.empty(np.shape(S))
Krr = np.empty(np.shape(S))
Kxx = np.empty(np.shape(S))
# create the matched filter eq 10.8
for ii in range(len(S[0])): #step thru each time step row to find phi_if
    for jj in range(len(S)): #step through each cross range in the current time step row
        # phi_mf(jj,ii) = -Rs*Kr(ii) + Rs*sqrt((Kr(ii))^2 - (Kx(jj))^2);
        phi_mf[jj][ii] = Rs*np.sqrt((np.square(Kr[ii]) - (Kx[jj])**2))
        Krr[jj][ii] = Kr[ii] #generate 2d Kr for plotting purposes
        Kxx[jj][ii] = Kx[jj] #generate 2d Kx for plotting purposes

smf = np.exp((1.0j)*phi_mf)

# note, we are in the Kx and Kr domain, thus our convention is S_mf(Kx,Kr)

# apply matched filter to S
S_mf = np.multiply(S,smf)

#perform the Stolt interpolation

#FOR DATA ANALYSIS
kstart =73;
kstop = 108.5;

# kstart = 95;
# kstop = 102;


Ky_even = np.linspace(kstart, kstop, 1024) #create evenly spaced Ky for interp for real data



count = 0;
Ky = np.empty(np.shape(S))
S_st = np.empty((2048,1024),dtype = 'complex_')
for ii in range(zpad):
    count = count + 1;
    Ky[ii] = np.sqrt(np.square(Kr) - Kx[ii]**2)
    S_st[ii] = (np.interp(Ky_even,Ky[ii], S_mf[ii] ))
    

# apply hanning window to data 
N = len(Ky_even);
H=np.empty(N)
for ii in range(N):
    H[ii] = 0.5 + 0.5*math.cos(2*math.pi*((ii+1)-N/2)/N)

S_sth=np.empty(np.shape(S_st),dtype = 'complex_')
for ii in range(len(S_st)):
    S_sth[ii] = S_st[ii]*H

S_st = S_sth

#perform the inverse FFT's
#new notation:  v(x,y), where x is crossrange
#first in the range dimmension

v = np.fft.ifft2(S_st,((len(S_st[0])*4),(len(S_st)*4)))


bw = c*(kstop-kstart)/(4*math.pi)

max_range = (c*len(S_st[0]))/(2*bw)*1/.3048

S_image = v #edited to scale range to d^3/2
S_image = np.fliplr(np.rot90(S_image));
trunc_image = S_image
cr1 = -40 #(ft)
cr2 = 40 #(ft)
dr1 = 1 + Rs/.3048 #(ft)
dr2 = 80 + Rs/.3048 #(ft)
# data truncation
dr_index1 = round((dr1/max_range)*len(S_image));
dr_index2 = round((dr2/max_range)*len(S_image));
cr_index1 = round(( (cr1+zpad*delta_x/(2*.3048)) /(zpad*delta_x/.3048))*len(S_image[0]));
cr_index2 = round(( (cr2+zpad*delta_x/(2*.3048))/(zpad*delta_x/.3048))*len(S_image[0]));


# trunc_image = S_image[cr_index1:cr_index2,dr_index1:dr_index2];
trunc_image = S_image[dr_index1:dr_index2,cr_index1:cr_index2];

# trunc_image = S_image
downrange = np.linspace(-1*dr1,-1*dr2, len(trunc_image)) + Rs/.3048;
crossrange = np.linspace(cr1, cr2, len(trunc_image[0]));


for ii in range(len(trunc_image[0])):
    trunc_image[:,ii] = np.multiply(trunc_image[:,ii],np.power((abs(downrange*0.3048)),(3/2)))

trunc_image = dbv(trunc_image); #added to scale to d^3/2


plt.figure(1)

# trunc_image = trunc_image - np.max(trunc_image)
vmax =  np.max(trunc_image)
vmin = vmax-40

# plt.imshow(trunc_image)
plt.imshow(np.flip(trunc_image),extent=[crossrange[0],crossrange[-1],downrange[0],downrange[-1]],cmap='viridis',interpolation='nearest', vmin=vmin, vmax=vmax)
plt.xlabel("Cross-Range Ft")
plt.ylabel("Down-Range Ft")
plt.title("SAR Image")
clb = plt.colorbar()
clb.set_label("dB", labelpad=-40, y=1.05, rotation=0)