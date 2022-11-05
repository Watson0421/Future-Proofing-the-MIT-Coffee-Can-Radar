# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 11:22:52 2021

@author: Will Watson
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

def process_fille(file_name,factor,Tp,zPad):
    
    FS, Y = wavfile.read(file_name)

    print(FS)

    C = 299792458

    N = int(Tp*FS) #Samples per Pulse

    fc = 2440E6; #(Hz) Center frequency within ISM band (VCO Vtune to +3.2V)


    # Seperate the channels
    s = Y[:]
    # trig = Y[:,1]

    # Delete first elemenmnt as it is always 0
    s = np.delete(s,0)
    # trig = np.delete(trig,0)

    sig_ave = mean(s)
    s = s - sig_ave
    # for i, sample in enumerate(s):
    #     s[i] = sample-sig_ave

    print("DC Removed")



    row_num = math.floor(len(s)/(N/factor))-(factor-1)
    # row_num = len(s)-N
    sif = np.zeros((row_num,N))

    timer = np.array([])
    # create doppler vs. time plot data set here
    for ii in range(row_num):
        # sif[ii] = s[ii*(N/factor):(ii+1)*N]
        x = int(ii*N/factor)
        # print(x)
        if len(s[x:x+N])==N:
            sif[ii] = s[x:x+N]
            # sif[ii] = s[ii:ii+N]
            timer=np.append(timer,ii*Tp/factor)
        else:
            print(len(s[x:x+N]))
            


    print("Sorted Data")
        
    zpad = int(zPad*N)
    # zpad = int(N)
    
    transformed = np.fft.ifft(sif,zpad,1)
    v = dbv(transformed)
    S = v[:,:int(len(v[0])/2)]

    print("Transformed")
    step = (FS/2)/(S.shape[1])
    # calculate velocity
    delta_f = np.arange(0,FS/2+step, step)#Hz
    wave_len=C/fc;
    velocity = delta_f*wave_len/2;

    vel = np.floor(velocity)

    vel_res = velocity[1]
    print("V_res= "+str(vel_res))

    

    max_vel = 10
    vel_index = np.where(vel==max_vel)
    index = vel_index[0][0]
    print(vel_index[0][0])
    im = S[:,:index]
    im=im-np.amax(im)
    max_vel = velocity[index]
    print(max_vel)

    

    
    return im,timer,max_vel,Tp,s

def show_plt(im,max_vel,timer,vmin,vmax,time_tick):
    plt.imshow(im,extent=[0,max_vel,timer[-1],timer[0]],cmap='viridis',interpolation='nearest', vmin=vmin, vmax=vmax)
    # plt.imshow(im,extent=[0,max_vel,5,Tp],cmap='viridis',interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.xlabel("m/s")
    plt.ylabel("Seconds")

    ax = plt.gca()

    extent=[0,max_vel,timer[-1],timer[0]]
    # extent=[0,max_vel,5,Tp]
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/1)

    plt.xticks(np.arange(0, max_vel, 1.5))
    plt.yticks(np.arange(timer[0], timer[-1], time_tick))
    # plt.yticks(np.arange(math.floor(Tp), math.ceil(5)-Tp, Tp))
    clb = plt.colorbar()
    clb.set_label("dB", labelpad=-40, y=1.05, rotation=0)
    
    return


plt.rcParams.update({'font.size': 12})
plt.rcParams["font.family"] = "Times New Roman"

# plt.figure(1)

# plt.subplot(1,2,1)

# file_name = 'May_12_Doppler_Test_60s.wav'

# factor = 1
# Tp = 0.1 #Pulse Time
# zPad = 64

# im,timer,max_vel,Tp = process_fille(file_name,factor,Tp,zPad)

# time_tick = 0.5

# vmax = 0
# vmin = vmax-30

# time_one = 1.8 # seconds
# time_two = 6.8 # seconds
# time_one_index = int(time_one/Tp)
# time_two_index = int(time_two/Tp)

# timer = timer [time_one_index:time_two_index]
# im = im [time_one_index:time_two_index]
# show_plt(im,max_vel,timer,vmin,vmax,time_tick)

# plt.title("5 Second Excerpt, Real")

# plt.subplot(1,2,2)

# file_name = 'May_19_Pendulum_Ideal_Filt.wav'

# im,timer,max_vel,Tp = process_fille(file_name,factor,Tp,zPad)


# timer = timer [time_one_index:time_two_index]
# im = im [time_one_index:time_two_index]
# show_plt(im,max_vel,timer,vmin,vmax,time_tick)

# plt.title("5 Second Excerpt, Ideal")

plt.figure(1)

plt.subplot(2,2,1)

file_name = 'Apr_18_Pendulum_Ideal.wav'

factor = 1
Tp = 0.1 #Pulse Time
zPad = 64

im,timer,max_vel,Tp,s1 = process_fille(file_name,factor,Tp,zPad)

time_tick = 0.5

vmax = 0
vmin = vmax-30

show_plt(im,max_vel,timer,vmin,vmax,time_tick)

plt.title("(a) Doppler Ideal")


plt.subplot(2,2,2)

file_name = 'Apr_18_Pendulum_Ideal_Filt.wav'

im,timer,max_vel,Tp,s1 = process_fille(file_name,factor,Tp,zPad)

show_plt(im,max_vel,timer,vmin,vmax,time_tick)

im_temp = im.transpose()

plt.title("(b) Doppler Ideal HPF")

plt.subplot(2,2,3)


file_name = 'Apr_09_Doppler_Test.wav'

im,timer,max_vel,Tp,s2 = process_fille(file_name,factor,Tp,zPad)

show_plt(im,max_vel,timer,vmin,vmax,time_tick)

im_temp2 = im.transpose()

plt.title("(c) Doppler Measured")

# plt.figure(2)

# N=5000
# s=s1[int(22E4):int(24E4)]
# row_num = math.floor(len(s)/(N/factor))-(factor-1)
# # row_num = len(s)-N
# sif = np.zeros((row_num,N))

# timer = np.array([])
# # create doppler vs. time plot data set here
# for ii in range(row_num):
#     # sif[ii] = s[ii*(N/factor):(ii+1)*N]
#     x = int(ii*N/factor)
#     # print(x)
#     if len(s[x:x+N])==N:
#         sif[ii] = s[x:x+N]
#         # sif[ii] = s[ii:ii+N]
#         timer=np.append(timer,ii*Tp/factor)
#     else:
#         print(len(s[x:x+N]))
        


# print("Sorted Data")
    
# zpad = int(zPad*N)

# transformed = np.fft.ifft(sif,zpad,1)
# v = dbv(transformed)
# S = v[:,:int(len(v[0])/2)]

# plt.plot(s,"k-")





# vel_index = []
# for i in im_temp:
#     vel_index.append(np.argmax(i))
    
# vel_index2 = []
# for i in im_temp2:
#     vel_index2.append(np.argmax(i))

# v1 = []
# v2 = []

# speeds=np.linspace(0, max_vel, len(i))
# for v in vel_index:
#       v1.append(speeds[v])
      
# for v in vel_index2:  
#       v2.append(speeds[v])


# plt.plot(v1)
# plt.plot(v2)


