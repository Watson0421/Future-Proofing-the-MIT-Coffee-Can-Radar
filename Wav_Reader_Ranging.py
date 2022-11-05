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

plt.rcParams.update({'font.size': 20})
plt.rcParams["font.family"] = "Times New Roman"

file1 = 'May_19_6ft.wav'
file2 = 'Mar_26_6ft_test.wav'

FS, Y = wavfile.read(file1)

print(FS)

C = 3E8
Tp = 20E-3 #Pulse Time
N = int(Tp*FS) #Samples per Pulse

fstart = 2400E6 #(Hz) LFM start frequency for ISM band
fstop = 2483E6 #(Hz) LFM stop frequency for ISM band

BW = fstop-fstart #Transmit bandwidth

# f=np.linspace(int(fstart),int(fstop),int(N/2)) # instantaneous transmit frequency

# plt.figure(3)
# plt.plot(f)
# plt.xlabel("S")
# plt.ylabel("Hz")


# Range
rr = C/(2*BW)

max_range = rr*N/2

# Seperate the channels
s = Y[:,0]
trig = Y[:,1]

# Delete first elemenmnt as it is always 0
s = np.delete(s,0)
trig = np.delete(trig,0)

sig_ave = mean(s)
s = s - sig_ave
print(len(s))
print("DC Removed")

# parse data based on trigger
thresh = 0
start = (trig>thresh)
start = start.astype(float)

timer = np.array([])

controller = 11

for ii in range (100,len(start)-N,1):
    q = mean(start[ii-controller:ii-1])
    if start[ii] and (q == 0):
        timer=np.append(timer,ii*1/FS)

print("timer done")

count = 0
sif = np.zeros(shape=(len(timer),N-1))
for ii in range (100,len(start)-N,1):
    if start[ii] and (mean(start[ii-controller:ii-1]) == 0):
        temp = s[ii:ii+N-1]
        sif[count]=temp
        count = count +1

print("sif done")

ave = np.mean(sif,axis=0)
for i, sample in enumerate(sif):
    sif[i] = sample-ave
    
print("Average of sif col removed")

zpad = int(16*N)
# zpad = int(N)

transformed = np.fft.ifft(sif,zpad,1)
v = dbv(transformed)
S = v[:,:int(len(v[0])/2)]

# m = np.amax(S)
# for i, sample in enumerate(S):
#     S[i] = sample-m

print("Max deducted")

time_tick = (math.ceil(timer[-1])-math.floor(timer[0]))/10
x_tick = round(rr)

vmax=0
vmin=vmax-20


# calculate velocity
r = np.arange(0,max_range+rr, max_range/(S.shape[1])) #range bins


ran = np.round(r)

maxrange_graph = 20
ran_index = np.where(ran==maxrange_graph)
index = ran_index[0][0]
maxrange_graph = r[index]

print(index)
print(maxrange_graph)
im = S[:,:index]
# im=im-np.amax(im)

# plt.figure(1)
# plt.subplot(1,2,1)
# plt.imshow(im,extent=[0,maxrange_graph,timer[-1],timer[0]],cmap='viridis',interpolation='nearest', vmin=vmin, vmax=vmax)
# plt.xlabel("m")
# plt.ylabel("Seconds")

# ax = plt.gca()
# extent=[0,maxrange_graph,timer[-1],timer[0]]
# ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/1)

# plt.xticks(np.arange(0, maxrange_graph, rr))
# plt.yticks(np.arange(math.floor(timer[0]), math.ceil(timer[-1]), time_tick))

# clb = plt.colorbar()
# clb.set_label("dB", labelpad=-40, y=1.05, rotation=0)

# plt.title("Range")

# plt.subplot(1,2,2)
ave = np.mean(im,axis=0)


indx1 = np.argmax(ave)

mx1 = r[indx1]
print("Distance to maximum "+str(r[indx1]))
# plt.bar(ft,ave, width=0.8*4)
# plt.plot(r[:index],ave[:index],"k-")
# plt.xlabel("m")
# plt.ylabel("dB")

# ax = plt.gca()

# plt.xticks(np.arange(0, maxrange_graph, rr))
# plt.title("Averaged Range Signal")
# plt.show()


ave_id = ave - np.amax(ave)




FS, Y = wavfile.read(file2)

print(FS)



# Seperate the channels
s = Y[:,0]
trig = Y[:,1]

# Delete first elemenmnt as it is always 0
s = np.delete(s,0)
trig = np.delete(trig,0)

sig_ave = mean(s)
s = s - sig_ave
print(len(s))

print("DC Removed")

# parse data based on trigger
thresh = 0
start = (trig>thresh)
start = start.astype(float)

timer = np.array([])

controller = 11

for ii in range (100,len(start)-N,1):
    q = mean(start[ii-controller:ii-1])
    if start[ii] and (q == 0):
        timer=np.append(timer,ii*1/FS)

print("timer done")

count = 0
sif = np.zeros(shape=(len(timer),N-1))
for ii in range (100,len(start)-N,1):
    if start[ii] and (mean(start[ii-controller:ii-1]) == 0):
        temp = s[ii:ii+N-1]
        sif[count]=temp
        count = count +1

print("sif done")

ave = np.mean(sif,axis=0)
for i, sample in enumerate(sif):
    sif[i] = sample-ave
    
print("Average of sif col removed")

zpad = int(16*N)
# zpad = int(N)

transformed = np.fft.ifft(sif,zpad,1)
v = dbv(transformed)
S = v[:,:int(len(v[0])/2)]

# m = np.amax(S)
# for i, sample in enumerate(S):
#     S[i] = sample-m

print("Max deducted")

time_tick = (math.ceil(timer[-1])-math.floor(timer[0]))/10
x_tick = round(rr)


# calculate velocity
# r = np.arange(0,max_range+rr, max_range/(S.shape[1])) #range bins


# ran = np.round(r)

# maxrange_graph = 20
# ran_index = np.where(ran==maxrange_graph)
# index = ran_index[0][0]
# maxrange_graph = r[index]

# print(index)
# print(maxrange_graph)
im = S[:,:index]
# im=im-np.amax(im)


ave = np.mean(im,axis=0)
real_sacle = np.amax(ave)
ave = ave - real_sacle


indx2 = np.argmax(ave)
mx2=r[indx2]
print("Distance to maximum "+str(r[indx2]))
# plt.bar(ft,ave, width=0.8*4)

plt.figure(3)
plt.plot(r[:index],ave_id[:index]+real_sacle,"k--")
plt.plot(r[:index],ave[:index]+real_sacle,"k-")
plt.xlabel("m")
plt.ylabel("dB")

ax = plt.gca()


plt.xticks(np.arange(0, maxrange_graph, rr))
plt.ylim((-20,0))
plt.legend(["Ideal", "Real"])
plt.show()
plt.title("Averaged Signal Measured vs. Ideal")

print("m off from ideal ="+str(abs(mx2-mx1)))
print("Percent error= "+str(abs(mx2-mx1)/mx1*100))

# plt.figure(2)
# plt.subplot(1,2,1)
# plt.imshow(im,extent=[0,maxrange_graph,timer[-1],timer[0]],cmap='viridis',interpolation='nearest', vmin=vmin, vmax=vmax)
# plt.xlabel("m")
# plt.ylabel("Seconds")

# ax = plt.gca()
# extent=[0,maxrange_graph,timer[-1],timer[0]]
# ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/1)

# plt.xticks(np.arange(0, maxrange_graph, rr))
# plt.yticks(np.arange(math.floor(timer[0]), math.ceil(timer[-1]), time_tick))
# clb = plt.colorbar()
# clb.set_label("dB", labelpad=-40, y=1.05, rotation=0)
# plt.title("Range No Clutter Processing")

# S2 = S.copy()
# ave = np.mean(S2,axis=0)
# S2 = S2 - ave
# # for i, sample in enumerate(S2):
# #     S2[i] = sample-ave

# # m = np.amax(S2)
# # for i, sample in enumerate(S2):
# #     S2[i] = sample-m


# im2 = S2[:,:index]
# im2=im2-np.amax(im2)

# plt.subplot(1,2,2)
# plt.imshow(im2,extent=[0,maxrange_graph,timer[-1],timer[0]],cmap='viridis',interpolation='nearest', vmin=vmin, vmax=vmax)
# plt.xlabel("m")
# plt.ylabel("Seconds")

# ax = plt.gca()
# extent=[0,maxrange_graph,timer[-1],timer[0]]
# ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/1)

# plt.xticks(np.arange(0, maxrange_graph, rr))
# plt.yticks(np.arange(math.floor(timer[0]), math.ceil(timer[-1]), time_tick))
# clb = plt.colorbar()
# clb.set_label("dB", labelpad=-40, y=1.05, rotation=0)
# plt.title("Range Clutter Processing")
# plt.show()
