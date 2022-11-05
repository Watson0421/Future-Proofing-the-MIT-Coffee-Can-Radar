# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 11:22:52 2021

@author: Will Watson
"""

#general code of the serial module
import serial

from scipy.io.wavfile import write
import numpy as np
import time
import math

# set up 
samplerate = 4400 # This is a guess that will be corrected if too low to meet time
seconds = 600

filename = "May_20_SAR_truck_to_street.wav"

print("Initial samplerate guess " ,str(samplerate))

ser = serial.Serial(port='COM3',baudrate = 115200,timeout=None)
ser.flushInput()

# ser.open()
if ser.is_open==True:
	print("\nSerial port now open. Configuration:\n")
	print(ser, "\n") #print serial parameters

true_start = time.time()
# Calculates the sample rate by a sampling for x seconds
x = 5
N = samplerate*x;
data_buffer = np.zeros((N,2), np.int16)
start_time = time.time()
i = 0
end_time = start_time+x
while (time.time() < end_time):
    try:
        line = ser.readline()
        if line:
            string = line.decode("utf-8").strip()
            read_data = string.split(',')
            if (len(read_data)==2):
                for datum in read_data:
                    datum=int(datum.strip())
                    
                data_buffer[i][0]=read_data[0]
                data_buffer[i][1]=read_data[1]
        
                
    except KeyboardInterrupt:
        print("Press Ctrl-C to terminate while statement")
        break
    except:
        print("Error Skip")
    i = i +1
    if (i>N):
        samplerate = samplerate*2
        N = samplerate*x;
        data_buffer = np.zeros((N,2), np.int16)
        i = 0
        print("Hit Data limit, sample rate guess too low")
        print("retrying with samplerate guess " ,str(samplerate))
        start_time = time.time()
        end_time = start_time+x

if (N>i):
    total_time = (time.time()- true_start)
    samplerate = math.ceil(i/total_time*1.1) # factor of 1.1 to ensure account for run to run FS changes
    print("guess time taken ", str(total_time))
    for i in range(100):
        print("go! go! go!")
    print("taking samples with samplerate guess " ,str(samplerate))

N = samplerate*seconds;
data_buffer = np.zeros((N,2), np.int16)
i = 0

# Calculates the sample rate by a sampling for 5 seconds
start_time = time.time()
end_time = start_time+seconds
while (time.time() < end_time):
    try:
        line = ser.readline()
        if line:
            string = line.decode("utf-8").strip()
            read_data = string.split(',')
            if (len(read_data)==2):
                for datum in read_data:
                    datum=int(datum.strip())
                    
                data_buffer[i][0]=read_data[0]
                data_buffer[i][1]=read_data[1]
        
                
    except KeyboardInterrupt:
        print("Press Ctrl-C to terminate while statement")
        break
    except:
        print("Error Skip")
    i = i +1
    if (i>N):
        samplerate = samplerate*2
        N = samplerate*seconds;
        data_buffer = np.zeros((N,2), np.int16)
        i = 0
        print("Hit Data limit, sample rate guess too low")
        print("retrying with samplerate guess " ,str(samplerate))
        start_time = time.time()
        end_time = start_time+seconds

total_time = time.time()- start_time
print ("Time Taken to sample ", total_time)

ser.close()

print ("Restuctruring data ")

data = np.zeros((i,2), np.int16)
for k in range(i):
    data[k][0]=data_buffer[k][0]
    data[k][1]=data_buffer[k][1]

i=i-1
FS = math.ceil(i/total_time)

print ("True sample rate ", str(FS))

write(filename, FS, data.astype(np.int16))
print("Done")    


for i in range(100):
    print("end! end! end!")


