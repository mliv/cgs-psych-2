# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 10:01:44 2021
@author: Matei

"""

#

"""

 npz files are the output of deepsleepnet/prepare_physionet.py by
 Akara Supratak et al
 https://github.com/akaraspt/deepsleepnet/blob/master/prepare_physionet.py
 
 Annotation scheme:
 ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4, -- target
    "Sleep stage ?": 5, -- remove
    "Movement time": 5  -- remove
    }
 
How to load and browse npz files:
    
import numpy
pt0 = numpy.load('SC4001E0.npz')
print(pt0.files)
pt0['x']
pt0['y']

That can also be done in Julia

import matplotlib.pyplot as plt
plt.plot(pt0['x'][12])

Characteristics of the timeseries (the arrays in x):
    -- samples (length of array) = 3000
    -- length = 30 seconds
    -- sampling rate = fs = 100
    -- amplitude varies a lot, most in wake (0), where it can be +/- 100+
    -- data type is float32
    
We are using SciPy to write wav files, with one channel (Fpz-Cz) --> mono

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile

# Trying to do this cuts off everything above and below +/- 1.0 amplitude
# scipy.io.wavfile.write("pt0-12th.wav", 100, pt0['x'][12])

"""
The following writes files with float32 values normalized from -1.0 to +1.0
This may be what we want if we don't care about absolute amplitude

audionorm = pt0['x'][12]
audionorm /= np.max(np.abs(audio)) # normalize array to -1.0 to +1.0 (numpy)
scipy.io.wavfile.write("pt0-12th.wav", 100, audionorm)

But absolute amplitude of the brain waves is probably an important feature
That we may want to have available for training
So instead we can try this.
The "32-bit PCM" WAV standard allows int32s with values +/- 2147483648
Multiplying values by 1000000 (e6) then converting to int32 only cuts off
at most 1 significant figure from the source data while giving us 1 order of 
magnitude of headroom before hitting the overflow (the data will be up to 9
figures since amplitude of the data generally tops out at 3 figures)

audio = pt0['x'][12]
audio *= 1000000
audioint = audio.astype(np.int32)
scipy.io.wavfile.write("pt0-12th.wav", 100, audioint)

But SciPy refuses to work with any integers despite explicitly offering that 
in the documentation (instead of determining the format from the data type it 
                      converts them to float32 and mangles the output).
"""

# So we will be using normalized float32s (-1.0 to +1.0) for now

# Normalizing it per patient seems like a better idea than normalizing by
# the largest amplitude in the data set, since it ameliorates rather than
# exacerbates differences between the different patients' brains and the
# conditions of the recordings

# The catch is that any extreme high value will ruin all of the samples for
# that patient, so we may have to implement range tests for each patient and 
# discard any epochs with extreme values that could be caused by glitches
# or electrodes falling off the patient's head, etc
# (also normalizing it by entire data set would cause one of these events to 
# ruin not just that patient, but the entire set)

# The above preprocessing module already discards epochs coded as 5 which
# means patient getting up to use the bathroom, having a seizure, etc and 
# any other activity the technician couldn't score as a sleep stage

stagenames = [
    "W",
    "1",
    "2",
    "34",
    "R"
    ]

# to make this work the data should be in a folder called "in"
# and the output will go in a folder called "out" with subfolders called 
# "1", "2", "34", "R", and "W"

for findex, filename in enumerate(os.listdir('in')):
    
    ptdata = np.load("in/"+filename)
    name = filename[:8] # trim the file extension off

    epochs = ptdata['x']
    stages = ptdata['y']
    for index, epoch in enumerate(epochs): 
        epoch /= epochs.max() # normalize
        # 100 Hz (the input rate)
        #scipy.io.wavfile.write(f'out/{stagenames[stages[index]]}/{name}-{index}.wav', 100, epoch)
        # 16 kHz (this is what CMPUtils and its modules expect)
        scipy.io.wavfile.write(f'wav/{stagenames[stages[index]]}/{name}-{index}.wav', 16000, epoch)

# With the first 71 patients (the preprocessing module crashes after that, it's
# certainly fixable but not necessary yet) the count is:
# 71 patients
# 81,490 samples
# 13,253 REM samples
# 68,237 Non-REM samples
# 937 MB, 954 MB on disk




