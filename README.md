# CGS3342 -- Psych2 -- REM Sleep Detection #



## Pre-processing ##

Credit for prepare_physionet.py goes to Akara Supratak et al and Ariel Gentile of [deepsleepnet](https://github.com/akaraspt/deepsleepnet), used under APACHE 2.0.
Credit for dhedfreader.py goes to Boris Reuderink, also used under APACHE 2.0.

Download the sleep cassette data from [Sleep-EDF Database Expanded from PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/) into ```/data```.
Run a python shell like Anaconda. 
To install [mne](https://mne.tools/stable/install/mne_python.html):

	cd <base directory of project where environment.yml is located>
	conda install --name base nb_conda_kernels "spyder>=4.2.1"
	conda env update --file environment.yml

Restart python shell.

	cd <project directory>
	activate mne
	python prepare_physionet.py --data_dir data --output_dir in --select_ch "EEG Fpz-Cz"
	python prepare_physionet.py --data_dir dataT --output_dir inT --select_ch "EEG Fpz-Cz"

This will take the polysomnogram data and select the Fpz-Cz (frontal electrode) channel as well as the manually-scored sleep stages (from the ```-hypnogram.edf``` file).
The result is epoched and converted to NumPy format (.npz) in two arrays, timeseries (x) and classifications (y).
Class 5 stages (M and ?, corresponding to movement time and unscorable activity) are stripped and left out.

Patient SC4362F0-PSG.edf/SC4362FC-Hypnogram.edf was removed due to errors in stripping unclassified (label=5, sleep stage ?) epochs.


## WAV conversion ##

Run the following:

	python wavconversion.py

This will convert the data from NPZ timeseries (30.0 seconds long, 100 Hz, 3000 samples) to normalized per-recording (-1.0 to +1.0 amplitude) 32-bit float WAV at 16,000 Hz (0.1875 seconds long).


## Feature extraction and recoding ##

This relies on [CMPUtils from COINSLab.](https://github.com/coinslab/CMPUtils.jl) 

Launch a Julia REPL eg in VSCode (in order to see plots etc).

	add https://github.com/coinslab/CMPUtils.jl

Run the recoding process in ```wavconversion.jl```.
This performs MFCC, obtaining a matrix of cep values using a Mel-frequency cepstral coefficient algorithm. 
The result is processed through SVD (singular value decomposition) and the number of features to keep can be chosen (up to 13).
Both 5-feature and 13-feature sets are included in this repo for comparison.
MFCC is intended for processing human speech and music, not EEG data, but the result is fairly good regardless.


## Final processing and handling ##

Run ```processing.jl``` to generate various csv files containing the feature data.
The main file that will be used for testing and training is ```audio_recoded_combined_binary.csv```.
This simply contains an array of feature values for each epoch as well as its sleep stage as REM (1) or non-REM (0).

## Training and testing

Run various parts of ```simpleclassifiers.jl```.
This work will focus on the multi-layer perceptron (MLP) and random forest (RFC) classifiers.


## Data Characteristics ##

152 patients

7.03 GB (7,555,228,642 bytes) in EDF+
2.17 GB (2,337,245,580 bytes) in NPZ 
2.20 GB (2,368,560,144 bytes) in WAV
8.62 MB (9,041,752 bytes)  in CSV (after SVD,  5 features)
53.4 MB (56,005,585 bytes) in CSV (after SVD, 13 features)


194,655 total epochs (WAV files)
21,469 stage 1
68,633 stage 2
12,991 stage 3/4
25,767 stage R
65,795 stage W




