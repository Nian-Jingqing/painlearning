# Code for the analyses of "Conditioned pain expect..."

**By Michel-Pierre Coll 2020**

## Reproducing the analyses in XXXX

Most analyses performed can be reproduced using the Docker container mpcoll2/eeg_XX. All Python scirpts can be executed using the following command with the appropriate path. The data path should point to a folder cotaining the bids formatted dataset in a "source" folder (e.g. for */data/source* path would be */data*).


```
docker run -it -v $PATHTODATA$ mpcoll2/eeg_2020:latest python $script.py$
```

SCR analyses as well as computation models specification and fitting was done in Matlab using the HFT toolbox and the VBA toolbox.

## 0- Import raw data

For the sake of simplicity and reducing online storage space requirements, we openly share the BIDS structured "almost raw" dataset. The BIDS formatted dataset is "almost raw" because some downsampling was applied to the physiological signals during importation so that all signals would be sampled at 1000 Hz for all participants. The script *raw_to_bids.py* contains all operations performed to restructure the raw files outputted by the various recording softwares into the BIDS format.

To run *raw_to_bids.py* : All raw data should be located in the /raw folder. Loads the raw .acq files, realigns triggers with eprime, correct some trigger issues in some participants and exports NFR and SCR data to a .tsv file for further analyses. This script also restructures the dataset in a format that is roughly equivalent to the EEG BIDS standard.


## 1- SCR

### 1.1. Import data for PsPM

The *scr_prepare_pspm.py* script imports the scr data and information (onsets, durations) in a matlab structure for each participant to facilitate the preprocessing in PSPM. The script outputs figures to visually inspect the SCR, a .mat structure for PSPM and the data in a .txt file for PSPM.

### 1.2 Process the data with PSPM

The *scr_glm_pspm.m* script imports the scr data in PsPM, performs the data cleaning and produces a GLM for each trial. It outputs figures illustrating the design matrix and other information on the GLM as well as the amplitude estimates for each trial in "_responses.mat" file. The */data/derivatives* path needs to be changed at the top of the script. SCR responses were further to fit computational models as described below.

## 2- Computational models

The computational models are described in their respective code in /code/compomodels. The MATLAB script comp_fitmodels.m fits the models specified at the start of the script and performs model comparison. For each model, it outputs the trial-wise values in a mat/csv files, some figures and a .mat file containing the results of the model comparison.

## 3- EMG


## 3- Computational modelling


## 4- EEG
