# Stanford Data Extraction
extracts the stanford data set corresponding to the flex 2 channels

## Table of Contents
1. [Run the extraction](#run-the-extraction)
2. [Folder Structure](#files-and-directories)
3. [About the Dataset](#about-the-data)


## Run the extraction

Download and activate the corresponding conda environment

    conda env create -f env.yml
    conda activate brain2flex2

Change the variables:

    DATA_DIR
    FLEX2STANFORD
    EXTRACT

**FLEX2STANFORD Mapping:**
* Represents the mapping of emotiv channel to flex2 channel

* Mapping will depend on our final decision on which area we want to extract
  * We need the name of the channel we give it (at least I think we name the channels)
  * We then need to find the corresponding channel in the stanford data set

* *Currently is ChatGPT generated LOL cause I'm not sure if we've decided on which areas we will specifically map*
   
**EXTRACT:**
* Represents the Channels we want to extract
* Will need to change depending on our decisions

Run ``python load_stanford.py`` to start the extraction

## Files and Directories

### load_stanford.py

Loads the stanford data set and saves it in a directory called ./stanford_data

Pipeline follows 3 steps
1. Extract the channel data
2. normalized the data
3. Save the data into `.edf` file in corresponding folder

### visualize_data.ipynb
Includes the visualization scripts for the dataset

### stanford_data

Split into directories according to the original data

Then in each directory the data is then split to each time frame as:

    trial_{timeframe_value}.edf # for the data
    trial_{timeframe_value}_label.txt # for the actual sentence


## About the data
Read more about the dataset using this [link](https://docs.google.com/document/d/1zk9Nt3RtSVQDCcyiZOGQXQ-OWUYKfMY2pL3aViWhZZk/edit?usp=sharing) in the Competition Data tab

