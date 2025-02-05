This repository trains MaskRCNN for instance segmentation of dwelling objects from different FDP settlement areas. This is an improved version of [Mapping of Dwellings in IDP/Refugee Settlements from Very High-Resolution Satellite Imagery Using a Mask Region-Based Convolutional Neural Network]() where the training scheme accounts for combining datasets from different geography and time. This enables not only using pre-trained weight from non-earth observation imagery, it provides ways to properly combine and train on different datasets. It also combines further sampling tools for custom sample selection.

# Instllation
to create the environment follow the instructions on "Instruction_Manual.pdf"

## Usage 
First, try to change parameters inthe  configs.py file and then 

for training


```python train.py```

for testing

```python geopredict.py```

for finetuning

```python finetune.py ```

# Usage for sample preparation for finetuning

In the folder "ArcGISSamplingtool" there is a "data_preparation.pyt" file containing custom sampling tools usable in the ArcGIS Pro environment. Simply link the folder in the ArcGIS Pro catalogue pane and see the functionality and run Python toolboxes for custom fishnet creation as per the input image, intended individual sample chip size, then select representative polygons where selected digitization could happen. 
