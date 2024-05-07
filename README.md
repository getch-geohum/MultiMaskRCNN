This repository trains MaskRCNN for instance segmentation of dwelling objects from different FDP settlement areas

# Instllation
to create the environment follow instructions on "Instruction_Manual.pdf"

## Usage 
First try to change parameters in configs.py file and then 

for training


```python train.py```

for testing

```python geopredict.py```

for finetuning

```python finetune.py ```

# Usage for sample preparation for finetuning

In folder "ArcGISSamplingtool" there is a "data_preparation.pyt" file containing custom sampling tools usable in ArcGIS Pro environment. Simply link the folder in ArcGIS Pro catalogue pane and yse the functionality and run python toolboxes for custom fishnet creation as per input image, intended individual sample chip size, then select representative polygons where slected digitization could happen. 
