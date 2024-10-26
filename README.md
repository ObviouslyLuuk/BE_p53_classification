
# Master Thesis Project: AI for Barret's Esophagus Dysplasia Detection

This project was done in collaboration with the Pathology department at Amsterdam UMC, location AMC, under supervision of Prof. Erik J. Bekkers, Michel Botros and Dr. Sybren Meijer. The goal was to use different AI model architectures for the task of detecting Barret's Esophagus (a precursor for Esophageal Cancer) lesions in microscopic scans of biopsies from the esophagus, stained with p53 immunohistochemistry.

The main model foundations used were CLAM and ResNet, with RetCCL used for feature extraction. I added double-binary classification, and added synthetic data to the dataset to improve the model's performance. The models were trained/tested on part of the LANS dataset and on the BOLERO dataset.

## Main files
- `vis_data.ipynb`: Visualize some aspects of the data like distribution
- `preprocess_latents.py`: Preprocess the png images into latents using RetCCL
- `resnet.py`: Train the ResNet model
- `pl_clam.py`: Train the CLAM model
- `vis_results.ipynb`: Load the trained models and test them on the test set and save them, then visualize the results in confusion matrices, ROC curves and such
- `vis_interpretability.ipynb`: Load the trained models and visualize the attention maps of the CLAM models and the other visualization methods for the ResNet models

## Model files
- `resnet.py`: Contains the ResNet model architecture implemented in PyTorch Lightning
- `pl_clam.py`: Contains the CLAM model architecture implemented in PyTorch Lightning
- `clam_model/clam_utils.py`: Contains a few utility functions for the CLAM model, including its attention nets
- `RetCCL/`: Contains the RetCCL architecture and the code to download its pretrained weights
- `load_data.py`: Contains various dataset classes, utils and transforms for loading the data
- `eval.py`: Contains the evaluation functions to use during PyTorch Lightning training

## Running the code
Before running, the code expects a data root directory "../../data" to be present, but this can be changed at the top of `load_data.py`. In that root should be directories for each dataset (for example "LANS" and "BOLERO"). The main dataset (LANS in my case) should have the following structure:
- `biopsies/` containing the biopsy pngs, with names like "11_3.png" for the 11th slide and 3rd biopsy on that slide
- `train.csv` and `test.csv` containing the image names and labels like:
```
id,label
11_3,1
11_4,0
...
```
where 0 is Wild Type, 1 is Overexpression, 2 is Null Mutation and 3 is Double Clones.

For BOLERO, the biopsies/ directory is also expected, but aside from that a labels file "P53_BOLERO_T.csv" with all pathologist ratings is expected in the form of:
```
Case ID,{pathologist id}, ... ,GS
{case id},{pathologist rating}, ... ,{gold standard label}
...
```
