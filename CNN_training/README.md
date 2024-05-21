# CNN Training

Here, scripts can be found that train AlexNet, ConvNeXt Tiny, DenseNet121, InceptionV3, ResNet50, or ShuffleNet in
performing binary classifications on input images


## Table of contents
* [Development](#development)
* [CNN models](#cnn-models)
* [CNN input](#CNN-input)
* [CNN output](#CNN-output)


##  Development
This folder contains debugging versions of the CNN scripts. This are not fully functional and will most likely not 
produce proper results


## CNN models
Each of the six python scripts available in this folder can be used to train the CNN the script is named after.
All scripts require the same type of [input](#CNN-input) and will log their training measurements to wandb. A csv 
file is also produced, which will be explained in detail down [below](#CNN-output).

The CNNs can all be run in the same manner, by activating the pytorch2 environment and running the desired script in 
said environment. An example of running one of the CNNs is given below:
```
conda activate pytorch2
```

```
python AlexNet.py
```


## CNN input

### Input images
Input JPEG images are taken from the training and validation folder, which can be specified in the hyperparameter csv file.
At minimum, the images used have to contain their corresponding study id within the filename in the following format: ```prefix_001_suffix.jpg```
For data organization purposes it is recommended to also have folders with a similar name structure, though this is not
required for the CNN in order for it to be trained.
All models require the images to be 512px in size.

An example of the correct folder structure is shown below:

Structured folder:
```
	.
	├── ...
	├── Training               
	│	├── SID_001_characteristics
	│	│	├── SID_001_characteristics-A.jpg
	│	│	├── SID_001_characteristics-B.jpg			
	│	│	└── ...
	│	├── SID_002_characteristics
	│	│	├── SID_002_characteristics-A.jpg
	│	│	├── SID_002_characteristics-B.jpg			
	│	│	└── ...
	├── Validation               
	│	├── SID_003_characteristics
	│	│	├── SID_003_characteristics-A.jpg
	│	│	├── SID_003_characteristics-B.jpg			
	│	│	└── ...
	│	│	└── ...
	│	└── ...
	└── ...
```

Unstructured folder:
```
	.
	├── ...
	├── Training               
	│	├── SID_001_characteristics-A.jpg
	│	├── SID_001_characteristics-B.jpg			
	│	├── SID_002_characteristics-A.jpg
	│	├── SID_002_characteristics-B.jpg			
	│	└── ...
	├── Validation               
	│	├── SID_003_characteristics-A.jpg
	│	├── SID_003_characteristics-B.jpg			
	│	└── ...
	│	└── ...
	└── ...
```


### Hyperparameter csv
The path to the hyperparameter csv file should be specified by the user, before running the CNNs. Paths can be specified
at the top of each of the CNN scripts. Below an example version of the csv file is shown, along with a description of
each variable. The hyperparameter.csv file in this folder can be modified and used for training the CNNs


| Variable            | Value                                          | Description                                                                                                                                                                                              |
|---------------------|------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| root_dir            | D:\path\to\root_dir                            | Path to local directory, containing a training and validation folder for training CNNs                                                                                                                   |
| xlsx_path           | D:\path\to\binary.xlsx                         | Path to local [xlsx file](#binary-classification-xlsx), containing two columns. The first column must have all unique study IDs, the second columns the binary classes for majority (1) and minority (0) |
| train_dirname       | Training                                       | The name of the folder in root_dir, containing the images for training the model                                                                                                                         |
| val_dirname         | Validation                                     | The name of the folder in root_dir, containing the images for validating the model                                                                                                                       |
| wandb_name          | wandb_project_name                             | Name of the project that will be stored on wandb                                                                                                                                                         |
| wandb_save          | D:\path\to\local_wandb_save_folder             | Path to local directory where models and CNN data will be stored that is logged on wandb                                                                                                                 |
| model_param_csv     | D:\path\to\CNN_model_hyperparameters.csv       | Name of the csv file where the model will be stored along with the hyperparameters to create said model                                                                                                  |
| dataload_workers    | 3                                              | Number of multiprocessing workers to be used for the dataloaders (3 workers was determined to be optimal for an Intel(R) I9-13900K                                                                       |
| accumulation_steps  | 5                                              | Accumulation step size to be taken during training and validation                                                                                                                                        |
| num_epochs          | 50                                             | Number of epochs a model should be trained for                                                                                                                                                           |
| num_trials          | 100                                            | Number of trials the code should run for, where one trial equals one model                                                                                                                               |
| es_counter          | 0                                              | Start value of the early stop counter                                                                                                                                                                    |
| es_limit            | 15                                             | Value at which early stop is triggered and a trial will be terminated                                                                                                                                    |
| tl_learn_rate       | 1e-4; 5e-4;1e-3;2e-3;3e-3;4e-3;6e-3;8e-3;1e-2  | Learning rate values to be chosen at random by the CNN during training                                                                                                                                   |
| tl_batch_norm       | True;False                                     | Allow for batch normalization for the entire trial. Statements chosen at random at the start of a trial                                                                                                  |
| tl_batch_size_min   | 64                                             | Minimum number of images to be processed simultaneously by a model during training and validation                                                                                                        | 
| tl_batch_size_max   | 512                                            | Maximum number of images to be processed simultaneously                                                                                                                                                  |                                                                   
| tl_batch_size_step  | 64                                             | Step size from the minimum batch size to the maximum batch size. Here resulting in options 64, 128, 192, 256, 320, 384, 448, and 512                                                                     |                                                                  
| tl_weight_decay_min | 1e-5                                           | Minimum weight decay value                                                                                                                                                                               |                                                                 
| tl_weight_decay_max | 1e-1                                           | Maximum weight decay value                                                                                                                                                                               |                                                                
| tl_gamma_min        | 0.1                                            | Minimum gamma                                                                                                                                                                                            |                                                               
| tl_gamma_max        | 1.0                                            | Maximum gamma                                                                                                                                                                                            |                                                              
| tl_gamma_step       | 0.1                                            | Step size used when determining gamma. Here resulting in possible values of 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0                                                                             |                                                             


### Binary classification xlsx
This Excel file contains two columns which detail which study id has which binary label, to be predicted by the model.
An example is provided below:

| study_id | binary_label |
|----------|--------------|
| 001      | 1            |
| 078      | 0            |
| 376      | 1            |


## CNN output
The CNNs will upload most of their results to their wandb project as specified in the 
[parameter file](#hyperparameter-csv). The results that are saved locally are the models it has produced, 
along with a '.csv' file that contains the hyperparameter details of the produced models. On wandb, various graphs are 
plotted that detail the progression of the models, alongside ROC-AUC plots of training and validation results in addition
to tables detailing the hyperparameters of each model.
