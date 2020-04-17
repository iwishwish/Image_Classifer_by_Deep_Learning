# Image Classifer by Deep Learning


### Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Motivation](#motivation)
4. [File Descriptions](#files)


## Installation <a name="installation"></a>    


## Usage<a name="usage"></a> 
**Train model**    
- Basic usage:     
`python train.py data_directory`     


- Options:        
Set directory to save checkpoints:     
`python train.py data_dir --save_dir save_directory`        
Choose architecture:     
`python train.py data_dir --arch "vgg13"`    
Set hyperparameters:     
`python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`    
Use GPU for training:     
`python train.py data_dir --gpu`    


**Predict flower name from an image**    
- Basic usage:     
`python predict.py /path/to/image checkpoint`    


- Options:    
Return top KK most likely classes:     
`python predict.py input checkpoint --top_k 3`
Use a mapping of categories to real names:     
`python predict.py input checkpoint --category_names cat_to_name.json`
Use GPU for inference:     
`python predict.py input checkpoint --gpu`



## Project Motivation<a name="motivation"></a>    



## File Descriptions <a name="files"></a>


