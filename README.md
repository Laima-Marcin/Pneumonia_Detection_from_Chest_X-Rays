# Pneumonia_Detection_from_Chest_X-Rays
CNN model for predicting Pneumonia presence or absence from Chest x-ray images

In this project we analyze data from the NIH Chest X-ray Dataset and train a CNN to classify a given chest x-ray for the presence or absence of pneumonia. 
This project culminates in a model that can predict the presence of pneumonia with human radiologist-level accuracy that can be prepared for submission to the FDA for 510(k) clearance as software as a medical device. As part of the submission preparation, we formally describe our model, the data that it was trained on, and a validation plan that meets FDA criteria.

We are provided with the medical images with clinical labels for each image that were extracted from their accompanying radiology reports. 

The project has access to 112,000 chest x-rays with disease label acquired from 30,000 patients.

## Pneumonia and X-Rays in the Wild

When it comes to pneumonia, chest X-rays are the best available method for diagnosis. More than 1 million adults are hospitalized with pneumonia and around 50,000 die from the disease every year in the US alone. The high prevalence of pneumonia makes it a good candidate for the development of a deep learning application for two reasons: 

1) Data availability in a high enough quantity for training deep learning models for image classification 
2) Opportunity for clinical aid by providing higher accuracy image reads of a difficult-to-diagnose disease and/or reduce clinical burnout by performing automated reads of very common scans. 

The diagnosis of pneumonia from chest X-rays is difficult for several reasons: 
1. The appearance of pneumonia in a chest X-ray can be very vague depending on the stage of the infection
2. Pneumonia often overlaps with other diagnoses
3. Pneumonia can mimic benign abnormalities

For these reasons, common methods of diagnostic validation performed in the clinical setting are to obtain sputum cultures to test for the presence of bacteria or viral bodies that cause pneumonia, reading the patient's clinical history and taking their demographic profile into account, and comparing a current image to prior chest X-rays for the same patient if they are available. 

## About the Dataset

The dataset provided to us for this project was curated by the NIH specifically to address the problem of a lack of large x-ray datasets with ground truth labels to be used in the creation of disease detection algorithms. 

The data is mounted in the Udacity Jupyter GPU workspace provided to us, along with code to load the data. Alternatively, we can download the data from the [kaggle website](https://www.kaggle.com/nih-chest-xrays/data) and run it locally. 

There are 112,120 X-ray images with disease labels from 30,805 unique patients in this dataset.  The disease labels were created using Natural Language Processing (NLP) to mine the associated radiological reports. The labels include 14 common thoracic pathologies: 
- Atelectasis 
- Consolidation
- Infiltration
- Pneumothorax
- Edema
- Emphysema
- Fibrosis
- Effusion
- Pneumonia
- Pleural thickening
- Cardiomegaly
- Nodule
- Mass
- Hernia 

The biggest limitation of this dataset is that image labels were NLP-extracted so there could be some erroneous labels but the NLP labeling accuracy is estimated to be >90%.

The original radiology reports are not publicly available but we can find more details on the labeling process [here.](https://arxiv.org/abs/1705.02315) 


### Dataset Contents: 

1. 112,120 frontal-view chest X-ray PNG images in 1024*1024 resolution (under images folder)
2. Meta data for all images (Data_Entry_2017.csv): Image Index, Finding Labels, Follow-up #,
Patient ID, Patient Age, Patient Gender, View Position, Original Image Size and Original Image
Pixel Spacing.


## Project Steps

### 1. Exploratory Data Analysis

The first part of this project involves exploratory data analysis (EDA) to understand and describe the content and nature of the data.

Note that much of the work performed during our EDA will enable the completion of the final component of this project which is focused on documentation of our algorithm for the FDA. During our EDA we focus on: 

* The patient demographic data such as gender, age, patient position,etc. (as it is available)
* The x-ray views taken (i.e. view position)
* The number of cases including: 
    * number of pneumonia cases,
    * number of non-pneumonia cases
* The distribution of other diseases that are comorbid with pneumonia
* Number of disease per patient 
* Pixel-level assessments of the imaging data for healthy & disease states of interest (e.g. histograms of intensity values) and compare distributions across diseases.

### 2. Building and Training our Model

**Training and validating Datasets**

From our findings in the EDA component of this project, we curate the appropriate training and validation sets for classifying pneumonia. We take the following into consideration: 

* Distribution of diseases other than pneumonia that are present in both datasets
* Demographic information, image view positions, and number of images per patient in each set
* Distribution of pneumonia-positive and pneumonia-negative cases in each dataset

**Model Architecture**

In this project we fine-tune an existing CNN architecture to classify x-rays images for the presence of pneumonia. 
We use VGG16 architecture with weights trained on the ImageNet dataset. Fine-tuning is performed by freezing our chosen pre-built network and adding several new layers to the end to train, or by doing this in combination with selectively freezing and training some layers of the pre-trained network. 


**Image Pre-Processing and Augmentation** 

We do some preprocessing prior to feeding images into our network for training and validating. This helps with our model's architecture and for the purposes of augmenting our training dataset for increasing our model performance. When performing image augmentation, we think about augmentation parameters that reflect real-world differences that may be seen in chest X-rays. 

**Training** 

In training our model, there are many parameters that can be tweaked to improve performance including: 
* Image augmentation parameters
* Training batch size
* Training learning rate 
* Inclusion and parameters of specific layers in your model 

We provide descriptions of the methods by which given parameters were chosen in the final FDA documentation.

 **Performance Assessment**

As you train our model, we monitor its performance over subsequence training epochs. We choose the appropriate metrics upon which to monitor performance. Note that 'accuracy' may not be the most appropriate statistic in this case, depending on the balance or imbalance of our validation dataset, and also depending on the clinical context that we want to use this model in.

 __Note that detecting pneumonia is *hard* even for trained expert radiologists, so we should *not* expect to acheive sky-high performance.__ [This paper](https://arxiv.org/pdf/1711.05225.pdf) describes some human-reader-level F1 scores for detecting pneumonia, and we use it as a reference point for how well our model could perform.

### 3. Clinical Workflow Integration 

The imaging data provided to us for training our model was transformed from DICOM format into .png to help aid in the image pre-processing and model training steps of this project. In the real world, however, the pixel-level imaging data are contained inside of standard DICOM files. 

For this project, we create a DICOM wrapper that takes in a standard DICOM file and outputs data in the format accepted by our model. We include several checks in our wrapper for the following: 
* Proper image acquisition type (i.e. X-ray)
* Proper image acquisition orientation 
* Proper body part in acquisition


### 4. FDA  Submission

For this project, we complete the following steps that are derived from the FDA's official guidance on both the algorithm description and the algorithm performance assessment. __*Much of this portion of the project relies on what we did during our EDA, model building, and model training. We use figures and statistics from those earlier parts in completing the following documentation.*__

**1. General Information:**

* First, we provide an Intended Use statement for our model 
* Then, we provide some indications for use that includes: 
    * Target population
    * When our device could be utilized within a clinical workflow
* Device limitations, including diseases/conditions/abnormalities for which the device has been found ineffective and should not be used
* We explain how a false positive or false negative might impact a patient.

**2. Algorithm Design and Function**

In this section, we describe our _fully trained_ algorithm and the DICOM header checks that we have built around it. We include a flowchart that describes the following: 

* Any pre-algorithm checks we performed on our DICOM
* Any preprocessing steps we performed by our algorithm on the original images (e.g. normalization)
* The architecture of the classifier

For each stage of our algorithm, we briefly describe the design and function.

**3. Algorithm Training**

We describe the following parameters of our algorithm and how they were chosen: 

* Types of augmentation used during training
* Batch size
* Optimizer learning rate
* Layers of pre-existing architecture that were frozen
* Layers of pre-existing architecture that were fine-tuned
* Layers added to pre-existing architecture

Also we describe the behavior of the following throughout training:

* Training loss
* Validation loss 

We describe the algorithm's final performance after training was completed by showing a precision-recall curve on our validation set.

We report the threshold for classification that we chosen and the corresponded F1 score, recall, and precision. 

**4. Databases**

For the database of patient data used, we provide specific information about the training and validation datasets that we curated separately, including: 

* Size of the dataset
* The number of positive cases and the its radio to the number of negative cases
* The patient demographic data (as it is available)
* The radiologic techniques used and views taken
* The co-occurrence frequencies of pneumonia with other diseases and findings

**5. Ground Truth**

The methodology used to establish the ground truth can impact reported performance. We describe how the NIH created the ground truth for the data that was provided to us for this project. We describe the benefits and limitations of this type of ground truth.  

**6. FDA Validation Plan**

We describe how a FDA Validation Plan would be conducted for our algorithm, rather than actually performing the assessment. We describe the following: 

* The patient population that we would request imaging data from from your clinical partner:
    * Age ranges
    * Sex
    * Type of imaging modality
    * Body part imaged
    * Prevalence of disease of interest
    * Any other diseases that should be included _or_ excluded as comorbidities in the population

* We provide a short explanation of how we would obtain an optimal ground truth 
* We provide a performance standard that we choose based on [this paper.](https://arxiv.org/pdf/1711.05225.pdf)

