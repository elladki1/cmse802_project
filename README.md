# cmse802_project
This repository contains the CMSE 802 final project and all relevant files.

* Project Title: Predicting the Radius of Gyration of Small Molecules Using Machine Learning
* Brief Description: The project will be on predicting the radius of gyration (RG2) of small molecules using molecular descriptors and machine learning (ML)  models. The ML models will be used to identify which desriptors have the strongest influence on RG2 and as a predictor of RG2 for unknown molecules. The RG2 is a structural property that provides information on polymer/molecule behavior. 
* Objectives:
1. Determine the ideal model to use such as linear vs non-linear model --> aligns with the topic we covered on models.

2. Determine which model minimizes the root mean squared error --> aligns with optimizing the models we use.

3. Determine which physicochemical fingerprints impact the radius of gyration --> aligns with data processing and machine learning. 
* Instructions: Will be updated once the code for the project is complete.


The layout of the directory is in the following structure:

```
├── data/
│    ├── raw - raw data from QM9 database/
│    └── processed - processed data with missing RG2 vals calculated/
│
├── reports - contains project report/
│
├── results - contains project results for each ML model/
│
├── src - contains relevant code/
│   ├── data/
│   ├── physicochemical discriptors/features/
│   ├── models/
|   │   ├── RR/
│   │   └── ANN/
│   └── visualization/
│
├── statistics/
│   ├── R2/
│   └── RMSE/
│
├── tests - unit tests/
│
├── .gitignore
└── README.md
```