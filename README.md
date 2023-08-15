# neuralcast

Repository for the NeuralCast project.

## Installation
To install the project, clone the repository and install the requirements with conda/mamba:
```
conda env create -f environment.yml
```
Then activate the environment:
```
conda activate neuralcast
```

## Data
The Data folder contains the means to extract the data from the original dataset, resample it and add derivate parameters.
## Model
The Model folder contains all the Models used: SARIMA,univariate LSTM and multivariate LSTM.

## Visaualization
The Visualiton folder contains all the means to visualize the data and compute the error metrics.