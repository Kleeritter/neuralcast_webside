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
For a quick test, run the following command:
```
python ./Model/run_lstm_multi.py
```
## Data
The Data folder contains the means to extract the data from the original dataset, resample it and add derivate parameters.
## Model
The Model folder contains all the Models used: SARIMA,univariate LSTM and multivariate LSTM.

The actual model and dataset definitions are in the funcs folder. The hyperparameter optimisation procedure can be found under the subfolder opti, with the code and results. Corpars and Cortest subfolders contain the procedure for the correlation analysis. Ruthe subfolder contains the code for the Ruthe dataset and timetest subfolder contains the code for the test to find optimal forecast horizons and windowsizes.

run_lstm_multi.py is the main file to run the multivariate LSTM model with optimized hyperparameters and to produce a netcdf output file with the predictions.

tem_hour and tem_hour_multi are just for testing purposes.

## Visaualization
The Visualiton folder contains all the means to visualize the data and compute the error metrics.