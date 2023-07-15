import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
import math
from sklearn.metrics import mean_squared_error
vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat", "barley"]
farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

#from sklearn.metrics import mean_squared_error

harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
#




def heatmap_rmse(data,timetest,forecast_horizonts,window_sizes):
    harvest = np.zeros((len(forecast_horizonts), len(window_sizes)))
    for i in range(0, len(window_sizes)):
        hans=0
        for j in range(0, len(forecast_horizonts)):
            rmse= math.sqrt(mean_squared_error(data["temp"], timetest[str(window_sizes[i])+"_"+str(forecast_horizonts[j])+"_temp"]))
            harvest[i,j]=rmse
            hans+=1


    #harvest=np.random.rand(len(forecast_horizonts),len(window_sizes))

    harvest=np.round(harvest,2)

    fig, ax = plt.subplots()
    im = ax.imshow(harvest,cmap="plasma")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(forecast_horizonts)), labels=forecast_horizonts)
    ax.set_yticks(np.arange(len(window_sizes)), labels=window_sizes)
    ax.set_xlabel("Forecast Horizont")
    ax.set_ylabel("Window Size")
    # Rotate the tick labels and set their alignment.
   # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")
    cbar = ax.figure.colorbar(im, ax=ax, )
    cbar.ax.set_ylabel("RMSE", rotation=-90, va="bottom")
    # Loop over data dimensions and create text annotations.
    for i in range(len(forecast_horizonts)):
        for j in range(len(window_sizes)):
            text = ax.text(j, i, harvest[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Root Mean Squared Error for Temperature")
    fig.tight_layout()
    plt.show()
    return

def heatmap_skills(data,timetest,forecast_horizonts,window_sizes,compare):
    harvest = np.zeros((len(forecast_horizonts), len(window_sizes)))
    for i in range(0, len(window_sizes)):
        hans=0
        for j in range(0, len(forecast_horizonts)):
            rmse= math.sqrt(mean_squared_error(data["temp"], timetest[str(window_sizes[i])+"_"+str(forecast_horizonts[j])+"_temp"]))
            compare_rmse=math.sqrt(mean_squared_error(data["temp"],  compare[str(window_sizes[i])+"_"+str(forecast_horizonts[j])]))
            harvest[i,j]=1-(rmse/compare_rmse)
            #hans+=1


    #harvest=np.random.rand(len(forecast_horizonts),len(window_sizes))

    harvest=np.round(harvest,2)

    fig, ax = plt.subplots()
    im = ax.imshow(harvest,cmap="plasma")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(forecast_horizonts)), labels=forecast_horizonts)
    ax.set_yticks(np.arange(len(window_sizes)), labels=window_sizes)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_xlabel("Forecast Horizont")
    ax.set_ylabel("Window Size")
    # Loop over data dimensions and create text annotations.
    for i in range(len(forecast_horizonts)):
        for j in range(len(window_sizes)):
            text = ax.text(j, i, harvest[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("SKillscores for Temperature")
    fig.tight_layout()
    plt.show()
    return

forecast_year=2022
nc_path = '../Data/stunden/'+str(forecast_year)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file
data = xr.open_dataset(nc_path).to_dataframe()
timetest="timetest_full.nc"
timetest=xr.open_dataset(timetest).to_dataframe()
references_path="timetest_sarima.nc"
compare=xr.open_dataset(references_path).to_dataframe()
forecast_horizonts=[2,4,6,12,15,18,24,32,48,60,72,84,96]
window_sizes=[8*7*24,4*7*24,2*7*24,7*24,6*24,5*24,4*24,3*24,2*24,24,12,6,3]
#heatmap_rmse(data,timetest,forecast_horizonts,window_sizes)
heatmap_skills(data,timetest,forecast_horizonts,window_sizes,compare)