import numpy as np
import matplotlib
import matplotlib as mpl
#from score_visualer import tagesmax,tagesmin,luftdrucktendenz
import matplotlib.pyplot as plt
from matplotlib import colors
import xarray as xr
import math
from sklearn.metrics import mean_squared_error
def tagesmax(data):
    tagesmax=data.groupby(data.index.date).max()
    return tagesmax

def tagesmin(data):
    tagesmin=data.groupby(data.index.date).min()
    return tagesmin


def heatmap_rmse(data,timetest,forecast_horizonts,window_sizes,tmax=0,var="temp"):
    plasma_without_yellow = colors.LinearSegmentedColormap.from_list(
        'PlasmaWithoutYellow',
        [(0.0, '#A3BE8C'),  # Dunkelblau
         (0.25, '#81A1C1'),  # Blau
         (0.50, '#B48EAD'),  # Violett
         (0.75, '#D08770'),  # Orange
         (1.0, '#BF616A')]  # Rot
    )
    monokai_colors = ['#272822', '#F92672', '#66D9EF', '#A6E22E', '#FD971F']
    dracula_colors = [
"#ffb14e",
"#fa8775",
"#ea5f94",
"#cd34b5",
"#9d02d7",
"#0000ff"]
    dracula_cmap = colors.LinearSegmentedColormap.from_list('Dracula', dracula_colors)

    # Erstellung der benutzerdefinierten Colormap
    monokai_cmap = colors.LinearSegmentedColormap.from_list('Monokai', monokai_colors)

    harvest = np.zeros((len(forecast_horizonts), len(window_sizes)))
    for i in range(0, len(window_sizes)):
        hans=0
        for j in range(0, len(forecast_horizonts)):
            match tmax:
                case 0:
                    rmse= math.sqrt(mean_squared_error(data[var], timetest[str(window_sizes[i])+"_"+str(forecast_horizonts[j])+"_"+var]))
                case 1:
                    rmse= math.sqrt(mean_squared_error(tagesmax(data)[var], tagesmax(timetest[str(window_sizes[i])+"_"+str(forecast_horizonts[j])+"_"+var])))
                case 2:
                    rmse= math.sqrt(mean_squared_error(tagesmin(data)[var], tagesmin(timetest[str(window_sizes[i])+"_"+str(forecast_horizonts[j])+"_"+var])))
            harvest[i,j]=rmse
            hans+=1


    #harvest=np.random.rand(len(forecast_horizonts),len(window_sizes))

    harvest_unroudnded=harvest
    harvest=np.round(harvest,2)

    fig, ax = plt.subplots()
    im = ax.imshow(harvest,cmap=dracula_cmap)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(forecast_horizonts)), labels=forecast_horizonts)
    ax.set_yticks(np.arange(len(window_sizes)), labels=window_sizes)
    ax.set_xlabel("forecast horrizon [hours]")
    ax.set_ylabel("window size [hours]")
    # Rotate the tick labels and set their alignment.
   # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")
    cbar = ax.figure.colorbar(im, ax=ax, )
    cbar.ax.set_ylabel("RMSE", rotation=-90, va="bottom")
    # Loop over data dimensions and create text annotations.
    for i in range(len(forecast_horizonts)):
        for j in range(len(window_sizes)):
            text = ax.text(j, i, harvest[i, j], ha="center", va="center", color="w")
            text.set_fontsize(8)
            if harvest_unroudnded[i, j] == np.min(harvest_unroudnded[:, j]):
                text.set_weight("bold")
                text.set_fontsize(9)  # Anpassen der Schriftgröße für die besten Werte
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='w', linewidth=1, facecolor='none')
                ax.add_patch(rect)

    ax.set_title("Root Mean Squared Error for Temperature")
    fig.tight_layout()
    plt.savefig("/home/alex/Dokumente/Bach/figures/time_eval_temp.png", dpi=300)
   # plt.show()
    return

def heatmap_skills(data,timetest,forecast_horizonts,window_sizes,compare,var="temp"):
    harvest = np.zeros((len(forecast_horizonts), len(window_sizes)))
    for i in range(0, len(window_sizes)):
        hans=0
        for j in range(0, len(forecast_horizonts)):
            rmse= math.sqrt(mean_squared_error(data[var], timetest[str(window_sizes[i])+"_"+str(forecast_horizonts[j])+"_"+var]))
            compare_rmse=math.sqrt(mean_squared_error(data[var],  compare[str(window_sizes[i])+"_"+str(forecast_horizonts[j])]))
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
timetest="../Model/timetest/lstm_multi/output/fullsets/temp_full.nc"#"timetest_full_new.nc"
timetest=xr.open_dataset(timetest).to_dataframe()

references_path="timetest_sarima.nc"
compare=xr.open_dataset(references_path).to_dataframe()
forecast_horizonts=[2,4,6,12,15,18,24,32,48,60,72]
window_sizes=[2*7*24,7*24,6*24,5*24,4*24,3*24,2*24,24,12,6,3] #16*7*24,8*7*24,4*7*24
heatmap_rmse(data,timetest,forecast_horizonts,window_sizes,tmax=0,var="temp")
#heatmap_skills(data,timetest,forecast_horizonts,window_sizes,compare)
##Tagesmax
#heatmap_rmse(tagesmax(data),tagesmax(timetest),forecast_horizonts,window_sizes)