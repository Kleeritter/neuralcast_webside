import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy
import contextily as ctx
from cartopy.io.img_tiles import OSM

import cartopy.io.img_tiles as cimgt

def create_map(station_lon, station_lat):
    # Definieren Sie die Projektion und erstellen Sie eine Karte
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    # Hinzufügen von geografischen Features (Ländergrenzen, Küstenlinien usw.)
    ax.coastlines(resolution='10m', color='black', linewidth=1)
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1)

    # Definieren Sie den Kartenausschnitt basierend auf der Position der Messstation
    lon_range = [station_lon - 5.5, station_lon + 5.5]
    lat_range = [station_lat - 5.5, station_lat + 5.5]
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
    # Hinzufügen von geografischen Features (Ländergrenzen, Küstenlinien usw.)
    ax.coastlines(resolution='10m', color='black', linewidth=1)
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1)

    # Hinzufügen von Landschaftsstrukturen (Flüsse, Seen, Berge, Städte usw.)
    ax.add_feature(cartopy.feature.RIVERS)
    ax.add_feature(cartopy.feature.LAKES)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black')
    ax.add_feature(cartopy.feature.OCEAN, facecolor='lightblue')
    #ax.add_feature(cartopy.feature.LANDCOVER, facecolor='lightgreen')
    # Markieren Sie die Position der Messstation
    ax.plot(station_lon, station_lat, marker='o', color='red', markersize=8, transform=ccrs.PlateCarree())

    # Zeigen Sie die Karte an
    plt.show()


def create_map_with_osm(station_lon, station_lat, dach_lon=0, dach_lat=0, dach=False,title="Measuring sites Hannover-Herrenhausen"):
    # Definieren Sie die Projektion und erstellen Sie eine Karte mit Kartopy als Hintergrundstruktur
    #fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    imagery = OSM()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=imagery.crs)
    #ax.set_extent([-0.14, -0.1, 51.495, 51.515], ccrs.PlateCarree())
    # Hinzufügen von geografischen Features (Ländergrenzen, Küstenlinien usw.)
    #ax.coastlines(resolution='10m', color='black', linewidth=1)
    #ax.add_feature(cartopy.feature.BORDERS, linestyle='-', linewidth=1)

    range_lon = 0.0125/2
    range_lat = 0.0075/2
    # Definieren Sie den Kartenausschnitt basierend auf der Position der Messstation
    lon_range = [station_lon - range_lon, station_lon + range_lon]
    lat_range = [station_lat - range_lat, station_lat + range_lat]
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
    ax.add_image(imagery, 16)
    # Markieren Sie die Position der Messstation
    ax.plot(station_lon, station_lat, marker='o', color='red', markersize=12, transform=ccrs.PlateCarree(),label="Tower")
    #ax.plot(station_lon, station_lat, marker='o', color='red', markersize=8, transform=ccrs.PlateCarree())
    if dach != False: ax.plot(dach_lon, dach_lat, marker='o', color='blue', markersize=12, transform=ccrs.PlateCarree(),label="Roof")

    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend( fontsize=18)
    plt.title(title ,fontsize=20)
    #plt.show()
    plt.savefig('/home/alex/Dokumente/Bach/figures/Herrenhausen.png', bbox_inches='tight')



if __name__ == '__main__':
    # Koordinaten der Messstation in Hannover (Beispielkoordinaten)
    station_lon = 9.7048090
    station_lat = 52.3938243

    dach_lat = 52.39134
    dach_lon = 9.70477

    ruthe_lat = 52.243490
    ruthe_lon = 9.813308
    #create_map(station_lon, station_lat)
    create_map_with_osm(station_lon, station_lat, dach_lon, dach_lat,dach=True)
    #create_map_with_osm(station_lon=ruthe_lon,station_lat=ruthe_lat,dach=False,title="Measuring site Ruthe")