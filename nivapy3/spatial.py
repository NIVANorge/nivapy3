import getpass
import json
import logging
import os
import shutil
import sys
import warnings
from pathlib import Path

os.environ["KMP_WARNINGS"] = "off"  # Prevents confusing warning linked to PySheds

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import folium
import geopandas as gpd
import geopandas.tools
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import pyproj
import rasterio
import rasterio.mask
import xarray as xr
from folium.plugins import FastMarkerCluster, MarkerCluster
from osgeo import gdal, ogr, osr
from osgeo.gdalconst import GA_ReadOnly
from pyresample.geometry import AreaDefinition
from rasterio import features
from rasterstats import zonal_stats
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.spatial import cKDTree as KDTree
from shapely.geometry import Point, Polygon, box, shape
from skimage.morphology import skeletonize
from tqdm.notebook import tqdm

from . import da


def quickmap(
    df,
    lon_col="longitude",
    lat_col="latitude",
    popup=None,
    cluster=False,
    tiles="Stamen Terrain",
    aerial_imagery=False,
    kartverket=False,
    layer_name="Stations",
):
    """Make an interactive map from a point dataset. Can be used with any dataframe
    containing lat/lon co-ordinates (in WGS84 decimal degrees), but primarily
    designed to be used directly with the functions in nivapy.da.

    Args:
        df:            Dataframe. Must include columns for lat and lon in WGS84 decimal degrees
        lon_col:       Str. Column with longitudes
        lat_col:       Str. Column with latitudes
        popup:         Str or None. Default None. Column containing text for popup labels
        cluster:       Bool. Whether to implement marker clustering
        tiles:         Str. Basemap to use. See folium.Map for full details. Choices:
                            - 'OpenStreetMap'
                            - 'Mapbox Bright' (Limited levels of zoom for free tiles)
                            - 'Mapbox Control Room' (Limited levels of zoom for free tiles)
                            - 'Stamen' (Terrain, Toner, and Watercolor)
                            - 'Cloudmade' (Must pass API key)
                            - 'Mapbox' (Must pass API key)
                            - 'CartoDB' (positron and dark_matter)
                            - Custom tileset by passing a Leaflet-style URL to the tiles
                              parameter e.g. http://{s}.yourtiles.com/{z}/{x}/{y}.png
        aerial_imagery: Bool. Whether to include Google satellite serial imagery as an
                        additional layer
        kartverket:     Bool. Whether to include Kartverket's topographic map as an additonal
                        layer
        layer_name:     Str. Name of layer to create in "Table of Contents"

    Returns:
        Folium map
    """
    # Drop NaN
    df2 = df.dropna(subset=[lon_col, lat_col])

    # Get data
    if popup:
        df2 = df2[[lat_col, lon_col, popup]]
        df2[popup] = df2[popup].astype(str)
    else:
        df2 = df2[[lat_col, lon_col]]

    # Setup map
    avg_lon = df2[lon_col].mean()
    avg_lat = df2[lat_col].mean()
    map1 = folium.Map(location=[avg_lat, avg_lon], zoom_start=4, tiles=tiles)

    # Add aerial imagery if desired
    if aerial_imagery:
        folium.raster_layers.TileLayer(
            tiles="http://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="google",
            name="Google satellite",
            max_zoom=20,
            subdomains=["mt0", "mt1", "mt2", "mt3"],
            overlay=False,
            control=True,
        ).add_to(map1)

    if kartverket:
        folium.raster_layers.TileLayer(
            tiles="https://opencache.statkart.no/gatekeeper/gk/gk.open_gmaps?layers=topo4&zoom={z}&x={x}&y={y}",
            attr="karkverket",
            name="Kartverket topographic",
            overlay=False,
            control=True,
        ).add_to(map1)

    # Add feature group to map
    grp = folium.FeatureGroup(name=layer_name)

    # Draw points
    if cluster and popup:
        locs = list(zip(df2[lat_col].values, df2[lon_col].values))
        popups = list(df2[popup].values)

        # Marker cluster with labels
        marker_cluster = MarkerCluster(locations=locs, popups=popups)
        grp.add_child(marker_cluster)
        grp.add_to(map1)

    elif cluster and not popup:
        locs = list(zip(df2[lat_col].values, df2[lon_col].values))
        marker_cluster = FastMarkerCluster(data=locs)
        grp.add_child(marker_cluster)
        grp.add_to(map1)

    elif not cluster and popup:  # Plot separate circle markers, with popup
        for idx, row in df2.iterrows():
            marker = folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=5,
                weight=1,
                color="black",
                popup=folium.Popup(row[popup], parse_html=False),
                fill_color="red",
                fill_opacity=1,
            )
            grp.add_child(marker)
        grp.add_to(map1)

    else:  # Plot separate circle markers, no popup
        for idx, row in df2.iterrows():
            marker = folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=5,
                weight=1,
                color="black",
                fill_color="red",
                fill_opacity=1,
            )
            grp.add_child(marker)
        grp.add_to(map1)

    # Add layer control
    folium.LayerControl().add_to(map1)

    # Zoom to data
    xmin, xmax = df2[lon_col].min(), df2[lon_col].max()
    ymin, ymax = df2[lat_col].min(), df2[lat_col].max()
    map1.fit_bounds([[ymin, xmin], [ymax, xmax]])

    return map1


def plot_vector(vec_path, fill_color="yellow", line_color="black", fill_opacity=0.2):
    """Folium map of vector data (e.g. shapefile, geojson, etc.).

    Args:
        vec_path: Raw str. Path to vector dataset (.sho, .geojson etc.)
                  Can be any format read by GeoPandas

    Returns:
        Folium map
    """
    # Read shapefile
    gdf = gpd.read_file(vec_path)

    # Convert to WGS84 json
    gdf = gdf.to_crs("epsg:4326")
    gjson = gdf.to_json()

    # Map
    m = folium.Map(location=[65, 10], zoom_start=4, tiles="Stamen Terrain")

    m.choropleth(
        geo_data=gjson,
        fill_color=fill_color,
        line_color=line_color,
        fill_opacity=fill_opacity,
    )

    # Zoom to data
    xmin, xmax = gdf.bounds["minx"].min(), gdf.bounds["maxx"].max()
    ymin, ymax = gdf.bounds["miny"].min(), gdf.bounds["maxy"].max()
    m.fit_bounds([[ymin, xmin], [ymax, xmax]])

    return m


def plot_polygons_overlay_points(
    poly_vec_path,
    pts_df,
    poly_layer="Polygons",
    poly_fill_color="yellow",
    poly_line_color="black",
    poly_fill_opacity=0.2,
    pts_layer="Points",
    pts_lon_col="longitude",
    pts_lat_col="latitude",
    pts_popup=None,
    tiles="Stamen Terrain",
):
    """Plot a polygon vector dataset (e.g. .shp, .geojson) and overlay it with
    a point dataset.

    NOTE: This function is not designed to handle huge datasets. For large
    shapefiles or with thousands of points, you'll need to use a GIS package.
    If the map appears as a blank white box, you're probably trying to plot
    too much data for your browser to be able to cope.

    Args:
        poly_vec_path:     Raw str. Path to vector polygons (.shp, .geojson etc.)
        pts_df:            Dataframe with lat/lon values in WGS84 decimal degrees
        poly_layer:        Str. Name for polygon layer in Table of Contents
        poly_fill_color:   Str. Matplotlib color name
        poly_line_color:   Str. Matplotlib color name
        poly_fill_opacity: Float. Between 0 and 1.
        pts_layer:         Str. Name for points layer in Table of Contents
        pts_lon_col:       Str. Name of longitude column in dataframe
        pts_lat_col:       Str. Name of latitude column in dataframe
        pts_popup:         Str. Name of column to use for popups
        tiles:             Str. Basemap to use. See folium.Map for full details. Choices:
                             - 'OpenStreetMap'
                             - 'Mapbox Bright' (Limited levels of zoom for free tiles)
                             - 'Mapbox Control Room' (Limited levels of zoom for free tiles)
                             - 'Stamen' (Terrain, Toner, and Watercolor)
                             - 'Cloudmade' (Must pass API key)
                             - 'Mapbox' (Must pass API key)
                             - 'CartoDB' (positron and dark_matter)
                             - Custom tileset by passing a Leaflet-style URL to the tiles
                               parameter e.g. http://{s}.yourtiles.com/{z}/{x}/{y}.png

    Returns:
        Folium map
    """
    assert (poly_fill_opacity >= 0) and (
        poly_fill_opacity <= 1
    ), "Opacity must be between 0 and 1."

    # Drop NaN
    df2 = pts_df.dropna(subset=[pts_lon_col, pts_lat_col])

    # Get data
    if pts_popup:
        df2 = df2[[pts_lat_col, pts_lon_col, pts_popup]]
        df2[pts_popup] = df2[pts_popup].astype(str)
    else:
        df2 = df2[[pts_lat_col, pts_lon_col]]

    # Read polys
    gdf = gpd.read_file(poly_vec_path)

    # Check polygons
    assert set(gdf.geom_type).issubset(
        set(["Polygon", "MultiPolygon"])
    ), "'poly_vec_path' must be a polygon dataset."

    # Convert to WGS84 json
    gdf = gdf.to_crs("epsg:4326")
    gjson = gdf.to_json()

    # Setup map
    avg_lon = df2[pts_lon_col].mean()
    avg_lat = df2[pts_lat_col].mean()
    map1 = folium.Map(location=[avg_lat, avg_lon], zoom_start=4, tiles=tiles)

    # Create feature groups
    shp_fg = folium.FeatureGroup(name=poly_layer)
    pts_fg = folium.FeatureGroup(name=pts_layer)

    # Add shp data
    polys = folium.GeoJson(
        data=gjson,
        style_function=lambda feature: {
            "fillColor": poly_fill_color,
            "color": poly_line_color,
            "weight": 1,
            "fillOpacity": poly_fill_opacity,
        },
        name=poly_layer,
    )
    polys.add_to(shp_fg)
    shp_fg.add_to(map1)

    # Add pt data
    if pts_popup:
        for idx, row in df2.iterrows():
            marker = folium.CircleMarker(
                location=[row[pts_lat_col], row[pts_lon_col]],
                radius=5,
                weight=1,
                color="black",
                popup=folium.Popup(row[pts_popup], parse_html=True),
                fill_color="red",
                fill_opacity=1,
            )
            pts_fg.add_child(marker)
        pts_fg.add_to(map1)

    else:
        for idx, row in df2.iterrows():
            marker = folium.CircleMarker(
                location=[row[pts_lat_col], row[pts_lon_col]],
                radius=5,
                weight=1,
                color="black",
                fill_color="red",
                fill_opacity=1,
            )
            pts_fg.add_child(marker)
        pts_fg.add_to(map1)

    # Add layer control
    folium.LayerControl().add_to(map1)

    # Zoom to data
    xmin = min([gdf.bounds["minx"].min(), df2[pts_lon_col].min()])
    xmax = max([gdf.bounds["maxx"].max(), df2[pts_lon_col].max()])
    ymin = min([gdf.bounds["miny"].min(), df2[pts_lat_col].min()])
    ymax = max([gdf.bounds["maxy"].max(), df2[pts_lat_col].max()])
    map1.fit_bounds([[ymin, xmin], [ymax, xmax]])

    return map1


def utm_to_wgs84_dd(utm_df, zone="utm_zone", east="utm_east", north="utm_north"):
    """Converts UTM co-ordinates to WGS84 decimal degrees.

    Args:
        utm_df: Dataframe containing UTM co-ords
        zone:   Str. Column defining UTM zone
        east:   Str. Column defining UTM Easting
        north:  Str. Column defining UTM Northing

    Returns:
        Copy of utm_df with 'lat' and 'lon' columns added.
    """
    # Copy utm_df
    df = utm_df.copy()

    # Containers for data
    lats = []
    lons = []

    # Loop over df
    for idx, row in df.iterrows():
        # Only convert if UTM co-ords are available
        if pd.isnull(row[east]) or pd.isnull(row[north]) or pd.isnull(row[zone]):
            lats.append(np.nan)
            lons.append(np.nan)
        else:
            # Build projection
            p = pyproj.Proj(proj="utm", zone=row[zone], ellps="WGS84")

            # Convert
            lon, lat = p(row[east], row[north], inverse=True)
            lats.append(lat)
            lons.append(lon)

    # Add to df
    df["lat"] = lats
    df["lon"] = lons

    return df


def wgs84_dd_to_utm(dd_df, lat="latitude", lon="longitude", utm_zone=33):
    """Converts WGS84 decimal degrees to a specified UTM zone.

    Args:
        dd_df:     Dataframe containing decimal degrees
        lat:       Str. Column defining latitudes
        lon:       Str. Column defining longitudes
        utm_zone:  Str. Desired UTM zone

    Returns:
        Copy of dd_df with 'utm_east', 'utm_north' and 'utm_zone' columns
        added.
    """
    # Copy utm_df
    df = dd_df.copy()

    # Containers for data
    easts = []
    norths = []

    # Loop over df
    for idx, row in df.iterrows():
        # Only convert if co-ords are available
        if pd.isnull(row[lat]) or pd.isnull(row[lon]):
            easts.append(np.nan)
            norths.append(np.nan)
        else:
            # Build projection
            p = pyproj.Proj(proj="utm", zone=utm_zone, ellps="WGS84")

            # Convert
            east, north = p(row[lon], row[lat])
            norths.append(north)
            easts.append(east)

    # Add to df
    df["utm_north"] = norths
    df["utm_east"] = easts
    df["utm_zone"] = utm_zone

    return df


def choropleth_from_gdf(
    gdf,
    val_col,
    geom="geom",
    tiles="Stamen Terrain",
    fill_color="YlOrRd",
    fill_opacity=1,
    line_opacity=1,
    legend_name="Legend title",
):
    """Create a choropleth map from a geodataframe. The geodataframe must have the
    'crs' correctly defined.

    Args:
        gdf:          Geodataframe
        val_col:      Str. Name of gdf value column
        geom:         Str. Name of gdf geometry column
        tiles:        Str. Basemap to use. See folium.Map for full details. Choices:
                        - 'OpenStreetMap'
                        - 'Mapbox Bright' (Limited levels of zoom for free tiles)
                        - 'Mapbox Control Room' (Limited levels of zoom for free tiles)
                        - 'Stamen' (Terrain, Toner, and Watercolor)
                        - 'Cloudmade' (Must pass API key)
                        - 'Mapbox' (Must pass API key)
                        - 'CartoDB' (positron and dark_matter)
                        - Custom tileset by passing a Leaflet-style URL to the tiles
                          parameter e.g. http://{s}.yourtiles.com/{z}/{x}/{y}.png
        fill_color:   Str. Valid Color Brewer palette:
                      'BuGn', 'BuPu', 'GnBu', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd',
                      'RdPu', 'YlGn', 'YlGnBu', 'YlOrBr', and 'YlOrRd'
        fill_opacity: Float. Between 0 and 1
        line_opacity: Float. Between 0 and 1
        legend_name:  Str. Title for legend

    Returns:
        Folium map object
    """
    # Check polygons
    assert set(gdf.geom_type).issubset(
        set(["Polygon", "MultiPolygon"])
    ), "'gdf' must be a polygon dataset."

    # Get index name, then reset so have something to join on
    gdf2 = gdf.copy()
    idx = gdf2.index.name
    if idx is None:
        idx = "index"
    gdf2.reset_index(inplace=True)

    # Convert to WGS84 json
    gjson = gdf2.to_crs("epsg:4326").to_json()

    # Get data component for choropleth
    data = pd.DataFrame(gdf2[[idx, val_col]])

    # Map
    m = folium.Map(location=[65, 10], zoom_start=4, tiles=tiles)

    m.choropleth(
        geo_data=gjson,
        data=data,
        columns=[idx, val_col],
        key_on="feature.properties.%s" % idx,
        fill_color=fill_color,
        fill_opacity=fill_opacity,
        line_opacity=line_opacity,
        legend_name=legend_name,
    )

    return m


def read_raster(src_path, band_no=1, flip=False, pg_dict=None):
    """Reads a raster to a numpy array. The raster can either be a file or
    a table in a PostGIS database.

    NOTE: To read from PostGIS, you will be asked to supply a valid user
    name and password.

    Args:
        src_path: Str. Either path to file or 'postgres'
        band_no:  Int. Band number to read
        flip:     Bool. Whether to flip array. Images and arrays are often
                  indexed differently. The default should be fine if the raster
                  was saved correctly initially
        pg_dict:  Connection and dataaset parameters for PostGIS database.
                  Only required if 'src_path'='postgres'.

                  NOTE: All values in the dict should be strings.

                      {'dbname':'database_name',
                       'host':'hostname',
                       'port':'port',
                       'schema':'schema',
                       'table':'table',
                       'column':'column'}

    Returns:
        Tuple: (array, NDV, epsg, (xmin, xmax, ymin, ymax))
        No data values in the array are set to np.nan
    """
    # Register drivers
    gdal.AllRegister()

    # Establish type of data source
    if (src_path == "postgres") and (pg_dict is None):
        raise ValueError(
            "The 'pg_dict' argument is mandatory when connecting to PostGIS."
        )

    elif (src_path == "postgres") and pg_dict:
        # Get credentials
        pg_dict["user"] = getpass.getpass(prompt="Username: ")
        pg_dict["pw"] = getpass.getpass(prompt="Password: ")

        # Build conn_str for GDAL
        conn_str = (
            "PG: dbname={dbname}"
            "    host={host}"
            "    user={user}"
            "    password={pw}"
            "    port={port}"
            "    schema={schema}"
            "    table={table}"
            "    column={column}"
            "    mode=2"
        ).format(**pg_dict)

        # Open dataset
        ds = gdal.Open(conn_str, gdal.GA_ReadOnly)

    else:
        # Assume file raster
        ds = gdal.Open(src_path, gdal.GA_ReadOnly)

    # Check if successful
    if ds is None:
        print("Could not open dataset.")
        sys.exit(1)

    # Dataset properties
    geotransform = ds.GetGeoTransform()
    originX = geotransform[0]  # Origin is top-left corner
    originY = geotransform[3]  # i.e. (xmin, ymax)
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = ds.RasterXSize
    rows = ds.RasterYSize

    # Get projection
    proj = osr.SpatialReference(wkt=ds.GetProjection())
    epsg = int(proj.GetAttrValue("AUTHORITY", 1))

    # Calculate extent
    xmin = int(originX)
    xmax = int(originX + cols * pixelWidth)
    ymin = int(originY + rows * pixelHeight)
    ymax = int(originY)

    # Read band 1
    band = ds.GetRasterBand(band_no)

    # Get NDV
    no_data_val = band.GetNoDataValue()

    # Read the data to an array
    data = band.ReadAsArray()

    # Set NDV to np.nan. Only applicable to float arrays, as Python
    # has no integer representation of NaN
    if no_data_val:
        if data.dtype == "float":
            data[data == no_data_val] = np.nan

    if flip:
        data = data[::-1, :]

    # Close the dataset
    ds = None

    return (data, no_data_val, epsg, (xmin, xmax, ymin, ymax))


def array_to_gtiff(
    xmin,
    ymax,
    cell_size,
    out_path,
    data_array,
    proj4_str,
    no_data_value=-9999,
    flip=False,
    bit_depth="Float32",
    compress=True,
):
    """Save numpy array as a single band GeoTiff (in a projected
    co-ordinate system).

    Args:
        xmin:          Int. Minimum x value in metres
        ymax:          Int. Maximum y value in metres
        cell_size:     Int. Grid cell size in metres
        out_path:      Raw str. Path to GeoTiff
        data:          Array. Array to save
        proj4_str      Str. proj.4 string defining the projection
        no_data_value: Int. Value to use to represent no data
        flip:          Bool. Whether to flip array. Images and arrays are often
                       indexed differently. The default should be fine if the raster
                       was saved correctly initially
        bit_depth:     Str. Specify bit depth. Choosing wisely can dramatically
                       reduce file sizes, but beware of numerical
                       over/underflow. Must be one of the following strings:

                           ('Byte', 'Int16', 'UInt16', 'UInt32',
                            'Int32', 'Float32', 'Float64')

        compress:      Bool. Whether to use LZW compression to reduce the size
                       of the saved grid. Can dramatically reduce file sizes,
                       but adds processing overhead for compression and
                       decompression

    Returns:
        None. Array is saved to specified path.
    """
    # Check and get bit depth
    bit_depth_dict = {
        "Byte": gdal.GDT_Byte,
        "Int16": gdal.GDT_Int16,
        "UInt16": gdal.GDT_UInt16,
        "UInt32": gdal.GDT_UInt32,
        "Int32": gdal.GDT_Int32,
        "Float32": gdal.GDT_Float32,
        "Float64": gdal.GDT_Float64,
    }

    assert bit_depth in bit_depth_dict.keys(), "'bit_depth' must be one of %s" % list(
        bit_depth_dict.keys()
    )
    bit_depth = bit_depth_dict[bit_depth]

    # Explicitly set NDV
    data_array[np.isnan(data_array)] = no_data_value

    if flip:
        data_array = data_array[::-1, :]

    # Get array shape
    cols = data_array.shape[1]
    rows = data_array.shape[0]

    # Get driver
    driver = gdal.GetDriverByName("GTiff")

    # Create a new raster data source
    if compress:
        out_ds = driver.Create(
            out_path, cols, rows, 1, bit_depth, options=["COMPRESS=LZW"]
        )
    else:
        out_ds = driver.Create(out_path, cols, rows, 1, bit_depth)

    # Get spatial reference
    sr = osr.SpatialReference()
    sr.ImportFromProj4(proj4_str)
    sr_wkt = sr.ExportToWkt()

    # Write metadata
    # (xmin, cellsize, 0, ymax, 0, -cellsize)
    out_ds.SetGeoTransform((int(xmin), cell_size, 0.0, int(ymax), 0.0, -cell_size))
    out_ds.SetProjection(sr_wkt)
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(no_data_value)
    out_band.WriteArray(data_array)

    # Tidy up
    del out_ds, out_band


def interp_cubic_spline(pts, z, gridx, gridy):
    """Interpolate irregular 2D point data to a regular grid using
    cubic splines. All co-ordinates (x, y, gridx, gridy) should
    be in an appropriate projected co-ordinate system.

    Args:
        pts:    2D array of (x, y) co-ordinate pairs for known points
        z:      Array. 1D array of values to interpolate
        gridx:  Array. 1D array of gridded x values to interpolate
        gridy:  Array. 1D array of gridded y values to interpolate

    Returns:
        Array of interpolated points.
    """
    # Build grid
    xx, yy = np.meshgrid(gridx, gridy)

    # Interpolate
    spl = griddata(pts, z, (xx, yy), method="cubic")

    # Flip
    spl = spl[::-1, :]

    return spl


def interp_bilinear(pts, z, gridx, gridy):
    """Interpolate irregular 2D point data to a regular grid using
    bilinear interpolation. All co-ordinates (x, y, gridx, gridy)
    should be in an appropriate projected co-ordinate system.

    Args:
        pts:    2D array of (x, y) co-ordinate pairs for known points
        z:      Array. 1D array of values to interpolate
        gridx:  Array. 1D array of gridded x values to interpolate
        gridy:  Array. 1D array of gridded y values to interpolate

    Returns:
        Array of interpolated points.
    """
    # Build grid
    xx, yy = np.meshgrid(gridx, gridy)

    # Interpolate
    bil = griddata(pts, z, (xx, yy), method="linear")

    # Flip
    bil = bil[::-1, :]

    return bil


def interp_idw(pts, z, gridx, gridy, n_near=8, p=1):
    """Interpolate irregular 2D point data to a regular grid using
    Inverse Distance Weighting (IDW). All co-ordinates (x, y,
    gridx, gridy) should be in an appropriate projected co-ordinate
    system.

    Simplified interface to Invdisttree class.

    Args:
        pts:    2D array of (x, y) co-ordinate pairs for known points
        z:      Array. 1D array of values to interpolate
        gridx:  Array. 1D array of gridded x values to interpolate
        gridy:  Array. 1D array of gridded y values to interpolate
        n_near: The number of nearest neighbours to consider
        p:      Power defining rate at which weights decrease with distance

    Returns:
        Array of interpolated points.
    """

    # Build interpolator
    invdisttree = Invdisttree(pts, z)

    # Build list of co-ords to interpolate
    xx, yy = np.meshgrid(gridx, gridy)
    pts_i = np.array(list(zip(xx.flatten(), yy.flatten())))

    # Perform interpolation
    interpol = invdisttree(pts_i, nnear=n_near, p=p)

    # Reshape output
    zi = interpol.reshape((len(gridy), len(gridx)))

    # Flip
    zi = zi[::-1, :]

    return zi


class Invdisttree:
    """The code for this class is taken from:

    http://stackoverflow.com/questions/3104781/inverse-distance-weighted-idw-interpolation-with-python

    inverse-distance-weighted interpolation using KDTree.

    invdisttree = Invdisttree( X, z )  -- data points, values
    interpol = invdisttree( q, nnear=3, eps=0, p=1, weights=None, stat=0 )

    interpolates z from the 3 points nearest each query point q;
    For example, interpol[ a query point q ]
    finds the 3 data points nearest q, at distances d1 d2 d3
    and returns the IDW average of the values z1 z2 z3
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
        = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3

    q may be one point, or a batch of points.
    eps: approximate nearest, dist <= (1 + eps) * true nearest
    p: use 1 / distance**p
    weights: optional multipliers for 1 / distance**p, of the same shape as q
    stat: accumulate wsum, wn for average weights

    How many nearest neighbors should one take ?
    a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
    b) make 3 runs with nnear= e.g. 6 8 10, and look at the results --
        |interpol 6 - interpol 8| etc., or |f - interpol*| if you have f(q).
        I find that runtimes don't increase much at all with nnear -- ymmv.

    p=1, p=2 ?
        p=2 weights nearer points more, farther points less.
        In 2d, the circles around query points have areas ~ distance**2,
        so p=2 is inverse-area weighting. For example,
            (z1/area1 + z2/area2 + z3/area3)
            / (1/area1 + 1/area2 + 1/area3)
            = .74 z1 + .18 z2 + .08 z3  for distances 1 2 3
        Similarly, in 3d, p=3 is inverse-volume weighting.

    Scaling:
        if different X coordinates measure different things, Euclidean distance
        can be way off.  For example, if X0 is in the range 0 to 1
        but X1 0 to 1000, the X1 distances will swamp X0;
        rescale the data, i.e. make X0.std() ~= X1.std().

    A nice property of IDW is that it's scale-free around query points:
    if I have values z1 z2 z3 from 3 points at distances d1 d2 d3,
    the IDW average
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
    is the same for distances 1 2 3, or 10 20 30 -- only the ratios matter.
    In contrast, the commonly-used Gaussian kernel exp( - (distance/h)**2 )
    is exceedingly sensitive to distance and to h.
    """

    def __init__(self, X, z, leafsize=10, stat=0):
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        self.tree = KDTree(X, leafsize=leafsize)  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None

    def __call__(self, q, nnear=6, eps=0, p=1, weights=None):
        # nnear nearest neighbours of each query point --
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)

        self.distances, self.ix = self.tree.query(q, k=nnear, eps=eps)
        interpol = np.zeros((len(self.distances),) + np.shape(self.z[0]))
        jinterpol = 0
        for dist, ix in zip(self.distances, self.ix):
            if nnear == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot(w, self.z[ix])
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1 else interpol[0]


def remap_categories(category_map, stats):
    """Helper function for zonal_stats(). Modified from:

    https://gist.github.com/perrygeo/5667173

    Original code copyright 2013 Matthew Perry
    """

    def lookup(m, k):
        """Dict lookup but returns original key if not found"""
        try:
            return m[k]
        except KeyError:
            return k

    return {lookup(category_map, k): v for k, v in stats.items()}


def bbox_to_pixel_offsets(gt, bbox):
    """Helper function for zonal_stats(). Modified from:

    https://gist.github.com/perrygeo/5667173

    Original code copyright 2013 Matthew Perry
    """
    originX = gt[0]
    originY = gt[3]
    pixel_width = gt[1]
    pixel_height = gt[5]
    x1 = int((bbox[0] - originX) / pixel_width)
    x2 = int((bbox[1] - originX) / pixel_width) + 1

    y1 = int((bbox[3] - originY) / pixel_height)
    y2 = int((bbox[2] - originY) / pixel_height) + 1

    xsize = x2 - x1
    ysize = y2 - y1

    return (x1, y1, xsize, ysize)


def zonal_statistics(
    gdf,
    raster_path,
    var_name,
    stats=[
        "min",
        "percentile_25",
        "median",
        "percentile_75",
        "max",
        "mean",
        "std",
        "count",
    ],
    all_touched=False,
):
    """Raster zonal statiscs for vector zones.

    Args
        gdf: Geodataframe of zones
        raster_path: Str. Path to value raster
        var_name:    Str. Name of varaible being summarised. Used to make the columns added to
                     the output easier to interpret. It is recommedned to use 'parameter_unit'
                     e.g. 'runoff_mm/yr'
        stats:       List. Statistics to calculate
        all_touched: Bool. Default False. Defines the rasterisation strategy. See
                     https://pythonhosted.org/rasterstats/manual.html#rasterization-strategy

    Returns
        Geodataframe. Copy of 'gdf' with one new column added for each statistic in 'stats'.
    """
    gdf = gdf.copy()

    with rasterio.open(raster_path) as src:
        crs = src.crs
    stats_gdf = gdf.to_crs(crs)

    df = pd.DataFrame(
        zonal_stats(
            vectors=stats_gdf,
            raster=raster_path,
            stats=stats,
            all_touched=all_touched,
        )
    )

    assert len(gdf) == len(df)

    # Rename for clarity
    name_dict = {i: f"{i}_{var_name}" for i in stats}
    df.rename(name_dict, axis="columns", inplace=True)

    for col in df.columns:
        assert (
            col not in gdf
        ), f"'gdf' already contains a column named {col}. Rename and try again?"
        gdf[col] = df[col]

    # Move 'geometry' to end
    cols = gdf.columns.tolist()
    cols.remove(gdf.geometry.name)
    cols.append(gdf.geometry.name)
    gdf = gdf[cols]

    return gdf


def shp_to_ras(in_shp, out_tif, snap_tif, attrib, ndv, data_type, fmt="GTiff"):
    """Converts a shapefile to a raster with values taken from the 'attrib' field.
    The 'snap_tif' is used to set the resolution and extent of the output raster.

    Args:
        in_shp:    Str. Raw string to shapefile
        out_tif:   Str. Raw string for geotiff to create
        snap_tif:  Str. Raw string to geotiff used to set resolution
                   and extent
        attrib:    Str. Shapefile field for values
        ndv:       Int. No data value
        data_type: Bit depth e.g. gdal.GDT_UInt32. See here for full list:
                   http://www.gdal.org/gdal_8h.html#a22e22ce0a55036a96f652765793fb7a4
        fmt:       Str. Format string.

    Returns:
        None. Raster is saved.
    """
    # 1. Create new, empty raster with correct dimensions
    # Get properties from snap_tif
    snap_ras = gdal.Open(snap_tif)
    cols = snap_ras.RasterXSize
    rows = snap_ras.RasterYSize
    proj = snap_ras.GetProjection()
    geotr = snap_ras.GetGeoTransform()

    # Create out_tif
    driver = gdal.GetDriverByName(fmt)
    out_ras = driver.Create(out_tif, cols, rows, 1, data_type)
    out_ras.SetProjection(proj)
    out_ras.SetGeoTransform(geotr)

    # Fill output with NoData
    out_ras.GetRasterBand(1).SetNoDataValue(ndv)
    out_ras.GetRasterBand(1).Fill(ndv)

    # 2. Rasterize shapefile
    shp_ds = ogr.Open(in_shp)
    shp_lyr = shp_ds.GetLayer()

    gdal.RasterizeLayer(out_ras, [1], shp_lyr, options=["ATTRIBUTE=%s" % attrib])

    # Flush and close
    snap_ras = None
    out_ras = None
    shp_ds = None


def plot_norway_point_data(
    df,
    par,
    lat="latitude",
    lon="longitude",
    vmax_pct=0.95,
    cmap="coolwarm",
    s=30,
    title=None,
    out_path=None,
):
    """ """
    # Check inputs
    assert par != None, '"par" argument must be specified.'

    # Define co-ord system
    crs = ccrs.AlbersEqualArea(
        central_longitude=15,
        central_latitude=65,
        false_easting=650000,
        false_northing=800000,
        standard_parallels=(55, 75),
    )

    # Define Natural Earth data
    # Land
    land_50m = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_0_countries",
        scale="50m",
        edgecolor="black",
        facecolor=cfeature.COLORS["land"],
    )
    # Sea
    sea_50m = cfeature.NaturalEarthFeature(
        category="physical",
        name="ocean",
        scale="50m",
        edgecolor="none",
        facecolor=cfeature.COLORS["water"],
        alpha=0.3,
    )

    # Plot data
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection=crs)
    ax.set_extent([0, 1300000, 0, 1600000], crs=crs)

    # Add geo data
    ax.add_feature(land_50m)
    ax.add_feature(sea_50m)

    # Set max value for colour ramp at 95th percentile
    if vmax_pct:
        pct = (
            df[par]
            .describe(
                percentiles=[
                    0.95,
                ]
            )
            .loc["95%"]
        )
    else:
        pct = pct = df[par].max()

    # Add points using linear colour ramp from 0 to vmax_pct
    cax = ax.scatter(
        df[lon].values,
        df[lat].values,
        c=df[par].values,
        cmap=cmap,
        vmax=pct,
        s=s,
        zorder=5,
        edgecolors="k",
        transform=ccrs.PlateCarree(),
    )

    # Add colourbar
    cbar = fig.colorbar(cax)

    # Title
    if title:
        ax.set_title(title, fontsize=20)
    # plt.tight_layout()

    # Save
    if out_path:
        plt.savefig(out_path, dpi=300)


def plot_norway_raster_data(
    data, vmax_pct=0.95, cmap="coolwarm", title=None, out_path=None
):
    """ """
    # Define co-ord system
    crs = ccrs.AlbersEqualArea(
        central_longitude=15,
        central_latitude=65,
        false_easting=650000,
        false_northing=800000,
        standard_parallels=(55, 75),
    )

    # Define Natural Earth data
    # Sea
    sea_50m = cfeature.NaturalEarthFeature(
        category="physical",
        name="ocean",
        scale="50m",
        edgecolor="none",
        facecolor=cfeature.COLORS["water"],
        alpha=1,
    )

    # Plot data
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection=crs)
    ax.set_extent([0, 1300000, 0, 1600000], crs=crs)

    # Title
    if title:
        ax.set_title(title, fontsize=20)

    # Plot array
    cax = ax.imshow(
        data,
        zorder=1,
        extent=(0, 1300000, 0, 1600000),
        cmap=cmap,
        alpha=0.7,
        interpolation="none",
        vmax=np.nanpercentile(data, int(vmax_pct * 100)),
    )

    # Add colourbar
    cbar = fig.colorbar(cax)

    # Get countries. Make Norway "hollow" so interpolated vaues are visible.
    # Make everywhere else opaque to mask wildly extrapolated values
    shp = cartopy.io.shapereader.natural_earth(
        resolution="50m", category="cultural", name="admin_0_countries"
    )
    reader = cartopy.io.shapereader.Reader(shp)
    countries = reader.records()

    # Loop over countries
    for country in countries:
        if country.attributes["NAME"] == "Norway":
            # Transparent
            ax.add_geometries(
                [country.geometry],
                ccrs.PlateCarree(),  # CRS of Natural Earth data
                facecolor="none",
                edgecolor="black",
                zorder=5,
            )
        else:
            # Opaque
            ax.add_geometries(
                [country.geometry],
                ccrs.PlateCarree(),  # CRS of Natural Earth data
                facecolor=cfeature.COLORS["land"],
                edgecolor="black",
                zorder=5,
            )

    # Add sea
    ax.add_feature(sea_50m, zorder=4)
    plt.tight_layout()

    # Save
    if out_path:
        plt.savefig(out_path, dpi=300)


def spatial_overlays(df1, df2, how="intersection", reproject=True):
    """Perform spatial overlay between two polygons. Improves
    performance compared to current GPD implementation. From here:

    https://github.com/geopandas/geopandas/pull/338#issuecomment-290303715

    Currently only supports GeoDataFrames with (multi-)polygons.

    Args:
        df1:       GeoDataFrame. (Multi-)Polygon geometry column
        df2:       GeoDataFrame. (Multi-)Polygon geometry column
        how:       Str. Method of spatial overlay:
                       - 'intersection'
                       - 'union'
                       - 'identity'
                       - 'symmetric_difference'
                       - 'difference'
                   See http://geopandas.org/set_operations.html for more details
        reproject: Bool. Whether to automatically reproject df2 to the same
                   CRS as df1

    Returns:
        Geodataframe
    """
    # Check user input
    valid_hows = [
        "intersection",
        "union",
        "identity",
        "symmetric_difference",
        "difference",
    ]
    assert how in valid_hows, "'how' must be one of %s." % valid_hows
    assert df1.geom_type.all() in [
        "Polygon",
        "MultiPolygon",
    ], "'df1' must be a polygon dataset."
    assert df2.geom_type.all() in [
        "Polygon",
        "MultiPolygon",
    ], "'df2' must be a polygon dataset."

    # Copy data
    df1 = df1.copy()
    df2 = df2.copy()
    df1["geometry"] = df1.geometry.buffer(0)
    df2["geometry"] = df2.geometry.buffer(0)

    # Reproject if desired
    if df1.crs != df2.crs and reproject:
        print("WARNING: The specified geodataframes have different projections.")
        print("Converting to the projection of first geodataframe (%s)" % df1.crs)
        df2.to_crs(crs=df1.crs, inplace=True)
    elif df1.crs != df2.crs:
        print(
            "WARNING: The specified geodataframes have different projections. "
            "Results may be unreliable. "
        )
        print("Consider passing 'project=True' to automatically re-project.")

    # Perform overlay
    if how == "intersection":
        # Spatial Index to create intersections
        spatial_index = df2.sindex
        df1["bbox"] = df1.geometry.apply(lambda x: x.bounds)
        df1["sidx"] = df1.bbox.apply(lambda x: list(spatial_index.intersection(x)))
        pairs = df1["sidx"].to_dict()
        nei = []
        for i, j in pairs.items():
            for k in j:
                nei.append([i, k])
        pairs = gpd.GeoDataFrame(nei, columns=["idx1", "idx2"], crs=df1.crs)
        pairs = pairs.merge(df1, left_on="idx1", right_index=True)
        pairs = pairs.merge(
            df2, left_on="idx2", right_index=True, suffixes=["_1", "_2"]
        )
        pairs["Intersection"] = pairs.apply(
            lambda x: (x["geometry_1"].intersection(x["geometry_2"])).buffer(0), axis=1
        )
        pairs = gpd.GeoDataFrame(pairs, columns=pairs.columns, crs=df1.crs)
        cols = pairs.columns.tolist()
        cols.remove("geometry_1")
        cols.remove("geometry_2")
        cols.remove("sidx")
        cols.remove("bbox")
        cols.remove("Intersection")
        dfinter = pairs[cols + ["Intersection"]].copy()
        dfinter.rename(columns={"Intersection": "geometry"}, inplace=True)
        dfinter = gpd.GeoDataFrame(dfinter, columns=dfinter.columns, crs=pairs.crs)
        dfinter = dfinter.loc[dfinter.geometry.is_empty == False]
        dfinter.drop(["idx1", "idx2"], inplace=True, axis=1)
        return dfinter
    elif how == "difference":
        spatial_index = df2.sindex
        df1["bbox"] = df1.geometry.apply(lambda x: x.bounds)
        df1["sidx"] = df1.bbox.apply(lambda x: list(spatial_index.intersection(x)))
        df1["new_g"] = df1.apply(
            lambda x: reduce(
                lambda x, y: x.difference(y).buffer(0),
                [x.geometry] + list(df2.iloc[x.sidx].geometry),
            ),
            axis=1,
        )
        df1.geometry = df1.new_g
        df1 = df1.loc[df1.geometry.is_empty == False].copy()
        df1.drop(["bbox", "sidx", "new_g"], axis=1, inplace=True)
        return df1
    elif how == "symmetric_difference":
        df1["idx1"] = df1.index.tolist()
        df2["idx2"] = df2.index.tolist()
        df1["idx2"] = np.nan
        df2["idx1"] = np.nan
        dfsym = df1.merge(df2, on=["idx1", "idx2"], how="outer", suffixes=["_1", "_2"])
        dfsym["geometry"] = dfsym.geometry_1
        dfsym.loc[dfsym.geometry_2.isnull() == False, "geometry"] = dfsym.loc[
            dfsym.geometry_2.isnull() == False, "geometry_2"
        ]
        dfsym.drop(["geometry_1", "geometry_2"], axis=1, inplace=True)
        dfsym = gpd.GeoDataFrame(dfsym, columns=dfsym.columns, crs=df1.crs)
        spatial_index = dfsym.sindex
        dfsym["bbox"] = dfsym.geometry.apply(lambda x: x.bounds)
        dfsym["sidx"] = dfsym.bbox.apply(lambda x: list(spatial_index.intersection(x)))
        dfsym["idx"] = dfsym.index.values
        dfsym.apply(lambda x: x.sidx.remove(x.idx), axis=1)
        dfsym["new_g"] = dfsym.apply(
            lambda x: reduce(
                lambda x, y: x.difference(y).buffer(0),
                [x.geometry] + list(dfsym.iloc[x.sidx].geometry),
            ),
            axis=1,
        )
        dfsym.geometry = dfsym.new_g
        dfsym = dfsym.loc[dfsym.geometry.is_empty == False].copy()
        dfsym.drop(
            ["bbox", "sidx", "idx", "idx1", "idx2", "new_g"], axis=1, inplace=True
        )
        return dfsym
    elif how == "union":
        dfinter = spatial_overlays(df1, df2, how="intersection")
        dfsym = spatial_overlays(df1, df2, how="symmetric_difference")
        dfunion = dfinter.append(dfsym)
        dfunion.reset_index(inplace=True, drop=True)
        return dfunion
    elif how == "identity":
        dfunion = spatial_overlays(df1, df2, how="union")
        cols1 = df1.columns.tolist()
        cols2 = df2.columns.tolist()
        cols1.remove("geometry")
        cols2.remove("geometry")
        cols2 = set(cols2).intersection(set(cols1))
        cols1 = list(set(cols1).difference(set(cols2)))
        cols2 = [col + "_1" for col in cols2]
        dfunion = dfunion[(dfunion[cols1 + cols2].isnull() == False).values]
        return dfunion


def identify_point_in_polygon(
    pt_df,
    poly_vec,
    pt_col="station_id",
    poly_col="fid",
    lat_col="latitude",
    lon_col="longitude",
):
    """Performs spatial join to identify containing polygon IDs for points
    with lat/lon co-ordinates.

    Args:
        pt_df:    Dataframe of point locations. Must include a unique site
                  ID and cols containing lat and lon in WGS84 decimal degrees
        poly_vec: Str or GeoDataFrame. Raw path to polygon vector dataset (.shp,
                  .geojson etc.) or GeoDataFrame object. Will be re-projected
                  to WGS84 geographic co-ordinates
        pt_col:   Str. Name of col with unique point IDs
        poly_col: Str. Name of col with unique polygon IDs
        lat_col:  Str. Name of lat col for points
        lon_col:  Str. Name of lon col for points

    Returns:
        Dataframe with "poly_col" column added specifying polygon ID.
    """
    # Validate user input
    assert isinstance(pt_df, pd.DataFrame), "'pt_df' must be a dataframe."
    assert isinstance(poly_vec, gpd.GeoDataFrame) or isinstance(
        poly_vec, str
    ), "'poly_vec' must be either a geodataframe or a file path."

    # Get just the spatial info and site IDs
    pt_df2 = pt_df[[pt_col, lat_col, lon_col]].copy()

    # Drop any rows without lat/lon from df
    if pt_df2.isnull().values.any():
        print(
            "WARNING: Not all sites have complete co-ordinate information. "
            "These rows will be dropped."
        )
        pt_df2.dropna(how="any", inplace=True)

    # Reset index (otherwise GPD join doesn't work)
    pt_df2.reset_index(inplace=True, drop=True)

    # Create the geometry column from point coordinates
    pt_df2["geometry"] = pt_df2.apply(
        lambda row: Point(row[lon_col], row[lat_col]), axis=1
    )

    # Convert to GeoDataFrame
    pt_df2 = gpd.GeoDataFrame(pt_df2, geometry="geometry")
    del pt_df2[lat_col], pt_df2[lon_col]

    # Set coordinate system as WGS84
    pt_df2 = pt_df2.set_crs("epsg:4326")

    # Load polygons
    if isinstance(poly_vec, gpd.GeoDataFrame):
        gdf = poly_vec.copy()
    else:
        gdf = gpd.GeoDataFrame.from_file(poly_vec)

    # Check polygons
    assert set(gdf.geom_type).issubset(
        set(["Polygon", "MultiPolygon"])
    ), "'poly_vec' must be a polygon dataset."

    # Project to WGS84 dd
    gdf = gdf.to_crs("epsg:4326")

    # Get cols of interest
    gdf = gdf[[poly_col, gdf.geometry.name]]

    # Drop any duplicates
    gdf.drop_duplicates(subset=[poly_col], inplace=True)

    # Spatial join
    join_gdf = gpd.tools.sjoin(pt_df2, gdf, how="left", predicate="within")

    # Join output back to original data table
    gdf = join_gdf[[pt_col, poly_col]]
    res_df = pd.merge(pt_df, gdf, how="left", on=pt_col)

    return res_df


def plot_raster(
    ds_or_path,
    cmap="coolwarm",
    title=None,
    vmin=None,
    vmax=None,
    res="110m",
    fill_sea=True,
    mask_countries=None,
    figsize=None,
    out_path=None,
    out_path_dpi=300,
):
    """Make a map of a raster dataset.

    Args:
        ds_or_path.     Str or xarray dataset. Can be a path to any datasets that can be read using
                        rasterio
        cmap:           Str. Colourmap to use. See
                        https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
        title:          Str. Plot title
        vmin:           Float. Min value for colour scale
        vmax:           Float. Max value for colour scale
        res:            Str. '110m', '50m', '10m'. Resolution (i.e. scale, as in 1:10 m) of vector
                        land and coastline. '110m' is coarse; '10m' is detailed
        fill_sea:       Bool. Whether to colour the sea
        mask_countries: List, Str or None. If None, only borders will be plotted. If 'all' all land
                        will be coloured. If list, should list the country names to be coloured. To
                        get a list of valid country names, use get_natural_earth_country_names()
        fig_size:       Tuple. Figure canvas size in inches (width, height)
        out_path:       Raw str or None. Path to image file (e.g. PNG) for saved map, if desired
        out_path_dpi:   Int. Resolution for saved image if 'out_path' is supplied.

    Returns:
        None. Figured is displayed in notebook and (optionally) saved to file.
    """
    assert isinstance(ds_or_path, str) or isinstance(
        ds_or_path, xr.DataArray
    ), "'ds_or_path' must be a file path or an xarray DataArray."

    if isinstance(ds_or_path, str):
        ds = xr.open_rasterio(ds_or_path)
    else:
        ds = ds_or_path

    # Setup plot
    fig = plt.figure(figsize=figsize)

    # Build area definition
    proj_string = ds.attrs["crs"]

    x_cell_size = ds.attrs["res"][0]
    y_cell_size = ds.attrs["res"][1]

    xmin = ds["x"].values.min() - (x_cell_size / 2)
    xmax = ds["x"].values.max() + (x_cell_size / 2)
    ymin = ds["y"].values.min() - (y_cell_size / 2)
    ymax = ds["y"].values.max() + (y_cell_size / 2)

    # Pyresample doesn't seems to cope well with EPSG 4326
    if proj_string == "+init=epsg:4326":
        crs = ccrs.PlateCarree()
        ax = fig.add_subplot(1, 1, 1, projection=crs)
        ax.set_extent([xmin, xmax, ymin, ymax], crs=crs)

    else:
        area_extent = (xmin, ymin, xmax, ymax)
        area_def = AreaDefinition(
            "area_id",
            "desc",
            "proj_id",
            proj_string,
            ds["x"].size,
            ds["y"].size,
            area_extent,
        )
        crs = area_def.to_cartopy_crs()
        ax = fig.add_subplot(1, 1, 1, projection=crs)

    # Mask no data
    ndv = ds.attrs["nodatavals"][0]
    ds2 = ds.where(ds != ndv)

    # Plot
    ds2.plot(cmap=cmap, vmin=vmin, vmax=vmax, ax=ax)

    # Sea
    if fill_sea:
        sea = cfeature.NaturalEarthFeature(
            category="physical",
            name="ocean",
            scale=res,
            edgecolor="none",
            facecolor=cfeature.COLORS["water"],
            alpha=1,
        )
        ax.add_feature(sea, zorder=4)

    # Land
    if mask_countries == "all":
        land = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_0_countries",
            scale=res,
            edgecolor="black",
            facecolor=cfeature.COLORS["land"],
            alpha=1,
        )
        ax.add_feature(land, zorder=5)

    elif mask_countries == None:
        # Just plot borders
        land = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_0_countries",
            scale=res,
            edgecolor="black",
            facecolor="none",
            alpha=1,
        )
        ax.add_feature(land, zorder=5)

    elif isinstance(mask_countries, list):
        # Get countries
        shp = cartopy.io.shapereader.natural_earth(
            resolution=res,
            category="cultural",
            name="admin_0_countries",
        )
        reader = cartopy.io.shapereader.Reader(shp)
        countries = reader.records()

        # Loop over countries
        for country in countries:
            if country.attributes["NAME"] in mask_countries:
                # Opaque
                ax.add_geometries(
                    [country.geometry],
                    ccrs.PlateCarree(),  # CRS of Natural Earth data
                    facecolor=cfeature.COLORS["land"],
                    edgecolor="black",
                    zorder=5,
                )
            else:
                # Transparent
                ax.add_geometries(
                    [country.geometry],
                    ccrs.PlateCarree(),  # CRS of Natural Earth data
                    facecolor="none",
                    edgecolor="black",
                    zorder=5,
                )
    else:
        raise ValueError(
            "'mask_countries' must be 'all', None or a list of country names."
        )

    ax.set_title(title, fontsize=16)
    # plt.tight_layout()

    # Save
    if out_path:
        plt.savefig(out_path, dpi=out_path_dpi)


def get_natural_earth_country_names():
    """Print a list of country names in the Natural Earth data. Useful in combination with
    nivapy.spatial.plot_raster().

    Args:
        None

    Returns:
        List of country names
    """
    # Get countries
    shp = cartopy.io.shapereader.natural_earth(
        resolution="110m",
        category="cultural",
        name="admin_0_countries",
    )
    reader = cartopy.io.shapereader.Reader(shp)
    countries = reader.records()

    # Loop over countries
    country_list = []
    for country in countries:
        country_list.append(country.attributes["NAME"])

    country_list.sort()

    return country_list


def downsample_raster(ds, scale_factor):
    """Downsample a single-band raster to a lower resulution. First read the raster
    with xarray using

        ds = xr.open_rasterio('path/to/raster')

    For the moment, scale_factor must be an odd integer. For example, if the
    original raster has a cell size of 1 and scale factor is 5, the downsampled
    raster will have a cell size of 5.

    Args:
        ds:           DataArray. Single band xarray data array
        scale_factor: Int. Odd integer

    Returns:
        DataArray. Single band xarray DataArray with larger cell size.
    """
    assert isinstance(scale_factor, int), "'scale_factor' must be an odd integer."

    if scale_factor % 2 == 0:
        # Even
        raise ValueError("'scale_factor' must be an odd integer.")
    else:
        # Downsample
        rebinned = rebin_array(ds.values, 1 / scale_factor)
        if len(rebinned.shape) == 2:
            rebinned = rebinned[np.newaxis, :]
        x = ds["x"][::scale_factor]
        y = ds["y"][::scale_factor]
        band = np.array([1])

        rebinned_ds = xr.DataArray(
            rebinned, coords=[band, y, x], dims=["band", "y", "x"]
        )

        # Add metadata
        rebinned_ds.attrs["crs"] = ds.attrs["crs"]
        rebinned_ds.attrs["is_tiled"] = ds.attrs["is_tiled"]
        rebinned_ds.attrs["nodatavals"] = ds.attrs["nodatavals"]

        trans = list(ds.attrs["transform"])
        trans[0] = trans[0] * scale_factor
        trans[4] = trans[4] * scale_factor
        trans = tuple(trans)
        rebinned_ds.attrs["transform"] = trans

        rebinned_ds.attrs["res"] = tuple([i * scale_factor for i in ds.attrs["res"]])

        return rebinned_ds


def get_features(gdf):
    """Helper function for clip_raster_to_gdf(). Converts 'gdf' to the format required
    by rasterio.mask.

    Args:
        gdf: Geodataframe. Must be of (multi-)polygon geometry type

    Returns:
        List of geometries.
    """
    return [json.loads(gdf.to_json())["features"][0]["geometry"]]


def clip_raster_to_bounding_box(raster_path, out_gtiff, bounding_box):
    """Clip a raster dataset to a bounding box and save the result
       as a new GeoTiff.

    Args:
        raster_path:  Str. Path to input raster dataset
        out_gtiff:    Str. Name and path of GeoTiff to be created. Should have a '.tif'
                      file extension
        bounding_box: Tuple. (xmin, ymin, xmax, ymax) in the same co-ordinate system as
                      'raster_path'

    Returns:
        None. The new raster is saved to the specified location.
    """
    # Read raster
    ras = rasterio.open(raster_path)
    crs = ras.crs.data["init"]

    bbox = box(*bounding_box)
    clip_gdf = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs=crs)

    # Apply mask
    shapes = get_features(clip_gdf)
    out_image, out_transform = rasterio.mask.mask(ras, shapes, crop=True)
    out_meta = ras.meta
    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )

    # Save result
    with rasterio.open(out_gtiff, "w", **out_meta) as dest:
        dest.write(out_image)

    ras.close()


def clip_raster_to_gdf(raster_path, out_gtiff, clip_gdf, bounding_box=False):
    """Clip a raster dataset to a (multi-)polygon geodataframe and save the result
    as a new GeoTiff.

    Args:
        raster_path:  Str. Path to input raster dataset
        out_gtiff:    Str. Name and path of GeoTiff to be created. Should have a '.tif'
                      file extension
        clip_gdf:     Geodataframe. Shape to clip to. Must be of (multi-)polygon geometry
                      type
        bounding_box: Bool. If True, use the bounding box of all features in 'clip_gdf'
                      instead of the features themselves

    Returns:
        None. The new raster is saved to the specified location.
    """
    assert isinstance(
        clip_gdf, gpd.GeoDataFrame
    ), "'clip' must be a (Multi-)Polygon geodataframe."

    clip_gdf = clip_gdf.copy()

    # Read raster
    ras = rasterio.open(raster_path)

    # Reproject if necessary
    if clip_gdf.crs != ras.crs.data:
        print("WARNING: The projection of 'clip' differs from the target dataset.")
        print("Converting to the projection of target dataset (%s)" % ras.crs.data)
        clip_gdf.to_crs(crs=ras.crs.data, inplace=True)

    if bounding_box:
        xmin, ymin, xmax, ymax = clip_gdf.bounds.values[0]
        bbox = box(xmin, ymin, xmax, ymax)
        clip_gdf = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs=clip_gdf.crs)

    # Apply mask
    shapes = get_features(clip_gdf)
    out_image, out_transform = rasterio.mask.mask(ras, shapes, crop=True)
    out_meta = ras.meta
    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )

    # Save result
    with rasterio.open(out_gtiff, "w", **out_meta) as dest:
        dest.write(out_image)

    ras.close()


def catchment_boundary_quickmap(stn_code, cat_gdf, title=None, out_png=None):
    """Create a simple map showing a catchment boundary and outflow location for a single
    catchment. If the exact outflow location is not known, the original monitoring location
    will be shown instead. Data are overlaid on a standard Kartverket basemap.

    Options for customisation are limited - this function is designed for quick visualisation
    & checking rather than producing "final" maps.

    Args:
        stn_code: Str. Station code for the site of interest in 'cat_gdf'
        cat_gdf:  Geodataframe. As returned by select_jhub_project_catchments
        title:    Str. Title for plot, if desired
        out_png:  Raw str. Path for PNG to save, if desired

    Returns:
        Map image. If desired, the image is also saved as a PNG.
    """
    # Get UTM co-ords
    cat_gdf = cat_gdf.copy()
    cat_gdf = wgs84_dd_to_utm(cat_gdf)

    # Get station and co-ords of interest
    cat = cat_gdf.query("station_code == @stn_code")
    zone = cat["utm_zone"].iloc[0]
    north = cat["utm_north"].iloc[0]
    east = cat["utm_east"].iloc[0]

    # Plot
    fig = plt.figure(figsize=(10, 10))
    crs = ccrs.UTM(zone)
    ax = fig.add_subplot(111, projection=crs)

    ax.add_wms(wms="https://openwms.statkart.no/skwms1/wms.topo4", layers=["topo4_WMS"])

    cat.to_crs("epsg:32633").plot(ax=ax, facecolor="none", edgecolor="k", lw=2)

    ax.scatter(east, north, s=100, c="r", edgecolors="k", transform=crs)

    if title:
        ax.set_title(title, fontsize=16)

    if out_png:
        plt.savefig(out_png, dpi=300)


def derive_watershed_boundaries(
    df,
    id_col="station_code",
    xcol="longitude",
    ycol="latitude",
    crs="epsg:4326",
    min_size_km2=None,
    dem_res_m=40,
    buffer_km=None,
    temp_fold=None,
    reproject=True,
):
    """Calculate watershed boundaries in Norway based on outflow co-ordinates provided
    as a dataframe.

    Args
        df:           Dataframe. Containing co-ordinates for catchment outflow points
        id_col:       Str. Name of column in 'df' containing a unique ID for each
                      outflow point. This will be used to link derived catchments
                      to the original points. Must be unique
        xcol:         Str. Name of column in 'df' containing 'eastings' (i.e. x or
                      longitude)
        ycol:         Str. Name of column in 'df' containing 'northings' (i.e. y or
                      latitude)
        crs:          Str. A valid co-ordinate reference system for Geopandas. Most
                      easily expressed using EPSG codes e.g. 'epsg:4326' for WGS84
                      lat/lon, or 'epsg:25833' for ETRS89 UTM zone 33N etc. See
                          https://epsg.io/
        min_size_km2: Int, Float or None. Default None. If None, the catchment is derived
                      upstream of the exact cell containing the specified outflow point.
                      If the provided co-ordinates do not exactly match the stream
                      location, this may result in small/incorrect catchments being
                      delineated. Setting 'min_size_km2' will snap the outflow point to
                      the nearest cell with an upstream catchment area of at least this
                      many square kilometres. It is usually a good idea to explicitly set
                      this parameter
        dem_res_m:    Int. Default 40. Resolution of elevation model to use. One of
                      [10, 20, 40]. Smaller values give better cacthments but take
                      longer to process
        buffer_km:    Int, Float or None. Default None. If None, the code will search
                      the entire vassdragsområde. This is a good default, but it can be
                      slow and memory-intensive. Setting a value for this parameter will
                      first subset the DEM to a square region centred on the outflow point
                      with a side length of (2*buffer_km) kilometres. E.g. if you know your
                      catchments do not extend more than 20 km in any direction from the
                      specified outflow points, set 'buffer_km=20'. This will significantly
                      improve performance
        temp_fold:    Str. Default None. Only used if 'buffer_km' is specified. Must be a
                      path to a folder on 'shared'. Will be used to store temporary files
                      (which are deleted at the end)
        reproject:    Bool. Default True. Whether to reproject the derived catchments
                      back to the original 'crs' that the outflow points were provided
                      in. If False, catchments are returned in the CRS of the underlying
                      DEM, which is ETRS89-based UTM zone 33N (EPSG 25833)

    Returns
        GeoDataframe of catchments.
    """
    # Import here to avoid building Cython every time nivapy gets imported
    from pysheds.grid import Grid

    method = "pysheds"

    # Check user input
    assert len(df[id_col].unique()) == len(df), "ERROR: 'id_col' is not unique."
    assert dem_res_m in [
        10,
        20,
        40,
    ], "ERROR: 'dem_res_m' must be one of [10, 20, 40]."
    if min_size_km2:
        assert isinstance(
            min_size_km2, (float, int)
        ), "'min_size_km2' must be an integer, float or None."
    if buffer_km:
        assert isinstance(
            buffer_km, (int, float)
        ), "'buffer_km' must be an integer, float or None."

        assert isinstance(
            temp_fold, (str)
        ), "'temp_fold' is required when 'buffer_km' is specified."

        shared_path = Path("/home/jovyan/shared/common")
        child_path = Path(temp_fold)
        assert (
            shared_path in child_path.parents
        ), "'temp_fold' must be a folder on the 'shared/common' drive."

    if temp_fold:
        assert buffer_km, "'buffer_km' is required when 'temp_fold' is specified."
        work_dir = os.path.join(temp_fold, "cat_delin_temp")
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

    # Build geodataframe and reproject to CRS of DEM
    dem_crs = "epsg:25833"
    gdf = gpd.GeoDataFrame(
        df.copy(), geometry=gpd.points_from_xy(df[xcol], df[ycol], crs=crs)
    )
    gdf = gdf.to_crs(dem_crs)
    gdf["x_proj"] = gdf["geometry"].x
    gdf["y_proj"] = gdf["geometry"].y

    # Get vassdragsområder
    eng = da.connect_postgis()
    vass_gdf = da.read_postgis("physical", "norway_nve_vassdragomrade_poly", eng)
    vass_gdf = vass_gdf.to_crs(dem_crs)

    # Assign points to vassdragsområder
    gdf = gpd.sjoin(
        gdf, vass_gdf[["vassdragsomradenr", "geom"]], predicate="intersects", how="left"
    )
    n_cats = len(gdf)
    gdf.dropna(subset=["vassdragsomradenr"], inplace=True)
    if len(gdf) != n_cats:
        msg = "Some outlet locations could not be assigned to a vassdragsområde. These will be skipped."
        warnings.warn(msg)

    # Loop over vassdragsområder
    cat_ids = []
    cat_geoms = []
    for vassom in tqdm(
        sorted(gdf["vassdragsomradenr"].unique()), desc="Looping over vassdragsområder"
    ):
        if method == "pysheds":
            dirmap = (1, 2, 3, 4, 5, 6, 7, 8)
            vassom_fdir_path = (
                f"/home/jovyan/shared/common/01_datasets/spatial/dtm_merged_utm33/dtm_{dem_res_m}m"
                f"/by_vassom/flow_direction/vassom_{vassom}_{dem_res_m}m_fdir.tif"
            )
            vassom_facc_path = (
                f"/home/jovyan/shared/common/01_datasets/spatial/dtm_merged_utm33/dtm_{dem_res_m}m"
                f"/by_vassom/flow_accumulation/vassom_{vassom}_{dem_res_m}m_facc.tif"
            )
        elif method == "wbt":
            dirmap = (128, 1, 2, 4, 8, 16, 32, 64)
            vassom_fdir_path = (
                f"/home/jovyan/shared/common/01_datasets/spatial/dtm_merged_utm33/dtm_{dem_res_m}m"
                f"/by_vassom/wbt_fdir/vassom_{vassom}_{dem_res_m}m_fdir.tif"
            )
            vassom_facc_path = (
                f"/home/jovyan/shared/common/01_datasets/spatial/dtm_merged_utm33/dtm_{dem_res_m}m"
                f"/by_vassom/wbt_facc/vassom_{vassom}_{dem_res_m}m_facc.tif"
            )
        else:
            raise ValueError("Method not valid.")

        if buffer_km is None:
            # Read the full grids in outer loop
            fdir_grid = Grid.from_raster(vassom_fdir_path)
            fdir = fdir_grid.read_raster(vassom_fdir_path)
            facc_grid = Grid.from_raster(vassom_facc_path)
            facc = facc_grid.read_raster(vassom_facc_path)

        # Loop over points in each vassdragsområde
        pts_vass_gdf = gdf.query("vassdragsomradenr == @vassom")
        for idx, row in tqdm(
            pts_vass_gdf.iterrows(),
            total=pts_vass_gdf.shape[0],
            desc=f"Looping over outlets in vassdragsområder {vassom}",
            # leave=False,
        ):
            cat_id = row[id_col]
            x = row["x_proj"]
            y = row["y_proj"]

            # Subset rasters for this point, if desired
            if buffer_km:
                xmin = x - (buffer_km * 1000)
                xmax = x + (buffer_km * 1000)
                ymin = y - (buffer_km * 1000)
                ymax = y + (buffer_km * 1000)
                bbox = (xmin, ymin, xmax, ymax)

                fdir_temp = os.path.join(work_dir, "fdir.tif")
                clip_raster_to_bounding_box(vassom_fdir_path, fdir_temp, bbox)

                facc_temp = os.path.join(work_dir, "facc.tif")
                clip_raster_to_bounding_box(vassom_facc_path, facc_temp, bbox)

                fdir_grid = Grid.from_raster(fdir_temp)
                fdir = fdir_grid.read_raster(fdir_temp)
                facc_grid = Grid.from_raster(facc_temp)
                facc = facc_grid.read_raster(facc_temp)

            # Snap outflow if desired
            if min_size_km2:
                # Convert min area to number of pixels
                acc_thresh = int((min_size_km2 * 1e6) / (dem_res_m**2))
                x_snap, y_snap = facc_grid.snap_to_mask(facc > acc_thresh, (x, y))
            else:
                x_snap, y_snap = x, y

            # Delineate catchment
            fdir = fdir.astype(np.int32)
            fdir.nodata = int(fdir.nodata)
            catch = fdir_grid.catchment(
                x=x_snap,
                y=y_snap,
                fdir=fdir,
                dirmap=dirmap,
                xytype="coordinate",
            )

            # Create a vector representation of the catchment mask
            catch_view = fdir_grid.view(catch, dtype=np.uint8)
            shapes = fdir_grid.polygonize(catch_view)
            for shapedict, value in shapes:
                # 'value' is 1 for the catchment and 0 for "not the catchment"
                if value == 1:
                    cat_ids.append(cat_id)
                    cat_geoms.append(shape(shapedict))

    if temp_fold:
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

    res_gdf = gpd.GeoDataFrame({id_col: cat_ids, "geometry": cat_geoms}, crs=dem_crs)
    res_gdf = res_gdf.merge(df, on=id_col)
    res_gdf = res_gdf[list(df.columns) + ["geometry"]]

    res_gdf.geometry = res_gdf.geometry.apply(lambda p: remove_polygon_holes(p))
    res_gdf = res_gdf.dissolve(by=id_col).reset_index()

    if reproject:
        res_gdf = res_gdf.to_crs(crs)

    return res_gdf


def remove_polygon_holes(poly):
    """Delete polygon holes by limitation to the exterior ring.

    https://stackoverflow.com/a/61466689/505698

    Args
        poly: Input shapely Polygon

    Returns
        Polygon
    """
    if poly.interiors:
        return Polygon(list(poly.exterior.coords))
    else:
        return poly


def get_elvis_streams_as_shapes(crs="epsg:25833"):
    """Get the ELVIS river network from PostGIS and convert to a format
    suitable for rasterisation.

    Args
        crs: Str. Valid CRS string for geopandas

    Returns
        Tuple of tuples. Geometries for rasterising
    """
    eng = da.connect_postgis()
    riv_gdf = da.read_postgis("physical", "norway_nve_elvis_river_network_line", eng)
    riv_gdf = riv_gdf.to_crs(crs)
    shapes = ((geom, 1) for geom in riv_gdf.geometry)

    return shapes


def pysheds_array_to_raster(data, base_tiff, out_tiff, dtype, nodata):
    """Pysheds doesn't seem to write array metadata correctly (especially the CRS
    information). This function takes result arrays from pysheds, but saves them
    using rasterio.

    Args
        data:      Array-like. Data to save
        base_tiff: Str. Path to TIFF with same properties/profile as desired output
        out_tiff:  Str. Path to TIFF to be created
        dtype:     Obj. Dtype for output array
        nodata:    Int. No data value for output array

    Returns
        None. Array is saved to disk.
    """
    with rasterio.open(base_tiff) as src:
        out_meta = src.meta.copy()
    data = data.astype(dtype)
    out_meta.update(
        {
            "driver": "GTiff",
            "dtype": dtype,
            "compress": "lzw",
            "nodata": nodata,
            "BIGTIFF": "IF_SAFER",
        }
    )
    with rasterio.open(out_tiff, "w", **out_meta) as dest:
        dest.write(data, 1)


def burn_stream_shapes(grid, dem, shapes, dz, sigma=None):
    """Burn streams represented by 'shapes' into 'dem'. The burn depth
    is 'dz' and Gaussian blur is applied with std. dev. 'sigma'. 'grid'
    is used to provide the transform etc. for the output raster.

    Args
        grid:   Obj. Pysheds grid object
        dem:    Obj. Pysheds raster object
        shapes: Tuple of obj. Stream shapes
        dz:     Float or int. Burn depth
        sigma:  Float or None. Default None. Std. dev. for (optional)
                Gaussian blur

    Returns
        Pysheds Raster object (essentially a numpy array)
    """
    if sigma:
        assert isinstance(
            sigma, (int, float)
        ), "'sigma' must be of type 'float' or 'int'."

    stream_raster = features.rasterize(
        shapes,
        out_shape=grid.shape,
        transform=grid.affine,
        all_touched=False,
    )
    stream_raster = skeletonize(stream_raster).astype(np.uint8)
    mask = stream_raster.astype(bool)

    if sigma:
        # Blur mask using a gaussian filter
        blurred_mask = ndimage.filters.gaussian_filter(
            mask.astype(np.float32), sigma=sigma
        )

        # Set central river channel to max of Gaussian to prevent pits, then normalise
        # s.t. max blur = 1 (and decays to 0 in Gaussian fashion either side)
        blur_max = blurred_mask.max()
        # blurred_mask[mask.astype(bool)] = blur_max
        blurred_mask = blurred_mask / blur_max
        mask = blurred_mask

    # Burn streams
    dem[(mask > 0)] = dem[(mask > 0)] - (dz * mask[(mask > 0)])

    return dem


def burn_lake_shapes(grid, dem, shapes, dz):
    """Burn streams represented by 'shapes' into 'dem'. The burn depth
    is 'dz'.

    'grid' is used to provide the transform etc. for the output raster.

    Args
        grid:   Obj. Pysheds grid object
        dem:    Obj. Pysheds raster object
        shapes: Tuple of obj. Lake shapes
        dz:     Float or int. Burn depth

    Returns
        Pysheds Raster object (essentially a numpy array)
    """
    lake_raster = features.rasterize(
        shapes,
        out_shape=grid.shape,
        transform=grid.affine,
        all_touched=False,
    )
    mask = lake_raster.astype(bool)
    dem[(mask > 0)] = dem[(mask > 0)] - (dz * mask[(mask > 0)])

    return dem


def condition_dem(
    raw_dem_path,
    fill_dem_path,
    fdir_path,
    facc_path,
    dem_dtype=np.int16,
    dem_ndv=-32767,
    burn=False,
    stream_shapes=None,
    lake_shapes=None,
    stream_sigma=None,
    stream_dz=None,
    lake_dz=None,
    max_iter=1e9,
    eps=1e-12,
):
    """Burns streams into DEM, fills pits and depressions, and calculates flow
    direction and accumulation. The filled DEM is converted to the specificed
    dtype and saved. Flow direction and accumulation rasters are also saved.

    Args
        raw_dem_path:  Str. Path to raw DEM to be processed
        fill_dem_path: Str. Output path for filled (and optionally burned) DEM
        fdir_path:     Str. Output path for flow direction
        facc_path:     Str. Output path for flow accumulation
        dem_dtype:     Obj. Numpy data type for output DEM
        dem_ndv:       Float or int. NoData value for output DEM
        burn:          Bool. Whether to burn streams
        stream_shapes: Tuple of tuples. Stream shapes to burn. Only valid if
                       'burn' is True
        lake_shapes:   Tuple of tuples. Lake shapes to burn. Only valid if
                       'burn' is True
        stream_sigma:  Float. Std. dev. for Gaussian blur applied to streams.
                       Only valid if 'burn' is True
        stream_dz:     Float or int. Stream burn depth. Only valid if 'burn' is
                       True
        lake_dz:       Float or int. Lake burn depth. Only valid if 'burn' is
                       True
        max_iter:      Int. Default 1e9. Maximum iterations for filling flats
        eps:           Float. Default 1e-12. Parameter for flat-filling
                       algorithm. See example notebook

    Returns
        Tuple of PySheds 'Raster' arrays (dem, fdir, facc).
    """
    # Import here to avoid building Cython every time nivapy gets imported
    from pysheds.grid import Grid

    # Check user input
    if burn:
        assert None not in (
            stream_shapes,
            lake_shapes,
            stream_dz,
            lake_dz,
        ), "'stream_shapes', 'lake_shapes', 'stream_dz' and 'lake_dz' are all required when 'burn' is True."
    else:
        assert all(
            v is None
            for v in (stream_shapes, lake_shapes, stream_sigma, stream_dz, lake_dz)
        ), "'stream_shapes', 'lake_shapes', 'stream_sigma', 'stream_dz' and 'lake_dz' are not required when 'burn' is False."

    dirmap = (1, 2, 3, 4, 5, 6, 7, 8)
    grid = Grid.from_raster(raw_dem_path)
    dem = grid.read_raster(raw_dem_path).astype(np.float32)
    ndv = dem.nodata.copy()

    # 'fill_depressions' isn't designed for NoData (see
    #     https://github.com/scikit-image/scikit-image/issues/4078)
    # Either set NoData and values < 0 to zero or -dz, depending on
    # whether streams are being burned. This forces all cells to drain
    # to the edge of the grid
    mask = dem == ndv
    if burn:
        dem[mask] = -(stream_dz + lake_dz)
        dem[dem < 0] = -(stream_dz + lake_dz)
    else:
        dem[mask] = 0
        dem[dem < 0] = 0

    if burn:
        dem = burn_lake_shapes(grid, dem, lake_shapes, lake_dz)
        dem = burn_stream_shapes(grid, dem, stream_shapes, stream_dz, stream_sigma)
    dem = grid.fill_pits(dem, nodata_in=np.nan, nodata_out=np.nan)
    dem = grid.fill_depressions(dem, nodata_in=np.nan, nodata_out=np.nan)
    dem = grid.resolve_flats(
        dem, max_iter=max_iter, eps=eps, nodata_in=np.nan, nodata_out=np.nan
    )

    npits = grid.detect_pits(dem).sum()
    nflats = grid.detect_flats(dem).sum()
    if (npits > 0) or (nflats > 0):
        fill_fname = os.path.split(fill_dem_path)[1]
        msg = f"        {fill_fname} has {npits} pits and {nflats} flats."
        print(msg)
        logging.info(msg)

    # Flow dir and accum
    fdir = grid.flowdir(
        dem, routing="d8", dirmap=dirmap, nodata_in=np.nan, nodata_out=0
    )
    facc = grid.accumulation(
        fdir, routing="d8", dirmap=dirmap, nodata_in=0, nodata_out=0
    )

    # Save results to disk
    if np.issubdtype(dem_dtype, np.integer):
        dem = np.rint(dem)
    dem[np.isnan(dem)] = dem_ndv
    dem = dem.astype(dem_dtype)
    pysheds_array_to_raster(dem, raw_dem_path, fill_dem_path, dem_dtype, dem_ndv)

    fdir = fdir.astype(np.int16)
    pysheds_array_to_raster(fdir, raw_dem_path, fdir_path, np.int16, 0)

    facc = facc.astype(np.uint32)
    pysheds_array_to_raster(facc, raw_dem_path, facc_path, np.uint32, 0)

    return (dem, fdir, facc)


def get_land_cover_for_polygons(
    poly_gdf,
    id_col="station_code",
    lc_data="corine_2018",
    full_extent=False,
    reproject=True,
):
    """Calculate land cover proportions for polygons within in Norway (supplied as a
    geodataframe). Available land cover datasets are AR50 or Corine (2000, 2006, 2012
    and 2018). All datasets were originally downloaded from NIBIO.

    Args:
        poly_gdf:    Polygon geodataframe. Must include a column named 'geometry'
                     containing polygon or multipolygon data types
        id_col:      Str. Name of column in 'poly_gdf' containing a unique ID for each
                     region/(multi)polygon of interest
        lc_data:     Str. Default 'corine_2018'. Land cover dataset to use. Must be one of
                     ['ar50', 'corine_2000', 'corine_2006', 'corine_2012', 'corine_2018'].
                     For further details on each dataset, see the links here:
                     https://github.com/NIVANorge/niva_jupyter_hub/blob/master/postgis_db/postgis_db_dataset_summary.md
        full_extent: Bool. Default False. Used to tune performance. If True, the function
                     will read the entire land cover dataset for the full bounding box of
                     'poly_gdf' in a single operation before performing the geometirc
                     intersection. This is a good strategy if your polgyons are close
                     together i.e. if the total area of your polygons is a large
                     proportion of the area of the bounding box of 'poly_gdf'. If False
                     (the default), the function will loop over polygons in 'poly_gdf'
                     and perform each intersection separately. This involves more
                     geometric operations, but will read much less data in total for
                     cases where the polygons are well-spaced (i.e. where the polygons
                     occupy a small proportion of the total area of the bounding box of
                     'poly_gdf'.
        reproject:   Bool. Default True. Whether to reproject the data to the original CRS
                     of 'poly_gdf'. If False, data will be returned in the CRS of 'lc_data'
                     (UTM Zone 33N)

    Returns:
        Geodataframe. Contains the the total area of each original polygon, plus the area of
        each land cover class in each polygon. Also includes a column showing the percentage
        land cover for each class. All area calculations are performed using a Cylindrical
        Equal Area projection (https://proj.org/operations/projections/cea.html).
    """
    # Validate user input
    assert len(poly_gdf[id_col].unique()) == len(
        poly_gdf
    ), "ERROR: 'id_col' is not unique."

    lc_data = lc_data.lower()
    assert lc_data in [
        "ar50",
        "corine_2000",
        "corine_2006",
        "corine_2012",
        "corine_2018",
    ], "'lc_data' must be one of ['ar50', 'corine_2000', 'corine_2006', 'corine_2012', 'corine_2018']."

    for col in ["class_area_km2", f"{id_col}_area_km2"]:
        assert (
            col not in poly_gdf.columns
        ), f"'poly_gdf' already contains column '{col}'. Please rename this and try again."

    orig_crs = poly_gdf.crs
    orig_cols = [col for col in poly_gdf.columns if col != "geometry"]

    # Calculate areas for whole polygons (before intersection)
    poly_gdf = poly_gdf.copy()
    poly_gdf = poly_gdf.dissolve(by=id_col, as_index=False)
    poly_gdf[f"{id_col}_area_km2"] = (
        poly_gdf.to_crs({"proj": "cea"}).geometry.area / 1e6
    )

    dataset_dict = {
        "ar50": "norway_nibio_ar50_poly",
        "corine_2000": "norway_nibio_corine_2000_poly",
        "corine_2006": "norway_nibio_corine_2006_poly",
        "corine_2012": "norway_nibio_corine_2012_poly",
        "corine_2018": "norway_nibio_corine_2018_poly",
    }

    if lc_data == "ar50":
        dissolve_fields = ["artype", "description", id_col]
        lc_crs = "epsg:25833"
    else:
        dissolve_fields = ["klasse", "norsk", "english", id_col]
        lc_crs = "epsg:32633"

    eng = da.connect_postgis()
    poly_gdf = poly_gdf.to_crs(lc_crs)
    if full_extent:
        # Perform intersection in one go
        lc_gdf = da.read_postgis(
            "physical",
            dataset_dict[lc_data],
            eng,
            clip=poly_gdf,
        )
        lc_gdf = gpd.overlay(lc_gdf, poly_gdf, how="intersection")
        lc_gdf = lc_gdf.dissolve(by=dissolve_fields, as_index=False)
        lc_gdf["class_area_km2"] = lc_gdf.to_crs({"proj": "cea"}).geometry.area / 1e6
        lc_gdf["land_cover_pct"] = (
            100 * lc_gdf["class_area_km2"] / lc_gdf[f"{id_col}_area_km2"]
        )
        lc_gdf = lc_gdf.sort_values(id_col).reset_index()
    else:
        # Loop over polys
        gdf_list = []
        for site_id in poly_gdf[id_col].unique():
            gdf = poly_gdf.query(f"{id_col} == @site_id")
            lc_gdf = da.read_postgis(
                "physical",
                dataset_dict[lc_data],
                eng,
                clip=gdf,
            )
            lc_gdf = gpd.overlay(lc_gdf, gdf, how="intersection")
            gdf_list.append(lc_gdf)

        lc_gdf = pd.concat(gdf_list, axis="rows")
        lc_gdf = lc_gdf.dissolve(by=dissolve_fields, as_index=False)
        lc_gdf["class_area_km2"] = lc_gdf.to_crs({"proj": "cea"}).geometry.area / 1e6
        lc_gdf["land_cover_pct"] = (
            100 * lc_gdf["class_area_km2"] / lc_gdf[f"{id_col}_area_km2"]
        )
        lc_gdf = lc_gdf.sort_values(id_col).reset_index()

    if reproject:
        lc_gdf = lc_gdf.to_crs(orig_crs)

    if lc_data == "ar50":
        cols = orig_cols + [
            "artype",
            "ardyrking",
            "arjordbr",
            "arkartstd",
            "arskogbon",
            "artreslag",
            "arveget",
            "description",
            f"{id_col}_area_km2",
            "class_area_km2",
            "land_cover_pct",
            "geometry",
        ]
        lc_gdf = lc_gdf[cols]
    else:
        cols = orig_cols + [
            "klasse",
            "kode",
            "norsk",
            "english",
            f"{id_col}_area_km2",
            "class_area_km2",
            "land_cover_pct",
            "geometry",
        ]
        lc_gdf = lc_gdf[cols]

    return lc_gdf
