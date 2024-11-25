import binascii
import configparser
import datetime as dt
import getpass
import json
import os
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cdsapi
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shapely
import sqlalchemy
import xarray as xr
from osgeo import gdal, osr
from pandas import json_normalize
from sqlalchemy import create_engine
from tqdm.notebook import tqdm

from . import spatial


def connect(src="nivabase", db_name=None, port=5432):
    """Connects either to the NIVABASE or to a local Postgres/GIS database on the
       Docker host. If src='postgres', the database can be either a local
       installation of Postgres/GIS, or a Postgres/GIS instance running in another
       container. Connecting to NIVABASE is only possible from inside of NIVA's
       network i.e. not from JupyterHub.

        NOTE: If src='nivabase', 'db_name' and 'port' will be ignored.

    Args:
        src:      Str. Either 'nivabase' or 'postgres'
        db_name:  Str. Name of database to connect to. Only used if src='postgres'
        port:     Int. Port for connecting to local Postgres/GIS instance. Only used
                  if src='postgres'

    Returns:
        SQLAlchemy database engine.
    """
    # Deal with encodings
    os.environ["NLS_LANG"] = ".AL32UTF8"

    # Validate user input
    assert src in (
        "nivabase",
        "postgres",
    ), "src must be either 'nivabase' or 'postgres'."

    if (src == "nivabase") and ((db_name is not None) or (port != 5432)):
        print(
            "WARNING: 'port' and 'db_name' are only used when src='postgres'.\n"
            "These arguments will be ignored."
        )

    if (src == "postgres") and (db_name is None):
        print("ERROR: You must specify 'db_name' when src='postgres'.")

        return

    # Get credentials
    user = getpass.getpass(prompt="Username: ")
    pw = getpass.getpass(prompt="Password: ")

    if src == "nivabase":
        conn_str = (
            f"oracle+cx_oracle://{user}:{pw}@DBORA-NIVA-PROD01.NIVA.CORP:1555/NIVABPRD"
        )
    else:
        conn_str = (
            f"postgresql+psycopg2://{user}:{pw}@host.docker.internal:{port}/{db_name}"
        )

    # Create and test connection
    try:
        engine = create_engine(conn_str)
        with engine.connect() as connection:
            print("Connection successful.")
        return engine

    except Exception as e:
        print("Connection failed.")
        print(e)


def connect_postgis(
    admin=False,
    user="jovyan",
    password="joyvan_ro_pw",
    host="postgis",
    port=5432,
    database="general",
):
    """Connects to a database on the DSToolkit's PostGIS instance.

    Args:
        admin:    Bool. Whether to connect with administrative privileges (for writing/
                  modifying datasets. By default, this function connects with a generic
                  "read-only" account that does not require authentication. If True, you
                  must supply a valid administrator username and password
        user:     Str. Username for administrator, if required
        password: Str. Password for administrator, if required
        host:     Str. Host for the database (only relevant if attempting to connect to a
                  different Postgres instance
        port:     Int. The port to connect to on the host
        database: Str. Name of database to connect to. The default, 'general', contains a
                  ranger of generally useful datasets. Other databases are project-specific

    Returns:
        SQLAlchemy database engine.
    """
    # Get credentials if necessary
    if admin:
        user = getpass.getpass(prompt="Username: ")
        password = getpass.getpass(prompt="Password: ")

    # Build connection string
    conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

    # Create and test connection
    try:
        engine = create_engine(conn_str)
        with engine.connect() as connection:
            print("Connection successful.")

        return engine

    except Exception as e:
        print("Connection failed.")
        print(e)


def select_resa_projects(engine):
    """Get full list of projects from RESA2.

    Direct connection to Nivabasen only.

    Args:
        engine: Obj. Active "engine" object

    Returns:
        Dataframe.
    """
    sql = (
        "SELECT project_id, "
        "  project_number, "
        "  project_name, "
        "  contact_person, "
        "  project_description "
        "FROM resa2.projects "
        "ORDER BY project_id"
    )
    with engine.connect() as connection:
        df = pd.read_sql(sql, connection)

    print("%s projects in the RESA database." % len(df))

    return df


def select_ndb_projects(engine):
    """Get full list of projects from the NDB.

    Direct connection to Nivabasen only.

    Args:
        engine: Obj. Active NDB "engine" object

    Returns:
        Dataframe
    """
    sql = (
        "SELECT project_id, "
        "  project_name, "
        "  project_description "
        "FROM nivadatabase.projects "
        "ORDER BY project_id"
    )
    with engine.connect() as connection:
        df = pd.read_sql(sql, connection)

    print("%s projects in the NIVADATABASE." % len(df))

    return df


def select_resa_stations(engine):
    """Get full list of stations from RESA2.

    Direct connection to Nivabasen only.

    Args:
        engine: Obj. Active "engine" object

    Returns:
        QGrid selection widget
    """
    sql = (
        "SELECT station_id, "
        "  station_code, "
        "  station_name, "
        "  latitude, "
        "  longitude, "
        "  altitude "
        "FROM resa2.stations "
        "ORDER BY station_id"
    )
    with engine.connect() as connection:
        df = pd.read_sql(sql, connection)

    print("%s stations in the RESA database." % len(df))

    return df


def select_ndb_stations(engine):
    """Get full list of stations from the NDB.

    Direct connection to Nivabasen only.

    Note: The NIVADATABASE allows multiple names for the same station.
    This function returns all unique id-code-name-type-lat-lon
    combinations, which will include duplicated station IDs in some
    cases.

    Args:
        engine: Obj. Active NDB "engine" object

    Returns:
        Dataframe
    """
    # Query db
    sql = (
        "SELECT DISTINCT a.station_id, "
        "  a.station_code, "
        "  a.station_name, "
        "  c.station_type, "
        "  d.latitude, "
        "  d.longitude "
        "FROM nivadatabase.projects_stations a, "
        "  nivadatabase.stations b, "
        "  nivadatabase.station_types c, "
        "  niva_geometry.sample_points d "
        "WHERE a.station_id    = b.station_id "
        "AND b.station_type_id = c.station_type_id "
        "AND b.geom_ref_id     = d.sample_point_id "
        "ORDER BY a.station_id"
    )
    with engine.connect() as connection:
        df = pd.read_sql(sql, connection)

    print("%s stations in the NIVADATABASE." % len(df))

    return df


def select_resa_project_stations(project_list, engine):
    """Get stations associated with projects.

    Direct connection to Nivabasen only.

    Args:
        project_list: Array-like or project dataframe. List of project IDs (Ints)
        engine:       Obj. Active "engine" object returned by
                      nivapy.da.connect

    Returns:
        Dataframe.
    """
    # If a project data frame is passed rather than a list of IDs, get IDs
    if isinstance(project_list, pd.DataFrame):
        project_list = project_list["project_id"]

    # Build query
    bind_pars = ",".join("%d" % i for i in project_list)

    sql = (
        "SELECT station_id, station_code, station_name, "
        "latitude, longitude, altitude FROM resa2.stations "
        "WHERE station_id IN ( "
        "  SELECT station_id FROM resa2.projects_stations "
        "  WHERE project_id IN (%s))" % bind_pars
    )
    with engine.connect() as connection:
        df = pd.read_sql(sql, con=connection)

    return df


def select_ndb_project_stations(proj_df, engine, drop_dups=False):
    """Get stations asscoiated with selected projects.

    Args:
        proj_df:   List or Dataframe. If dataframe, must have a column named
                   'project_id' with the project IDs of interest
        engine:    Obj. Active NDB "engine" object
        drop_dups: Bool. The same station may have different names in different
                   projects. If some of the selected projects include the
                   same station, this will result in duplicates in the
                   stations table (i.e. same station ID, but multiple names).
                   By default, the duplicates will be returned. Setting
                   'drop_dups=True' will select one set of names per station
                   ID and return a dataframe with no duplicates (but the
                   station codes and names may not be what you're expecting)

    Returns:
        Dataframe
    """
    # Get proj IDs
    assert len(proj_df) > 0, "ERROR: Please select at least one project."
    if isinstance(proj_df, pd.DataFrame):
        proj_df["project_id"].drop_duplicates(inplace=True)
        proj_ids = proj_df["project_id"].values.astype(int).tolist()
    else:  # Should be a list
        proj_ids = list(set(proj_df))

    # Query db
    bind_pars = ",".join("%d" % i for i in proj_ids)

    sql = (
        "SELECT DISTINCT a.station_id, "
        "  a.station_code, "
        "  a.station_name, "
        "  c.station_type, "
        "  d.longitude, "
        "  d.latitude "
        "FROM nivadatabase.projects_stations a, "
        "  nivadatabase.stations b, "
        "  nivadatabase.station_types c, "
        "  niva_geometry.sample_points d "
        "WHERE a.station_id IN "
        "  (SELECT station_id "
        "  FROM nivadatabase.projects_stations "
        "  WHERE project_id IN (%s) "
        "  ) "
        "AND a.station_id      = b.station_id "
        "AND b.station_type_id = c.station_type_id "
        "AND b.geom_ref_id     = d.sample_point_id "
        "ORDER BY a.station_id" % bind_pars
    )
    with engine.connect() as connection:
        df = pd.read_sql(sql, connection)

    # Drop duplictaes, if desired
    if drop_dups:
        df.drop_duplicates(subset="station_id", inplace=True)
        df.reset_index(inplace=True, drop=True)

    return df


def select_resa_station_parameters(stn_df, st_dt, end_dt, engine):
    """Gets the list of available water chemistry parameters for the
    selected stations in RESA.

    Args:
        stn_df: List or Dataframe. If dataframe, must have a column named
                'station_id' with the station IDs of interest
        st_dt:  Str. Format 'YYYY-MM-DD'
        end_dt: Str. Format 'YYYY-MM-DD'
        engine: Obj. Active NDB "engine" object

    Returns:
        Dataframe
    """
    # Get stn IDs
    assert len(stn_df) > 0, "ERROR: Please select at least one station."
    if isinstance(stn_df, pd.DataFrame):
        stn_df["station_id"].drop_duplicates(inplace=True)
        stn_ids = stn_df["station_id"].values.astype(int).tolist()
    else:  # Should be a list
        stn_ids = list(set(stn_df))

    # Convert dates
    st_dt = dt.datetime.strptime(st_dt, "%Y-%m-%d")
    end_dt = dt.datetime.strptime(end_dt, "%Y-%m-%d")

    # Query db
    bind_pars = ",".join("(1, %d)" % i for i in stn_ids)

    sql = (
        "SELECT DISTINCT parameter_id, "
        "  name AS parameter_name, "
        "  unit "
        "FROM resa2.parameter_definitions a "
        "JOIN resa2.wc_parameters_methods b "
        "ON a.parameter_id = b.wc_parameter_id "
        "JOIN resa2.water_chemistry_values2 c "
        "ON b.wc_method_id = c.method_id "
        "JOIN resa2.water_samples d "
        "ON c.sample_id = d.water_sample_id "
        "WHERE (1, d.station_id) IN (%s) "
        "AND sample_date >= :st_dt "
        "AND sample_date <= :end_dt "
        "ORDER BY parameter_name, "
        "  unit" % bind_pars
    )
    par_dict = {"end_dt": end_dt, "st_dt": st_dt}
    with engine.connect() as connection:
        df = pd.read_sql(sql, params=par_dict, con=connection)

    print("%s parameters available for the selected stations and dates." % len(df))

    return df


def select_ndb_station_parameters(stn_df, st_dt, end_dt, engine):
    """Gets the list of available water chemistry parameters for the
    selected stations.

    NOTE: This is an alternative to get_station_parameters().

    It looks as though Tore/Roar have already written code to
    summarise data into NIVADATABASE.WCV_CALK. Assuming this table
    is reliable, it is easier to query directly than to refactor lots
    of PL/SQL into Python.

    Args:
        stn_df: List or Dataframe. If dataframe, must have a column named
                'station_id' with the station IDs of interest
        st_dt:  Str. Format 'YYYY-MM-DD'
        end_dt: Str. Format 'YYYY-MM-DD'
        engine: Obj. Active NDB "engine" object

    Returns:
        Dataframe
    """
    # Get stn IDs
    assert len(stn_df) > 0, "ERROR: Please select at least one station."
    if isinstance(stn_df, pd.DataFrame):
        stn_df["station_id"].drop_duplicates(inplace=True)
        stn_ids = stn_df["station_id"].values.astype(int).tolist()
    else:  # Should be a list
        stn_ids = list(set(stn_df))

    # Convert dates
    st_dt = dt.datetime.strptime(st_dt, "%Y-%m-%d")
    end_dt = dt.datetime.strptime(end_dt, "%Y-%m-%d")

    # Query db
    bind_pars = ",".join("(1, %d)" % i for i in stn_ids)

    sql = (
        "SELECT DISTINCT parameter_id, "
        "  name AS parameter_name, "
        "  unit "
        "FROM nivadatabase.wcv_calk "
        "WHERE (1, station_id) IN (%s) "
        "AND sample_date  >= :st_dt "
        "AND sample_date  <= :end_dt "
        "ORDER BY name, "
        "  unit" % bind_pars
    )

    par_dict = {"end_dt": end_dt, "st_dt": st_dt}
    with engine.connect() as connection:
        df = pd.read_sql(sql, params=par_dict, con=connection)

    print("%s parameters available for the selected stations and dates." % len(df))

    # Convert to qgrid
    # col_opts = { 'editable': False }
    # grid = qgrid.show_grid(df, column_options=col_opts, show_toolbar=False)

    return df  # grid


def select_resa_water_chemistry(
    stn_df, par_df, st_dt, end_dt, engine, lod_flags=False, drop_dups=False
):
    """Get water chemistry data from RESA2 for selected station-parameter-
    date combinations.

    Args:
        stn_df:    List or Dataframe. If dataframe, must have a column named
                   'station_id' with the station IDs of interest
        par_df:    List or Dataframe. If dataframe, must have a column named
                   'parameter_id' with the parameter IDs of interest
        st_dt:     Str. Format 'YYYY-MM-DD'
        end_dt:    Str. Format 'YYYY-MM-DD'
        engine:    Obj. Active NDB "engine" object
        lod_flags: Bool. Whether to include LOD flags in output
        drop_dups: Bool. Whether to retain duplicated rows in cases where
                   the same station ID is present with multiple names

    Returns:
        Tuple of dataframes (wc_df, dup_df)
    """
    # Get stn IDs
    assert len(stn_df) > 0, "ERROR: Please select at least one station."
    if isinstance(stn_df, pd.DataFrame):
        stn_df["station_id"].drop_duplicates(inplace=True)
        stn_ids = stn_df["station_id"].values.astype(int).tolist()
    else:  # Should be a list
        stn_ids = list(set(stn_df))

    # Get par IDs
    assert len(par_df) > 0, "ERROR: Please select at least one parameter."
    if isinstance(par_df, pd.DataFrame):
        par_df["parameter_id"].drop_duplicates(inplace=True)
        par_ids = par_df["parameter_id"].values.astype(int).tolist()
    else:  # Should be a list
        par_ids = list(set(par_df))

    # Convert dates
    st_dt = dt.datetime.strptime(st_dt, "%Y-%m-%d")
    end_dt = dt.datetime.strptime(end_dt, "%Y-%m-%d")

    # Number from 0 to n_stns. Uses hack to avoid Oracle 1000 item limit in lists
    # https://stackoverflow.com/questions/17842453/is-there-a-workaround-for-ora-01795-maximum-number-of-expressions-in-a-list-is
    bind_stns = ",".join("(1, %d)" % i for i in stn_ids)

    # Number from n_stns to (n_stns+n_params)
    bind_pars = ",".join("%d" % i for i in par_ids)

    sql = (
        "SELECT b.station_id, "
        "  e.station_code, "
        "  e.station_name, "
        "  b.sample_date, "
        "  b.depth1, "
        "  b.depth2, "
        "  d.name AS parameter_name, "
        "  d.unit, "
        "  a.flag1, "
        "  (a.value*c.conversion_factor) as value, "
        "  a.entered_date "
        "FROM resa2.water_chemistry_values2 a, "
        "resa2.water_samples b, "
        "resa2.wc_parameters_methods c, "
        "resa2.parameter_definitions d, "
        "resa2.stations e "
        "WHERE a.sample_id IN ( "
        "  SELECT b.water_sample_id FROM resa2.water_samples "
        "  WHERE (1, b.station_id) IN (%s) "
        "  AND b.sample_date <= :end_dt "
        "  AND b.sample_date >= :st_dt) "
        "AND a.approved = 'YES' "
        "AND c.wc_parameter_id IN (%s) "
        "AND a.sample_id = b.water_sample_id "
        "AND a.method_id = c.wc_method_id "
        "AND c.wc_parameter_id = d.parameter_id "
        "AND b.station_id = e.station_id" % (bind_stns, bind_pars)
    )

    par_dict = {"end_dt": end_dt, "st_dt": st_dt}
    with engine.connect() as connection:
        df = pd.read_sql(sql, params=par_dict, con=connection)

    # Drop exact duplicates (i.e. including value)
    df.drop_duplicates(
        subset=[
            "station_id",
            "station_code",
            "station_name",
            "sample_date",
            "depth1",
            "depth2",
            "parameter_name",
            "unit",
            "flag1",
            "value",
        ],
        inplace=True,
    )

    # Check for "problem" duplicates i.e. duplication NOT caused by having
    # several names for the same station
    dup_df = df[
        df.duplicated(
            subset=[
                "station_id",
                "station_code",
                "station_name",
                "sample_date",
                "depth1",
                "depth2",
                "parameter_name",
                "unit",
            ],
            keep=False,
        )
    ].sort_values(
        by=[
            "station_id",
            "station_code",
            "station_name",
            "sample_date",
            "depth1",
            "depth2",
            "parameter_name",
            "unit",
            "entered_date",
        ]
    )

    if len(dup_df) > 0:
        print(
            "WARNING\nThe database contains unexpected duplicate values for "
            "some station-date-parameter combinations.\nOnly the most recent "
            "values will be used, but you should check the repeated values are "
            "not errors.\nThe duplicated entries are returned in a separate "
            "dataframe.\n"
        )

        # Choose most recent record for each duplicate
        df.sort_values(by="entered_date", inplace=True, ascending=True)

        # Drop duplicates
        df.drop_duplicates(
            subset=[
                "station_id",
                "station_code",
                "station_name",
                "sample_date",
                "depth1",
                "depth2",
                "parameter_name",
                "unit",
            ],
            keep="last",
            inplace=True,
        )

    # Drop "expected" duplicates (i.e. duplicated station names), if desired
    if drop_dups:
        df.drop_duplicates(
            subset=[
                "station_id",
                "sample_date",
                "depth1",
                "depth2",
                "parameter_name",
                "unit",
            ],
            keep="last",
            inplace=True,
        )

    # Restructure data
    del df["entered_date"]
    df["parameter_name"].fillna("", inplace=True)
    df["unit"].fillna("", inplace=True)
    df["par_unit"] = df["parameter_name"].astype(str) + "_" + df["unit"].astype(str)
    del df["parameter_name"], df["unit"]

    # Include LOD flags?
    if lod_flags:
        df["flag1"].fillna("", inplace=True)
        df["value"] = df["flag1"].astype(str) + df["value"].astype(str)
        del df["flag1"]

    else:  # Ignore flags
        del df["flag1"]

    # Unstack
    df.set_index(
        [
            "station_id",
            "station_code",
            "station_name",
            "sample_date",
            "depth1",
            "depth2",
            "par_unit",
        ],
        inplace=True,
    )
    df = df.unstack(level="par_unit")

    # Tidy
    df.reset_index(inplace=True)
    df.index.name = ""
    df.columns = list(df.columns.get_level_values(0)[:6]) + list(
        df.columns.get_level_values(1)[6:]
    )
    df.sort_values(by=["station_id", "sample_date"], inplace=True)

    return (df, dup_df)


def select_ndb_water_chemistry(
    stn_df, par_df, st_dt, end_dt, engine, lod_flags=False, drop_dups=False
):
    """Get water chemistry data from NIVADATABASE for selected station-
    parameter-date combinations.

        NOTE: This function queries NIVADATABASE.WCV_CALK.

        It looks as though Tore/Roar have already written code to deal
        with duplicates etc., similar to what I originally implemented
        Flask for get_chemistry_values(). See NIVADATABASE.PKG_WC_COMPUTED
        (which updates NIVADATABASE.WCV_CALK) for details.

        I think there may still be some issues with WCV_CALK, but one
        advantage is that it includes *some* of the "calculated"
        parameters available in RESA2 (e.g. ANC).

        Assuming this table is reliable, it is easier to query directly
        than to refactor lots of PL/SQL into Python.

    Args:
        stn_df:    List or Dataframe. If dataframe, must have a column named
                   'station_id' with the station IDs of interest
        par_df:    List or Dataframe. If dataframe, must have a column named
                   'parameter_id' with the parameter IDs of interest
        st_dt:     Str. Format 'YYYY-MM-DD'
        end_dt:    Str. Format 'YYYY-MM-DD'
        engine:    Obj. Active NDB "engine" object
        lod_flags: Bool. Whether to include LOD flags in output
        drop_dups: Bool. Whether to retain duplicated rows in cases where
                   the same station ID is present with multiple names

    Returns:
        Tuple of dataframes (wc_df, dup_df)
    """
    # Get stn IDs
    assert len(stn_df) > 0, "ERROR: Please select at least one station."
    if isinstance(stn_df, pd.DataFrame):
        stn_df["station_id"].drop_duplicates(inplace=True)
        stn_ids = stn_df["station_id"].values.astype(int).tolist()
    else:  # Should be a list
        stn_ids = list(set(stn_df))

    # Get par IDs
    assert len(par_df) > 0, "ERROR: Please select at least one parameter."
    if isinstance(par_df, pd.DataFrame):
        par_df["parameter_id"].drop_duplicates(inplace=True)
        par_ids = par_df["parameter_id"].values.astype(int).tolist()
    else:  # Should be a list
        par_ids = list(set(par_df))

    # Convert dates
    st_dt = dt.datetime.strptime(st_dt, "%Y-%m-%d")
    end_dt = dt.datetime.strptime(end_dt, "%Y-%m-%d")

    # Number from 0 to n_stns. Uses hack to avoid Oracle 1000 item limit in lists
    # https://stackoverflow.com/questions/17842453/is-there-a-workaround-for-ora-01795-maximum-number-of-expressions-in-a-list-is
    bind_stns = ",".join("(1, %d)" % i for i in stn_ids)

    # Number from n_stns to (n_stns+n_params)
    bind_pars = ",".join("%d" % i for i in par_ids)

    # Query db
    sql = (
        "SELECT a.station_id, "
        "  a.station_code, "
        "  a.station_name, "
        "  b.sample_date, "
        "  b.depth1, "
        "  b.depth2, "
        "  b.name AS parameter_name, "
        "  b.unit, "
        "  b.flag1, "
        "  b.value, "
        "  b.entered_date "
        "FROM nivadatabase.projects_stations a, "
        "  nivadatabase.wcv_calk b "
        "WHERE a.station_id  = b.station_id "
        "AND (1, a.station_id)   IN (%s) "
        "AND b.parameter_id IN (%s) "
        "AND sample_date    >= :st_dt "
        "AND sample_date    <= :end_dt" % (bind_stns, bind_pars)
    )
    par_dict = {"end_dt": end_dt, "st_dt": st_dt}
    with engine.connect() as connection:
        df = pd.read_sql(sql, params=par_dict, con=connection)

    # Drop exact duplicates (i.e. including value)
    df.drop_duplicates(
        subset=[
            "station_id",
            "station_code",
            "station_name",
            "sample_date",
            "depth1",
            "depth2",
            "parameter_name",
            "unit",
            "flag1",
            "value",
        ],
        inplace=True,
    )

    # Check for "problem" duplicates i.e. duplication NOT caused by having
    # several names for the same station
    dup_df = df[
        df.duplicated(
            subset=[
                "station_id",
                "station_code",
                "station_name",
                "sample_date",
                "depth1",
                "depth2",
                "parameter_name",
                "unit",
            ],
            keep=False,
        )
    ].sort_values(
        by=[
            "station_id",
            "station_code",
            "station_name",
            "sample_date",
            "depth1",
            "depth2",
            "parameter_name",
            "unit",
            "entered_date",
        ]
    )

    if len(dup_df) > 0:
        print(
            "WARNING\nThe database contains unexpected duplicate values for "
            "some station-date-parameter combinations.\nOnly the most recent "
            "values will be used, but you should check the repeated values are "
            "not errors.\nThe duplicated entries are returned in a separate "
            "dataframe.\n"
        )

        # Choose most recent record for each duplicate
        df.sort_values(by="entered_date", inplace=True, ascending=True)

        # Drop duplicates
        df.drop_duplicates(
            subset=[
                "station_id",
                "station_code",
                "station_name",
                "sample_date",
                "depth1",
                "depth2",
                "parameter_name",
                "unit",
            ],
            keep="last",
            inplace=True,
        )

    # Drop "expected" duplicates (i.e. duplicated station names), if desired
    if drop_dups:
        df.drop_duplicates(
            subset=[
                "station_id",
                "sample_date",
                "depth1",
                "depth2",
                "parameter_name",
                "unit",
            ],
            keep="last",
            inplace=True,
        )

    # Restructure data
    del df["entered_date"]
    df["parameter_name"].fillna("", inplace=True)
    df["unit"].fillna("", inplace=True)
    df["par_unit"] = df["parameter_name"].astype(str) + "_" + df["unit"].astype(str)
    del df["parameter_name"], df["unit"]

    # Include LOD flags?
    if lod_flags:
        df["flag1"].fillna("", inplace=True)
        df["value"] = df["flag1"].astype(str) + df["value"].astype(str)
        del df["flag1"]

    else:  # Ignore flags
        del df["flag1"]

    # Unstack
    df.set_index(
        [
            "station_id",
            "station_code",
            "station_name",
            "sample_date",
            "depth1",
            "depth2",
            "par_unit",
        ],
        inplace=True,
    )
    df = df.unstack(level="par_unit")

    # Tidy
    df.reset_index(inplace=True)
    df.index.name = ""
    df.columns = list(df.columns.get_level_values(0)[:6]) + list(
        df.columns.get_level_values(1)[6:]
    )
    df.sort_values(by=["station_id", "sample_date"], inplace=True)

    return (df, dup_df)


def extract_resa_discharge(stn_id, st_dt, end_dt, engine, plot=False):
    """Extracts daily flow time series for the selected site. Returns a
        dataframe and, optionally, a plot object

    Args:
        stn_id:    Int. Valid RESA2 STATION_ID
        st_dt:     Str. Start date as 'yyyy-mm-dd'
        end_dt:    Str. End date as 'yyyy-mm-dd'
        engine:    SQL-Alchemy 'engine' object already connected to RESA2
        plot:      Bool. Choose whether to return a grid plot as well as the
                   dataframe

    Returns:
        If plot is False, returns a dataframe of flows, otherwise
        returns a tuple (q_df, figure object)
    """
    # Check stn is valid
    sql = "SELECT * FROM resa2.stations WHERE station_id = :stn_id"
    with engine.connect() as connection:
        stn_df = pd.read_sql_query(sql, params={"stn_id": stn_id}, con=connection)
    assert len(stn_df) == 1, "Error in station code."

    # Get station_code
    stn_code = stn_df["station_code"].iloc[0]

    # Check a discharge station is defined for this WC station
    sql = "SELECT * FROM resa2.default_dis_stations WHERE station_id = :stn_id"
    with engine.connect() as connection:
        dis_df = pd.read_sql_query(sql, params={"stn_id": stn_id}, con=connection)
    assert len(dis_df) == 1, "Error identifying discharge station."

    # Get ID for discharge station
    dis_stn_id = int(dis_df["dis_station_id"].iloc[0])

    # Get catchment areas
    # Discharge station
    sql = (
        "SELECT area FROM resa2.discharge_stations "
        "WHERE dis_station_id = :dis_stn_id"
    )
    with engine.connect() as connection:
        area_df = pd.read_sql_query(
            sql, params={"dis_stn_id": dis_stn_id}, con=connection
        )
    dis_area = area_df["area"].iloc[0]

    # Chemistry station
    wc_area = stn_df["catchment_area"].iloc[0]

    # Get the daily discharge data for this station
    st_dt = dt.datetime.strptime(st_dt, "%Y-%m-%d")
    end_dt = dt.datetime.strptime(end_dt, "%Y-%m-%d")
    sql = (
        "SELECT xdate, xvalue FROM resa2.discharge_values "
        "WHERE dis_station_id = :dis_stn_id "
        "AND xdate >= :st_dt "
        "AND xdate <= :end_dt"
    )
    with engine.connect() as connection:
        q_df = pd.read_sql_query(
            sql,
            params={"dis_stn_id": dis_stn_id, "st_dt": st_dt, "end_dt": end_dt},
            con=connection,
        )

    q_df.columns = ["date", "flow_m3/s"]
    q_df.index = q_df["date"]
    del q_df["date"]

    # Scale flows by area
    q_df = q_df * wc_area / dis_area

    # Convert to daily
    q_df = q_df.resample("D").mean()

    # Linear interpolation of NoData gaps
    q_df.interpolate(method="linear", inplace=True)

    # Plot
    if plot:
        ax = q_df.plot(legend=False, figsize=(8, 4))
        ax.set_xlabel("")
        ax.set_ylabel("Flow (m3/s)", fontsize=16)
        ax.set_title("Discharge at %s" % stn_code, fontsize=16)
        plt.tight_layout()

        return (q_df, ax)

    else:
        return q_df


def gdf_to_postgis(gdf, table_name, schema, eng, sp_index, create_pk=True, **kwargs):
    """Writes a geodataframe to a postgis database. Adapted from:

           https://github.com/geopandas/geopandas/pull/440

    Args:
        gdf:        Geodataframe
        table_name: Str. Name of table in database. If the table exists,
                    be sure to specify the 'if_exists' kwarg (see
                    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html)
        schema:     Str. Name of schema to use
        con:        SQLAlchemy engine connected to PostGIS db
        sp_index:   Str. Name of spatial index to create. Pass None to ignore.
                    Building a spatial index can dramatically improve the
                    performance of spatial queries and is usually a good idea.
                    A name such as 'schema_tablename_spidx' is a reasonable
                    default
        create_pk:  Bool. Whether to create a primary key column named 'id'

        **kwargs are passed to pandas.to_sql; see documentation for available
        parameters

    Returns:
        None. The table is written to the database.
    """

    def convert_geometry(geom):
        return binascii.hexlify(shapely.wkb.dumps(geom)).decode()

    # Get EPSG
    epsg = gdf.crs.to_epsg()

    # Tidy for postgis
    temp_gdf = gdf.copy()
    temp_gdf["geom"] = gdf.geometry.apply(convert_geometry)
    if gdf.geometry.name != "geom":
        del temp_gdf[gdf.geometry.name]

    # Check for consistent geometries
    geom_types = set(gdf.geom_type.unique())

    if len(geom_types) == 1:
        # All geoms the same
        geom_type = list(geom_types)[0].upper()

        # Write to db
        with eng.connect() as conn:
            temp_gdf.to_sql(table_name, conn, schema=schema, **kwargs)
            sql = (
                "ALTER TABLE {schema}.{table_name} "
                "ALTER COLUMN geom TYPE geometry({geom_type}, {epsg}) "
                "USING ST_SetSRID(geom, {epsg})"
            )
            sql_args = {
                "table_name": table_name,
                "schema": schema,
                "geom_type": geom_type,
                "epsg": epsg,
            }
            conn.execute(sql.format(**sql_args))
            conn.commit()

    elif geom_types in [
        set(["Point", "MultiPoint"]),
        set(["LineString", "MultiLineString"]),
        set(["Polygon", "MultiPolygon"]),
    ]:
        print("WARNING! The dataframe contains mixed geometries:")
        print(" ", geom_types)
        print(
            'These will be cast to "Multi" type. If this is '
            "not what you want, consider using gdf.explode() first"
        )

        # Get correct multi type
        geom_type = [i for i in geom_types if i[0] == "M"]
        geom_type = geom_type[0].upper()

        # Write to db
        with eng.connect() as conn:
            temp_gdf.to_sql(table_name, conn, schema=schema, **kwargs)
            sql = (
                "ALTER TABLE {schema}.{table_name} "
                "ALTER COLUMN geom TYPE geometry({geom_type}, {epsg}) "
                "USING ST_Multi(ST_SetSRID(geom, {epsg}))"
            )
            sql_args = {
                "table_name": table_name,
                "schema": schema,
                "geom_type": geom_type,
                "epsg": epsg,
            }
            conn.execute(sql.format(**sql_args))
            conn.commit()

    else:
        raise ValueError(
            "The geometry types in the dataframe seem inconsistent:\n"
            "    %s" % geom_types
        )

    # Build spatial index if required
    if sp_index:
        sql = "CREATE INDEX {idx_name} " "ON {schema}.{table_name} " "USING GIST (geom)"
        sql_args = {"table_name": table_name, "schema": schema, "idx_name": sp_index}
        with eng.connect() as conn:
            conn.execute(sql.format(**sql_args))
            conn.commit()

    # Add primary key if required
    if create_pk:
        sql = "ALTER TABLE {schema}.{table_name} " "ADD COLUMN id SERIAL PRIMARY KEY"
        sql_args = {"table_name": table_name, "schema": schema}
        with eng.connect() as conn:
            conn.execute(sql.format(**sql_args))
            conn.commit()


def select_resa_stations_in_polygon(poly_vec, id_col, eng):
    """Identifies all stations within RESA2 that are located within the
        specified vector polygon layer (e.g. .shp or .geojson). Return a stations
        dataframe where each station is assigned the ID from 'id_col'.

    Args:
        poly_vec:  Str. Raw path to polygon vector dataset (.shp, .geojson etc.).
        id_col:    Str. Column containing unique ID for each polygon in 'poly_vec'
        eng:       Obj. Active "engine" object returned by nivapy.da.connect

    Returns:
        Dataframe.
    """
    # Query all NDB stations
    stn_df = select_resa_stations(eng)

    # Identify point in poly
    stn_df = spatial.identify_point_in_polygon(stn_df, poly_vec, poly_col=id_col)

    # Get just stations within polys
    stn_df.dropna(
        subset=[
            id_col,
        ],
        inplace=True,
    )

    print("%s stations within the specified polygons." % len(stn_df))

    return stn_df


def select_ndb_stations_in_polygon(poly_vec, id_col, eng, drop_dups=False):
    """Identifies all stations within the NIVADATABASE that are located within the
        specified vector polygon layer (e.g. .shp or .geojson). Return a stations
        dataframe where each station is assigned the ID from 'id_col'.

    Args:
        poly_vec:  Str. Raw path to polygon vector dataset (.shp, .geojson etc.).
        id_col:    Str. Column containing unique ID for each polygon in 'poly_vec'
        eng:       Obj. Active "engine" object returned by nivapy.da.connect
        drop_dups: Bool. The same station may have different names in different
                   projects. If some of the selected projects include the
                   same station, this will result in duplicates in the
                   stations table (i.e. same station ID, but multiple names).
                   By default, the duplicates will be returned. Setting
                   'drop_dups=True' will select one set of names per station
                   ID and return a dataframe with no duplicates (but the
                   station codes and names may not be what you're expecting)

    Returns:
        Dataframe.
    """
    # Query all NDB stations
    stn_df = select_ndb_stations(eng)

    # Identify point in poly
    stn_df = spatial.identify_point_in_polygon(stn_df, poly_vec, poly_col=id_col)

    # Get just stations within polys
    stn_df.dropna(
        subset=[
            id_col,
        ],
        inplace=True,
    )

    # Drop duplicated stations if necessary
    if drop_dups:
        stn_df.drop_duplicates(
            subset=[
                "station_id",
            ],
            inplace=True,
        )

    print("%s stations within the specified polygons." % len(stn_df))

    return stn_df


def postgis_raster_to_array(eng, pg_ras, schema="public", band=1):
    """DEPRECATED! Performance and memory consumptions of raster SQL in
    PostGIS seems terrible. Reading ratsers directly using GDAL seems much
    more reliable. Use nivapy.spatial.read_raster() instead.

    Read a PostGIS ratser dataset into a numpy array.

    Args:
        eng:    Obj. Active database engine object connected to PostGIS
        pg_ras: Str or result object. Either a string identifying the
                table name of interest, or a raster result object
                returned by a previous spatial query
        schema: Str. Name of schema in which 'pg_ras' is located.
                Ignored if pg_ras is a result object
        band:   Int. Band to read

    Returns:
        Tuple (arr, ndv, extent, epsg).

        arr:    Array. 2D array of data values
        ndv:    Float. NoData value
        extent: Tuple. (xmin, xmax, ymin, ymax)
        epsg:   Int. EPSG code defining the spatial reference
    """
    print(
        "WARNING: This function has been deprecated.\n"
        "Consider using nivapy.spatial.read_raster() instead."
    )

    # What is pg_ras
    if isinstance(pg_ras, sqlalchemy.engine.ResultProxy):
        # Already have raster 'result' obj
        res = pg_ras
    elif isinstance(pg_ras, str):
        # User wants to select an entire raster
        # Build query
        sql = "SELECT ST_AsGDALRaster(ST_Union(rast), 'GTiff') " "FROM %s.%s" % (
            schema,
            pg_ras,
        )
        res = eng.execute(sql)

    # Make virtual file for GDAL
    vsipath = "/vsimem/from_postgis"
    gdal.FileFromMemBuffer(vsipath, bytes(res.fetchone()[0]))

    # Read chosen band of raster with GDAL
    ds = gdal.Open(vsipath)
    band = ds.GetRasterBand(band)
    arr = band.ReadAsArray()

    # Get ds properties
    geotransform = ds.GetGeoTransform()
    originX = geotransform[0]  # Origin is top-left corner
    originY = geotransform[3]  # i.e. (xmin, ymax)
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = ds.RasterXSize
    rows = ds.RasterYSize

    # Calculate extent
    xmin = int(originX)
    xmax = int(originX + cols * pixelWidth)
    ymin = int(originY + rows * pixelHeight)
    ymax = int(originY)
    extent = (xmin, xmax, ymin, ymax)

    # Get No Data Value
    ndv = band.GetNoDataValue()

    # Get CRS
    proj = osr.SpatialReference(wkt=ds.GetProjection())
    proj.AutoIdentifyEPSG()
    epsg = int(proj.GetAttrValue("AUTHORITY", 1))

    # Close and clean up virtual memory file
    ds = band = None
    gdal.Unlink(vsipath)

    return (arr, ndv, extent, epsg)


def postgis_raster_to_geotiff(
    dbname, host, port, schema, table, out_tif, column="rast", band=1, flip=False
):
    """Convert a PostGIS raster dataset to a GeoTiff. You will be asked to
                                enter a user name and password.

                            Args:
                                dbname:  Str. Name of db
                                host:    Str. Hostname. Use 'host.docker.internal' for the Docker host
                                port:    Int. Port number for db connection
                                schema:  Str. Name of schema
                                table:   Str. Name of table
                                out_tif: Raw str. Path for GeoTiff to create
                                column:  Str. Name of 'raster' column in db table
                                band:    Int. Band to read
        flip:    Bool. Sometimes required?

                            Returns:
                                The GeoTiff is saved to the specified path.
    """
    # Extract PostGIS raster to array
    pg_dict = {
        "dbname": dbname,
        "host": host,
        "port": str(port),
        "schema": schema,
        "table": table,
        "column": column,
    }

    data, ndv, epsg, extent = spatial.read_raster("postgres", pg_dict=pg_dict)

    if ndv is None:
        ndv = -9999

    # Get properties for writing GeoTiff
    xmin, xmax, ymin, ymax = extent
    rows, cols = data.shape
    cell_size = (xmax - xmin) / cols

    # Projection details
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(epsg)
    proj4_str = proj.ExportToProj4()

    # Write to GeoTiff
    spatial.array_to_gtiff(
        xmin, ymax, cell_size, out_tif, data, proj4_str, no_data_value=ndv
    )


def raster_to_postgis(
    ras_path,
    db_name,
    schema,
    table,
    epsg,
    host="host.docker.internal",
    port=25432,
    tile_size="100x100",
    pyramids=True,
    pyramid_levels=[2, 4, 8, 16, 32, 64, 128, 256, 512],
):
    """Import a raster to a PostGIS database using the 'raster2pgsql' command
        line tool.

    Args:
        ras_path:       Str. Raw path to raster file
        db_name:        Str. Name of database to connect to
        schema:         Str. Name of schema in database to which table will be added
        table:          Str. Name of ratser table to create
        epsg:           Int. EPSG code for projection of raster
        host:           Str. Host name or IP address
        port:           Int. Port to connect to on 'host'
        tile_size:      Str. E.g. '100x100' pixels
        pyramids:       Bool. Whether to build raster pyramids
        pyramid_levels: List of ints. Number and scale factor of levels to use. Only
                        used if 'pyramids'=True.

    Returns:
        None. The raster is added to the database.
    """
    # Dict of params
    cmd_dict = {
        "ras_path": ras_path,
        "schema": schema,
        "table": table,
        "epsg": epsg,
        "host": host,
        "port": port,
        "db": db_name,
        "tile_size": tile_size,
    }

    # Get user and pw
    cmd_dict["user"] = getpass.getpass(prompt="Username: ")
    cmd_dict["pw"] = getpass.getpass(prompt="Password: ")

    # Deal with pyramids
    if pyramids:
        # Check levels provided
        assert isinstance(pyramid_levels, list), (
            "ERROR: 'pyramid_levels' must be " "passed as a list when 'pyramids'=True."
        )

        # Check levels are valid
        valid_levs = set([2, 4, 8, 16, 32, 64, 128, 256, 512])
        user_levs = set(pyramid_levels)
        assert user_levs.issubset(valid_levs), (
            "ERROR: Valid pyramid levels are " "[2, 4, 8, 16, 32, 64, 128, 256, 512]."
        )

        # Build str
        pyram_str = "-l " + ",".join([str(i) for i in pyramid_levels])

    else:
        pyram_str = ""

    cmd_dict["pyramids"] = pyram_str

    # Build cmd string
    cmd = (
        "raster2pgsql "
        "-s {epsg} "
        "-d -C -I -M "
        "{pyramids} "
        "{ras_path} "
        "-t {tile_size} "
        "{schema}.{table} "
        "| PGPASSWORD={pw} "
        "psql -h {host} "
        "-U {user} "
        "-p {port} "
        "-d {db}"
    ).format(**cmd_dict)

    # Execute
    print("\nProcessing data for %s.%s..." % (schema, table))
    res = subprocess.check_call([cmd], shell=True)

    if res == 0:
        print("\nRaster loaded successfully.")

    return None


def read_postgis(schema, table, engine, clip=None):
    """Read a layer/table from the JupyterHub's PostGis instance.

        Available layers are listed here:

        https://github.com/NIVANorge/niva_jupyter_hub/blob/master/postgis_db/postgis_db_dataset_summary.md

    Args:
        schema:    Str. Name of schema
        table:     Str. Name of layer/table
        engine:    Obj. Valid connection object from nivapy.da.connect_postgis()
        clip:      Geodataframe or None. Optionally, clip the result to the
                   bounding box of a geodataframe. The returned features will have
                   a rectangular extent in the co-ordinate system of the original
                   target table (not the CRS of 'clip'). Features will be clipped
                   to the full extent of all features in clip. The main aim is to
                   reduce data volumes. For spatial queries (joins, intersections
                   etc.), either write your own SQL statements or use the spatial
                   analysis features in GeoPandas and NivaPy

    Returns:
        Geodataframe.
    """
    if clip is not None:
        clip = clip.copy()
        assert isinstance(
            clip, gpd.GeoDataFrame
        ), "'clip' must be a (Multi-)Polygon geodataframe."

        # Get CRS of target dataset
        sql = f"SELECT * FROM {schema}.{table} " "LIMIT 1"
        with engine.connect() as connection:
            col_gdf = gpd.read_postgis(sql, connection)
        cols = list(col_gdf.columns)
        cols.remove("geom")
        cols = [f"a.{i}" for i in cols]
        cols = ", ".join(cols)

        # Reproject if necessary
        if col_gdf.crs != clip.crs:
            print("WARNING: The projection of 'clip' differs from the target dataset.")
            print("Converting to the projection of target dataset (%s)" % col_gdf.crs)
            clip.to_crs(crs=col_gdf.crs, inplace=True)

        # Get bounding box for clip
        xmin, ymin, xmax, ymax = clip.total_bounds

        # Get intersection
        sql = (
            f"WITH bbox AS ( "
            f"  SELECT ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax}, "
            f"    Find_SRID('{schema}', '{table}', 'geom')) AS geom) "
            f"SELECT {cols}, ST_Intersection(a.geom, b.geom) AS geom "
            f"FROM {schema}.{table} AS a, "
            f"  bbox AS b "
            f"WHERE ST_Intersects(a.geom, b.geom)"
        )
        with engine.connect() as connection:
            gdf = gpd.read_postgis(sql, connection)

    else:
        # Read layer
        sql = f"SELECT * FROM {schema}.{table}"
        with engine.connect() as connection:
            gdf = gpd.read_postgis(sql, connection)

    return gdf


def select_jhub_projects(eng):
    """List all NIVA projects in the 'general' schema of the JupyterHub's PostGIS
        database.

    Args:
        eng:      Obj. Database engine returned by nivapy.da.connect_postgis

    Returns:
        Dataframe of projects.
    """
    sql = "SELECT * FROM niva.projects"
    with eng.connect() as connection:
        df = pd.read_sql(sql, connection)

    return df


def select_jhub_project_catchments(proj_ids, eng):
    """Catch catchment boundaries (polygons) and outflow points for the specified projects.

    Args:
        proj_ids: List. List of project IDs of interest
        eng:      Obj. Database engine returned by nivapy.da.connect_postgis

    Returns:
        Tuple of geodataframes (outlets, catchments).

            outlets:    A POINT geodataframe with the catchment outlets used to calculate the
                        boundaries (if available; if not, the "raw" station locations are stored
                        instead)
            catchments: A (MULTI)POLYGON geodataframe with the corresponding catchment boundaries
    """
    assert isinstance(proj_ids, list), "'proj_ids' must be a list."

    # Get stations
    sql = (
        "SELECT * FROM niva.stations "
        "WHERE station_id IN ( "
        "  SELECT station_id FROM niva.projects_stations "
        "  WHERE project_id = %(proj_ids)s "
        ")"
    )
    with eng.connect() as connection:
        stn_gdf = gpd.read_postgis(
            sql, connection, params={"proj_ids": tuple(proj_ids)}
        )

    # Get catchments
    sql = "SELECT * from niva.catchments " "WHERE station_id IN %(stns)s"
    stns = tuple(stn_gdf["station_id"].tolist())
    with eng.connect() as connection:
        cat_gdf = gpd.read_postgis(sql, connection, params={"stns": stns})

    # Join
    cat_gdf = cat_gdf.merge(
        stn_gdf[
            [
                "station_id",
                "station_code",
                "station_name",
                "aquamonitor_id",
                "longitude",
                "latitude",
            ]
        ],
        on="station_id",
        how="left",
    )

    cat_gdf = cat_gdf[
        [
            "station_id",
            "station_code",
            "station_name",
            "aquamonitor_id",
            "longitude",
            "latitude",
            "geom",
        ]
    ]

    return (stn_gdf, cat_gdf)


def authenticate_nve_hydapi(authfile_path=r"/home/jovyan/.nve-hydapi-key"):
    """
    Attempts to read the user's API key for NVE's Hydrological API from a file.

    Args:
        authfile_path: Str. The path to the authentication file. Default is
            '/home/jovyan/.nve-hydapi-key'.

    Returns:
        Str. The API key if the file exists and the key is found. Otherwise, raises
        an exception.

    Raises:
        FileNotFoundError: If the authentication file does not exist.
        KeyError: If the 'Auth' section or 'key' option is not found in the file.
    """
    if not os.path.isfile(authfile_path):
        raise FileNotFoundError(
            f"Could not find the file '{authfile_path}'.\n"
            "Either create this file (recommended) or specify your API key explicitly in the function call."
        )

    config = configparser.ConfigParser()
    config.read(authfile_path)

    try:
        return config.get("Auth", "key")
    except (configparser.NoSectionError, configparser.NoOptionError):
        raise KeyError(
            "Could not read API key from file. Ensure the file has an 'Auth' section with a 'key' option."
        )


def get_nve_hydapi_parameters(api_key=None):
    """List parameters available via NVE's Hydrological API (https://hydapi.nve.no/UserDocumentation/)."""
    # Authenticate
    if api_key is None:
        api_key = authenticate_nve_hydapi()

    url = "https://hydapi.nve.no/api/v1/Parameters"
    request_headers = {"Accept": "application/json", "X-API-Key": api_key}
    request = Request(url, headers=request_headers)
    response = urlopen(request)
    content = response.read().decode("utf-8")
    df = json_normalize(json.loads(content)["data"])

    return df


def get_nve_hydapi_stations(api_key=None):
    """List all stations available via NVE's Hydrological API (https://hydapi.nve.no/UserDocumentation/)."""
    # Authenticate
    if api_key is None:
        api_key = authenticate_nve_hydapi()

    url = "https://hydapi.nve.no/api/v1/Stations"
    request_headers = {"Accept": "application/json", "X-API-Key": api_key}
    request = Request(url, headers=request_headers)
    response = urlopen(request)
    content = response.read().decode("utf-8")
    df = json_normalize(json.loads(content)["data"])

    df.rename(
        {"stationId": "station_id", "stationName": "station_name"},
        axis="columns",
        inplace=True,
    )

    return df


def query_nve_hydapi(
    stn_ids, par_ids, st_dt, end_dt, resolution=1440, api_key=None, series_ver=None
):
    """Query data via NVE's Hydrological API (https://hydapi.nve.no/UserDocumentation/).

    Args:
        stn_ids:    List of str. Station IDs of interest. Use
                    get_nve_hydapi_stations() to see a list of available sites
        par_ids:    List of int. Parameter IDs for NVE parameters of interest. Use
                    get_nve_hydapi_parameters() to see a list of available parameters
        st_dt:      Str. Start date (inclusive) in format "yyyy-mm-dd"
        end_dt:     Str. End date (non-inclusive) in format "yyyy-mm-dd"
        resolution: Int. One of 0 (instantenous), 60 (hourly) or 1440 (daily)
        api_key:    Str. Valid API key for HydApi. It is recommended that you DO NOT pass this
                    directly to the function. Instead, create a file containing your key and
                    store it in your HOME directory. See docstring for authenticate_nve_hydapi()
                    for further details
        series_ver: Int or None. For series with multiple versions, the desired version can be
                    specified. If None, returns the version with the most recent data.
    """
    assert dt.datetime.strptime(st_dt, "%Y-%m-%d") < dt.datetime.strptime(
        end_dt, "%Y-%m-%d"
    ), "'st_dt' must be before 'end_dt' (format 'YYYY-MM-DD')."

    # Authenticate
    if api_key is None:
        api_key = authenticate_nve_hydapi()

    baseurl = "https://hydapi.nve.no/api/v1/Observations?StationId={stns}&Parameter={pars}&ResolutionTime={resolution}&ReferenceTime={st_dt}/{end_dt}"
    if series_ver:
        baseurl += f"&VersionNumber={series_ver}"

    par_ids = [str(i) for i in par_ids]
    stns = ",".join(stn_ids)
    pars = ",".join(par_ids)

    url = baseurl.format(
        stns=stns, pars=pars, resolution=resolution, st_dt=st_dt, end_dt=end_dt
    )

    request_headers = {"Accept": "application/json", "X-API-Key": api_key}

    request = Request(url, headers=request_headers)

    response = urlopen(request)
    content = response.read().decode("utf-8")

    parsed_result = json.loads(content)

    df = json_normalize(
        parsed_result["data"],
        "observations",
        [
            "stationId",
            "stationName",
            "parameter",
            "parameterName",
            "parameterNameEng",
            "method",
            "unit",
        ],
    )
    cols = [
        "stationId",
        "stationName",
        "parameter",
        "parameterName",
        "parameterNameEng",
        "method",
        "time",
        "value",
        "unit",
        "correction",
        "quality",
    ]
    df = df[cols]
    df.rename(
        {
            "stationId": "station_id",
            "stationName": "station_name",
            "parameterName": "parameter_name",
            "parameterNameEng": "parameter_name_eng",
            "time": "datetime",
        },
        axis="columns",
        inplace=True,
    )
    df["datetime"] = pd.to_datetime(df["datetime"])

    return df


def create_point_grid(xmin, ymin, xmax, ymax, cell_size):
    """Create a uniform grid of points bounded by xmin, ymin, xmax, ymax.
    The lower-left point will be at

        (xmin + cell_size/2, ymin + cell_size/2)

    And the upper-right point will be at

        (xmax - cell_size/2, ymax - cell_size/2)
    """
    xrange = np.arange(xmin, xmax, cell_size) + (cell_size / 2)
    yrange = np.arange(ymin, ymax, cell_size) + (cell_size / 2)

    xx, yy = np.meshgrid(xrange, yrange)
    coords = np.array([xx, yy]).reshape(2, -1).T

    df = pd.DataFrame(coords, columns=["x", "y"])
    df.index.name = "point_id"
    df.reset_index(inplace=True)

    return df


def get_nve_gts_api_parameters():
    """List parameters available via NVE's GridTimeSeries API
    (http://api.nve.no/doc/gridtimeseries-data-gts/).

    Args
        None

    Returns
        Dataframe of available parameters
    """
    url = r"http://gts.nve.no/api/GridTimeSeries/Themes/json"
    request = Request(url)
    response = urlopen(request)
    content = response.read().decode("utf-8")
    df = json_normalize(json.loads(content))

    # Only return daily resolution options for now
    df = df.query("TimeResolutionInMinutes == 1440")

    return df


def get_nve_gts_api_time_series(
    loc_df,
    pars,
    st_dt,
    end_dt,
    id_col="station_code",
    xcol="longitude",
    ycol="latitude",
    crs="epsg:4326",
):
    """Get time series for the locations, parameters and time period of interest
    from NVE's GridTimeSeries API (http://api.nve.no/doc/gridtimeseries-data-gts/).

    Args
        loc_df: DataFrame. Locations of interest
        pars:   Dataframe or list. If dataframe, must be in the format returned by
                get_nve_gts_api_parameters(), filtered to the parameters of interest.
                If list, must be a list of 'str' matching valid parameter names in
                the 'Name' columne retruned by get_nve_gts_api_parameters()
        st_dt:  Str. Start date of interest 'YYYY-MM-DD'
        end_dt: Str. End date of interest 'YYYY-MM-DD'
        id_col: Str. Name of column in 'loc_df' containing a unique ID for each location
                of interest
        xcol:   Str. Name of colum in 'loc_df' containing 'eastings' (i.e. x or longitude)
        ycol:   Str. Name of colum in 'loc_df' containing 'northings' (i.e. y or latitude)
        crs:    Str. A valid co-ordinate reference system for Geopandas. Most easily
                expressed as e.g. 'epsg:4326' (for WGS84 lat/lon) or 'epsg:25833'
                (ETRS89/UTM zone 33N)

    Returns
        Dataframe of time series data.
    """
    # Validate user input
    assert len(loc_df[id_col].unique()) == len(loc_df), "ERROR: 'id_col' is not unique."

    assert dt.datetime.strptime(st_dt, "%Y-%m-%d") < dt.datetime.strptime(
        end_dt, "%Y-%m-%d"
    ), "'st_dt' must be before 'end_dt' (format 'YYYY-MM-DD')."

    par_df = get_nve_gts_api_parameters()
    if isinstance(pars, pd.DataFrame):
        pars = list(pars["Name"])
    assert set(pars).issubset(
        list(par_df["Name"])
    ), "Some parameters in 'pars' not recognised."

    orig_len = len(loc_df)
    loc_df.dropna(subset=[id_col, xcol, ycol], inplace=True)
    if len(loc_df) < orig_len:
        print(
            "WARNING: 'loc_df' contains NaN values in the 'id_col', 'xcol' or 'ycol' columns. These rows will be dropped."
        )

    # Build geodataframe and reproject to CRS reuired by API (EPSG 25833
    gdf = gpd.GeoDataFrame(
        loc_df, geometry=gpd.points_from_xy(loc_df[xcol], loc_df[ycol], crs=crs)
    )
    gdf = gdf.to_crs("epsg:25833")
    gdf["x_proj"] = gdf["geometry"].x
    gdf["y_proj"] = gdf["geometry"].y

    df_list = []
    for par in tqdm(pars, desc="Looping over parameters"):
        for idx, row in tqdm(
            gdf.iterrows(),
            total=gdf.shape[0],
            desc="Looping over grid cells",
            leave=False,
        ):
            loc_id = row[id_col]
            x = int(row["x_proj"])
            y = int(row["y_proj"])
            url = f"http://gts.nve.no/api/GridTimeSeries/{x}/{y}/{st_dt}/{end_dt}/{par}.json"
            try:
                request = Request(url)
                response = urlopen(request)
                content = json.loads(response.read().decode("utf-8"))

                par_name = content["FullName"]
                ndv = content["NoDataValue"]
                unit = content["Unit"]
                time_res = content["TimeResolution"]
                alt = content["Altitude"]
                data = content["Data"]
                ser_start = pd.to_datetime(content["StartDate"], dayfirst=True)
                ser_end = pd.to_datetime(content["EndDate"], dayfirst=True)

                # Generate range of dates
                if time_res == 1440:
                    dates = pd.date_range(ser_start, ser_end, freq="D")
                else:
                    raise ValueError("Frequency not yet implemented.")

                assert len(dates) == len(
                    data
                ), "Mismatch between length of dates and data."

                df = pd.DataFrame(
                    {
                        id_col: loc_id,
                        "x_utm_33n": x,
                        "y_utm_33n": y,
                        "altitude_m": alt,
                        "par": par,
                        "full_name": par_name,
                        "unit": unit,
                        "time_resolution": time_res,
                        "datetime": dates,
                        "value": data,
                    }
                )
                df.loc[df["value"] == ndv, "value"] = np.nan
                df_list.append(df)

            except HTTPError:
                print(f"WARNING: No data for site '{loc_id}'.")

    df = pd.concat(df_list, axis="rows")

    return df


def get_nve_gts_api_aggregated_time_series(
    poly_gdf,
    pars,
    st_dt,
    end_dt,
    id_col="station_code",
):
    """Get time series for the parameters and time period of interest, aggregated over the
    polygons in 'poly_gdf'. Data comes from NVE's GridTimeSeries API
    (http://api.nve.no/doc/gridtimeseries-data-gts/).

    Args
        poly_gdf: Geodataframe. Polygons of interest. Make sure the CRS is set and valid.
        pars:     Dataframe or list. If dataframe, must be in the format returned by
                  get_nve_gts_api_parameters(), filtered to the parameters of interest.
                  If list, must be a list of 'str' matching valid parameter names in
                  the 'Name' columne retruned by get_nve_gts_api_parameters()
        st_dt:    Str. Start date of interest 'YYYY-MM-DD'
        end_dt:   Str. End date of interest 'YYYY-MM-DD'
        id_col:   Str. Name of column in 'poly_gdf' containing a unique ID for each polygon
                  of interest

    Returns
        Dataframe of aggregated time series data for each polygon.
    """
    # Validate user input
    assert len(poly_gdf[id_col].unique()) == len(
        poly_gdf
    ), "ERROR: 'id_col' is not unique."

    assert dt.datetime.strptime(st_dt, "%Y-%m-%d") < dt.datetime.strptime(
        end_dt, "%Y-%m-%d"
    ), "'st_dt' must be before 'end_dt' (format 'YYYY-MM-DD')."

    par_df = get_nve_gts_api_parameters()
    if isinstance(pars, pd.DataFrame):
        pars = list(pars["Name"])
    assert set(pars).issubset(
        list(par_df["Name"])
    ), "Some parameters in 'pars' not recognised."

    # Reproject to CRS required by API (EPSG 25833)
    poly_gdf = poly_gdf.copy().to_crs("epsg:25833")

    # Build gdf of points at grid cell centres on a 1 km grid
    # Norway bounding box in EPSG 25833
    xmin, ymin, xmax, ymax = -80000, 6449000, 1120000, 7945000
    pt_df = create_point_grid(xmin, ymin, xmax, ymax, 1000)
    pt_gdf = gpd.GeoDataFrame(
        pt_df,
        geometry=gpd.points_from_xy(pt_df["x"], pt_df["y"], crs="epsg:25833"),
    )

    # Get just points within polys
    pt_gdf = gpd.sjoin(pt_gdf, poly_gdf, predicate="intersects")
    pt_df = pd.DataFrame(pt_gdf[[id_col, "point_id", "x", "y"]])

    # Get data for points from API
    res_df = get_nve_gts_api_time_series(
        pt_df,
        pars,
        st_dt,
        end_dt,
        id_col="point_id",
        xcol="x",
        ycol="y",
        crs="epsg:25833",
    )

    # Join poly IDs
    res_df = pd.merge(res_df, pt_df[["point_id", id_col]], how="left", on="point_id")

    # Aggregate
    res_df = (
        res_df.groupby([id_col, "par", "datetime"])
        .agg(
            {
                "altitude_m": ["mean"],
                "full_name": "first",
                "unit": "first",
                "time_resolution": "first",
                "value": ["min", "median", "max", "mean", "std", "count"],
            }
        )
        .reset_index()
    )

    res_df.columns = ["_".join(i) for i in res_df.columns.to_flat_index()]
    res_df.rename(
        {
            f"{id_col}_": id_col,
            "par_": "par",
            "datetime_": "datetime",
            "altitude_m_mean": "mean_altitude_m",
            "full_name_first": "full_name",
            "unit_first": "unit",
            "time_resolution_first": "time_resolution",
        },
        axis="columns",
        inplace=True,
    )

    if len(res_df[id_col].unique()) < len(poly_gdf[id_col].unique()):
        missing = sorted(
            list(set(poly_gdf[id_col].unique()) - set(res_df[id_col].unique()))
        )
        msg = (
            "The following catchments do not contain any grid cell centres:\n"
            f"{missing}\n"
            "Summary statistics for these catchments have not been calculated. "
            "This is a known limitation that will be fixed soon."
        )
        warnings.warn(msg)

    return res_df


def get_metno_ngcd_time_series(
    loc_df,
    par,
    st_dt,
    end_dt,
    id_col="station_code",
    xcol="longitude",
    ycol="latitude",
    crs="epsg:4326",
    ngcd_version="23.03",
    ngcd_type="type2",
):
    """Query time series from the NGCD based on points provided in a dataframe.
    Data is downloaded from Thredds here:

        https://thredds.met.no/thredds/catalog/ngcd/catalog.html

    Due to bandwidth restrictions imposed by Met, the function can be very slow.

    Args
        loc_df:       DataFrame. Locations of interest
        par:          Str. NGCD parameter of interest. One of the following:
                          'TG': daily mean temperature (K)
                          'TN': daily minimum temperature (K)
                          'TX': daily maximum temperature (K)
                          'RR': daily precipitation sum (mm)
        st_dt:        Str. Start date of interest 'YYYY-MM-DD'
        end_dt:       Str. End date of interest 'YYYY-MM-DD'
        id_col:       Str. Name of column in 'loc_df' containing a unique ID for each location
                      of interest
        xcol:         Str. Name of colum in 'loc_df' containing 'eastings' (i.e. x or longitude)
        ycol:         Str. Name of colum in 'loc_df' containing 'northings' (i.e. y or latitude)
        crs:          Str. A valid co-ordinate reference system for Geopandas. Most easily
                      expressed as e.g. 'epsg:4326' (for WGS84 lat/lon) or 'epsg:25833'
                      (ETRS89/UTM zone 33N) etc.
        ngcd_version: Str. Default '23.03'. Version of NGCD to use. See
                          https://thredds.met.no/thredds/catalog/ngcd/catalog.html
        ngcd_type:    Str. Either 'type1' or 'type2'. Default 'type2'. Interpolation method to use.
                      See
                          https://thredds.met.no/thredds/catalog/ngcd/catalog.html

    Returns
        Dataframe of time series data.
    """
    # Validate user input
    assert len(loc_df[id_col].unique()) == len(loc_df), "ERROR: 'id_col' is not unique."

    assert dt.datetime.strptime(st_dt, "%Y-%m-%d") < dt.datetime.strptime(
        end_dt, "%Y-%m-%d"
    ), "'st_dt' must be before 'end_dt' (format 'YYYY-MM-DD')."

    assert par in [
        "TG",
        "TN",
        "TX",
        "RR",
    ], "'par' must be one of ('TG', 'TN', 'TX', 'RR')."

    assert ngcd_type in [
        "type1",
        "type2",
    ], "'ngcd_type' must be either 'type1' or 'type2'."

    orig_len = len(loc_df)
    loc_df.dropna(subset=[id_col, xcol, ycol], inplace=True)
    if len(loc_df) < orig_len:
        print(
            "WARNING: 'loc_df' contains NaN values in the 'id_col', 'xcol' or 'ycol' columns. These rows will be dropped."
        )

    # Build geodataframe and reproject to CRS of NGCD
    ngcd_crs = "epsg:3035"
    gdf = gpd.GeoDataFrame(
        loc_df.copy(),
        geometry=gpd.points_from_xy(loc_df[xcol], loc_df[ycol], crs=crs),
    )
    gdf = gdf.to_crs(ngcd_crs)
    gdf["x_proj"] = gdf["geometry"].x
    gdf["y_proj"] = gdf["geometry"].y

    # Build OPENDAP URLs for period of interest
    dates = pd.date_range(st_dt, end_dt, freq="D")
    years = [date.year for date in dates]
    months = [date.month for date in dates]
    dates = [date.strftime("%Y%m%d") for date in dates]
    base_url = f"https://thredds.met.no/thredds/dodsC/ngcd/version_{ngcd_version}/{par}/{ngcd_type}/"
    urls = [
        f"{base_url}{years[idx]}/{months[idx]:02d}/NGCD_{par}_{ngcd_type}_version_{ngcd_version}_{date}.nc"
        for idx, date in enumerate(dates)
    ]

    print("Concatenating files from Thredds. This may take a while...")
    try:
        ds = xr.open_mfdataset(
            urls,
            concat_dim="time",
            combine="nested",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            parallel=True,
        )
    except OSError as e:
        if str(e).startswith("[Errno -90] NetCDF: file not found"):
            msg = (
                "\nCheck date ranges available for NGCD here: "
                "https://thredds.met.no/thredds/catalog/ngcd/catalog.html"
            )
            msg = str(e) + msg
            raise OSError(msg)
        else:
            raise

    print("Extracting values for points. This may take a while...")
    x_indexer = xr.DataArray(gdf["x_proj"], dims=[id_col], coords=[gdf[id_col]])
    y_indexer = xr.DataArray(gdf["y_proj"], dims=[id_col], coords=[gdf[id_col]])
    pts_ds = ds[par].sel(X=x_indexer, Y=y_indexer, method="nearest")
    pts_df = pts_ds.to_dataframe().reset_index()
    pts_df.rename({"time": "datetime"}, inplace=True, axis="columns")

    return pts_df


def get_metno_ngcd_aggregated_time_series(
    poly_gdf,
    par,
    st_dt,
    end_dt,
    id_col="station_code",
    ngcd_version="23.03",
    ngcd_type="type2",
):
    """Query time series from the NGCD, aggregating values over polygons provided in a
    geodataframe. Data is downloaded from Thredds here:

        https://thredds.met.no/thredds/catalog/ngcd/catalog.html

    Due to bandwidth restrictions imposed by Met, the function can be very slow.

    Args
        poly_gdf:     Geodataframe. Polygons of interest. Make sure the CRS is set and valid.
        par:          Str. NGCD parameter of interest. One of the following:
                          'TG': daily mean temperature (K)
                          'TN': daily minimum temperature (K)
                          'TX': daily maximum temperature (K)
                          'RR': daily precipitation sum (mm)
        st_dt:        Str. Start date of interest 'YYYY-MM-DD'
        end_dt:       Str. End date of interest 'YYYY-MM-DD'
        id_col:       Str. Name of column in 'poly_gdf' containing a unique ID for each location
                      of interest
        ngcd_version: Str. Default '23.03'. Version of NGCD to use. See
                          https://thredds.met.no/thredds/catalog/ngcd/catalog.html
        ngcd_type:    Str. Either 'type1' or 'type2'. Default 'type2'. Interpolation method to use.
                      See
                          https://thredds.met.no/thredds/catalog/ngcd/catalog.html

    Returns
        Dataframe of time series data.
    """
    # Validate user input
    assert len(poly_gdf[id_col].unique()) == len(
        poly_gdf
    ), "ERROR: 'id_col' is not unique."

    assert dt.datetime.strptime(st_dt, "%Y-%m-%d") < dt.datetime.strptime(
        end_dt, "%Y-%m-%d"
    ), "'st_dt' must be before 'end_dt' (format 'YYYY-MM-DD')."

    assert par in [
        "TG",
        "TN",
        "TX",
        "RR",
    ], "'par' must be one of ('TG', 'TN', 'TX', 'RR')."

    assert ngcd_type in [
        "type1",
        "type2",
    ], "'ngcd_type' must be either 'type1' or 'type2'."

    # Reproject to CRS required by API (EPSG 3035)
    poly_gdf = poly_gdf.copy().to_crs("epsg:3035")

    # Build gdf of points at grid cell centres on a 1 km grid
    # NGCD bounding box in EPSG 3035
    xmin, ymin, xmax, ymax = 4000000, 3410000, 5550000, 5430000
    pt_df = create_point_grid(xmin, ymin, xmax, ymax, 1000)
    pt_gdf = gpd.GeoDataFrame(
        pt_df,
        geometry=gpd.points_from_xy(pt_df["x"], pt_df["y"], crs="epsg:3035"),
    )

    # Get just points within polys
    pt_gdf = gpd.sjoin(pt_gdf, poly_gdf, predicate="intersects")
    pt_df = pd.DataFrame(pt_gdf[[id_col, "point_id", "x", "y"]])

    # Get data for points from API
    res_df = get_metno_ngcd_time_series(
        pt_df,
        par,
        st_dt,
        end_dt,
        id_col="point_id",
        xcol="x",
        ycol="y",
        crs="epsg:3035",
        ngcd_version=ngcd_version,
        ngcd_type=ngcd_type,
    )

    # Join poly IDs
    res_df = pd.merge(res_df, pt_df[["point_id", id_col]], how="left", on="point_id")

    # Aggregate
    res_df = (
        res_df.groupby([id_col, "datetime"])
        .agg({par: ["min", "median", "max", "mean", "std", "count"]})
        .reset_index()
    )

    res_df.columns = ["_".join(i) for i in res_df.columns.to_flat_index()]
    res_df.rename(
        {
            f"{id_col}_": id_col,
            "datetime_": "datetime",
        },
        axis="columns",
        inplace=True,
    )

    if len(res_df[id_col].unique()) < len(poly_gdf[id_col].unique()):
        missing = sorted(
            list(set(poly_gdf[id_col].unique()) - set(res_df[id_col].unique()))
        )
        msg = (
            "The following catchments do not contain any grid cell centres:\n"
            f"{missing}\n"
            "Summary statistics for these catchments have not been calculated. "
            "This is a known limitation that will be fixed soon."
        )
        warnings.warn(msg)

    return res_df


def get_era5land_cds_api_gridded(
    pars,
    st_dt,
    end_dt,
    out_nc_path,
    raw_freq="M",
    out_freq="M",
    xmin=-180,
    ymin=-90,
    xmax=180,
    ymax=90,
):
    """Download, merge and (optionally) resample ERA5-Land data from Copernicus. Note that
    getting data from Copernicus can be slow, so expect to wait hours or even days for this
    function to complete!

    ERA5-Land is a global reanalysis dataset with a spatial resolution of 0.1 x 0.1 degrees.
    The raw data has an hourly temporal resolution, but it's also available aggregated to
    monthly means.

    This function allows you to specify your parameters, time period and region of interest,
    and then download data from either the raw (hourly) or aggregated (monthly) dataset.

    The function first downloads one netCDF file per month for the specified period, then
    merges all the files into a single netCDF and saves it to disk. The separate monthly
    files are then deleted. Optionally, you can choose to up- or downsample the final netCDF
    to annual, monthly, daily or hourly resolution. Downsampling to lower frequencies is
    performed by taking the mean; upsampling to higher frequencies uses linear
    interpolation.

    NOTE: Different parameters are treated differently in the raw netCDF files. Please read
    the example notebook carefully to understand the output from this function properly.

    The minimum data period that can be requested is one full year. Note that data volumes can
    become quite large, so 'out_nc_path' must be a location on the 'shared' drive. For example,
    create a folder on 'shared' with your initials and use that.

    Documentation for the hourly dataset is here:

        https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview

    And for the monthly dataset it's here:

        https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means?tab=overview

    Args
        pars:        List of str. Parameters of interest. See the table here for an overview:
                     https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means?tab=overview
                     Note that all parameter names are lower case and spaces are replaced with
                     underscores (see the example notebook for a full list)
        st_dt:       Str. Start date of interest. 'YYYY-MM-DD'
        end_dt:      Str. End date of interest. 'YYYY-MM-DD'
        out_nc_path: Str. Path to output netCDF file to be created. Must be somewhere on 'shared'
        raw_freq:    Str. Default 'M'. Either 'M' for monthly or 'H' for hourly
        out_freq:    Str. Default 'M'. One of 'A' (annual), 'M' (monthly), 'D' (daily) or
                     'H' (hourly). Downsampling used the mean and upsampling uses linear interpolation
        xmin:        Float. Minimum x-coordinate for bounding box in WGS84 decimal degrees
        ymin:        Float. Minimum y-coordinate for bounding box in WGS84 decimal degrees
        xmax:        Float. Maximum x-coordinate for bounding box in WGS84 decimal degrees
        ymax:        Float. Maximum y-coordinate for bounding box in WGS84 decimal degrees
    """
    # Validate user input
    assert dt.datetime.strptime(st_dt, "%Y-%m-%d") < dt.datetime.strptime(
        end_dt, "%Y-%m-%d"
    ), "'st_dt' must be before 'end_dt' (format 'YYYY-MM-DD')."

    assert raw_freq in (
        "M",
        "H",
    ), "'raw_freq' must be either 'M' (monthly) or 'H' (hourly)."

    assert out_freq in (
        "A",
        "M",
        "D",
        "H",
    ), "'out_freq' must be 'A' (annual), 'M' (monthly), 'D' (daily) or 'H' (hourly)."

    assert (
        (-180 <= xmin <= 180)
        and (-180 <= xmax <= 180)
        and (-90 <= ymin <= 90)
        and (-90 <= ymax <= 90)
    ), "Bounding box co-ordinates out of range."

    assert (xmin < xmax) and (
        ymin < ymax
    ), "'xmin' and 'ymin' must be smaller than 'xmax' and 'ymax'."

    shared_path = Path("/home/jovyan/shared/common")
    child_path = Path(out_nc_path)
    assert (
        shared_path in child_path.parents
    ), "'out_nc_path' must be a folder on the 'shared/common' drive (data volumes may be large)."

    assert out_nc_path[-3:] == ".nc", "'out_nc_path' must be a .nc file."

    # Valid pars {long_name:short_name}
    instant_pars_dict = {
        "lake_mix_layer_temperature": "lmlt",
        "lake_mix_layer_depth": "lmld",
        "lake_bottom_temperature": "lblt",
        "lake_total_layer_temperature": "ltlt",
        "lake_shape_factor": "lshf",
        "lake_ice_temperature": "lict",
        "lake_ice_depth": "licd",
        "snow_cover": "snowc",
        "snow_depth": "sde",
        "snow_albedo": "asn",
        "snow_density": "rsn",
        "volumetric_soil_water_layer_1": "swvl1",
        "volumetric_soil_water_layer_2": "swvl2",
        "volumetric_soil_water_layer_3": "swvl3",
        "volumetric_soil_water_layer_4": "swvl4",
        "leaf_area_index_low_vegetation": "lai_lv",
        "leaf_area_index_high_vegetation": "lai_hv",
        "surface_pressure": "sp",
        "soil_temperature_level_1": "stl1",
        "snow_depth_water_equivalent": "sd",
        "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10",
        "2m_temperature": "2t",
        "2m_dewpoint_temperature": "2d",
        "soil_temperature_level_2": "stl2",
        "soil_temperature_level_3": "stl3",
        "skin_reservoir_content": "src",
        "skin_temperature": "skt",
        "soil_temperature_level_4": "stl4",
        "temperature_of_snow_layer": "tsn",
        "forecast_albedo": "fal",
    }
    accum_pars_dict = {
        "surface_runoff": "sro",
        "sub_surface_runoff": "ssro",
        "snowmelt": "smlt",
        "snowfall": "sf",
        "surface_sensible_heat_flux": "sshf",
        "surface_latent_heat_flux": "slhf",
        "surface_solar_radiation_downwards": "ssrd",
        "surface_thermal_radiation_downwards": "strd",
        "surface_net_solar_radiation": "ssr",
        "surface_net_thermal_radiation": "str",
        "total_evaporation": "e",
        "runoff": "ro",
        "total_precipitation": "tp",
        "evaporation_from_the_top_of_canopy": "evatc",
        "evaporation_from_bare_soil": "evabs",
        "evaporation_from_open_water_surfaces_excluding_oceans": "evaow",
        "evaporation_from_vegetation_transpiration": "evavt",
        "potential_evaporation": "pev",
    }
    all_pars = list(instant_pars_dict.keys()) + list(accum_pars_dict.keys())
    assert set(pars).issubset(
        set(all_pars)
    ), "Some of the parameters in 'pars' are not valid."

    # Create subfolder in output directory to store temporary files
    out_fold = os.path.split(out_nc_path)[0]
    temp_fold = os.path.join(out_fold, "era5land_temp")
    if not os.path.exists(temp_fold):
        os.makedirs(temp_fold)

    print("Getting data from Copernicus. This may take a while...")
    time.sleep(0.5)
    c = cdsapi.Client()

    if raw_freq == "M":
        dataset = "reanalysis-era5-land-monthly-means"
    else:
        dataset = "reanalysis-era5-land"

    # Query one month at a time to avoid being penalised in the CDS API queue
    st_yr = int(st_dt[:4])
    end_yr = int(end_dt[:4])
    for year in range(st_yr, end_yr + 1):
        for month in range(1, 13):
            request_dict = {
                "format": "netcdf",
                "variable": pars,
                "year": str(year),
                "month": f"{month:02d}",
                "area": [ymax, xmin, ymin, xmax],
            }

            if raw_freq == "M":
                request_dict["product_type"] = "monthly_averaged_reanalysis"
                request_dict["time"] = "00:00"
                temp_path = os.path.join(
                    temp_fold, f"era5land_raw_monthly_{year}-{month:02d}.nc"
                )
            else:
                request_dict["day"] = [f"{day:02d}" for day in range(1, 32)]
                request_dict["time"] = [f"{hour:02d}:00" for hour in range(0, 24)]
                temp_path = os.path.join(
                    temp_fold, f"era5land_raw_hourly_{year}-{month:02d}.nc"
                )

            # Get data
            c.retrieve(dataset, request_dict, temp_path)

    # Combine datasets
    print("Processing downloaded datasets...")
    nc_paths = os.path.join(temp_fold, "*.nc")
    with xr.open_mfdataset(nc_paths, combine="by_coords") as ds:
        # Resample to desired frequency
        if raw_freq == "M":
            if out_freq == "A":
                # Downsampling
                ds = ds.resample({"time": out_freq}, skipna=True).mean()
            elif out_freq == "M":
                pass
            else:
                # Upsampling to hourly or daily
                ds = ds.resample({"time": out_freq}).interpolate("linear")
        else:
            # Raw is hourly
            if out_freq == "H":
                pass
            else:
                # Downsampling
                # 'Accumulated' parameters are represpended strangely in ERA5-Land. See
                # https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation#ERA5Land:datadocumentation-Temporalfrequency
                # For these parameters, daily totals for day X are the values for hour 00:00
                # on day (X+1)
                for par in accum_pars_dict.values():
                    if par in ds.keys():
                        # Get just the results for 00:00, setting all others to NaN
                        ds[par] = ds[par].where(ds["time"].dt.hour == 0).shift(time=-1)
                ds = ds.resample({"time": out_freq}, skipna=True).mean()

        # # Tidy variables
        # for var_name in ds.variables:
        #     if var_name[-4:] == '0005':
        #         # Not needed
        #         del ds[var_name]
        #     elif var_name[-4:] == '0001':
        #         # Recent data (2019 only). Merge with other series
        #         var, idx = var_name.split('_')
        #         ds[var] = ds[var].combine_first(ds[var_name])
        #         del ds[var_name]

        ds.load()

    # Save and tidy
    ds.to_netcdf(out_nc_path)
    shutil.rmtree(temp_fold)

    return ds


def get_era5land_cds_api_time_series(
    loc_df,
    nc_path,
    st_dt,
    end_dt,
    id_col="station_code",
    xcol="longitude",
    ycol="latitude",
    crs="epsg:4326",
):
    """Query time series from ERA5-Land data based on points provided in a dataframe.
    Data should first be downloaded and merged using
    nivapy.da.get_era5land_cds_api_gridded().

    Args
        loc_df:       DataFrame. Locations of interest
        nc_path:      Str. Path to netCDF of gridded data
        st_dt:        Str. Start date of interest 'YYYY-MM-DD'
        end_dt:       Str. End date of interest 'YYYY-MM-DD'
        id_col:       Str. Name of column in 'loc_df' containing a unique ID for each location
                      of interest
        xcol:         Str. Name of colum in 'loc_df' containing 'eastings' (i.e. x or longitude)
        ycol:         Str. Name of colum in 'loc_df' containing 'northings' (i.e. y or latitude)
        crs:          Str. A valid co-ordinate reference system for Geopandas. Most easily
                      expressed as e.g. 'epsg:4326' (for WGS84 lat/lon) or 'epsg:25833'
                      (ETRS89/UTM zone 33N) etc.

    Returns
        Dataframe of time series data.
    """
    # Validate user input
    assert len(loc_df[id_col].unique()) == len(loc_df), "ERROR: 'id_col' is not unique."

    assert dt.datetime.strptime(st_dt, "%Y-%m-%d") < dt.datetime.strptime(
        end_dt, "%Y-%m-%d"
    ), "'st_dt' must be before 'end_dt' (format 'YYYY-MM-DD')."

    assert nc_path[-3:] == ".nc", "'nc_path' must be a netCDF file."

    orig_len = len(loc_df)
    loc_df.dropna(subset=[id_col, xcol, ycol], inplace=True)
    if len(loc_df) < orig_len:
        print(
            "WARNING: 'loc_df' contains NaN values in the 'id_col', 'xcol' or 'ycol' columns. These rows will be dropped."
        )

    # Build geodataframe and reproject to CRS of ERA5-Land
    era5_crs = "epsg:4326"
    gdf = gpd.GeoDataFrame(
        loc_df.copy(),
        geometry=gpd.points_from_xy(loc_df[xcol], loc_df[ycol], crs=crs),
    )
    gdf = gdf.to_crs(era5_crs)
    gdf["x_proj"] = gdf["geometry"].x
    gdf["y_proj"] = gdf["geometry"].y

    ds = xr.open_dataset(nc_path)

    # Extract values for points
    x_indexer = xr.DataArray(gdf["x_proj"], dims=[id_col], coords=[gdf[id_col]])
    y_indexer = xr.DataArray(gdf["y_proj"], dims=[id_col], coords=[gdf[id_col]])
    pts_ds = ds.sel(longitude=x_indexer, latitude=y_indexer, method="nearest")
    pts_df = pts_ds.to_dataframe().reset_index()
    pts_df.rename({"time": "datetime"}, inplace=True, axis="columns")

    return pts_df


def get_era5land_cds_api_aggregated_time_series(
    poly_gdf,
    nc_path,
    st_dt,
    end_dt,
    id_col="station_code",
):
    """Query time series from ERA5-Land, aggregating values over polygons provided in a
    geodataframe. Data should first be downloaded and merged using
    nivapy.da.get_era5land_cds_api_gridded().

    Args
        poly_gdf:     Geodataframe. Polygons of interest. Make sure the CRS is set and valid.
        nc_path:      Str. Path to netCDF of gridded data
        st_dt:        Str. Start date of interest 'YYYY-MM-DD'
        end_dt:       Str. End date of interest 'YYYY-MM-DD'
        id_col:       Str. Name of column in 'poly_gdf' containing a unique ID for each location
                      of interest

    Returns
        Dataframe of time series data.
    """
    # Validate user input
    assert len(poly_gdf[id_col].unique()) == len(
        poly_gdf
    ), "ERROR: 'id_col' is not unique."

    assert nc_path[-3:] == ".nc", "'nc_path' must be a netCDF file."

    assert dt.datetime.strptime(st_dt, "%Y-%m-%d") < dt.datetime.strptime(
        end_dt, "%Y-%m-%d"
    ), "'st_dt' must be before 'end_dt' (format 'YYYY-MM-DD')."

    # Reproject to CRS of ERA5-Land (EPSG 4326)
    era5_crs = "epsg:4326"
    poly_gdf = poly_gdf.copy().to_crs(era5_crs)

    # Build gdf of points at grid cell centres on a 0.1 deg grid
    # World bounding box in EPSG 4326
    xmin, ymin, xmax, ymax = -180, -90, 180, 90
    pt_df = create_point_grid(xmin, ymin, xmax, ymax, 0.1)
    pt_gdf = gpd.GeoDataFrame(
        pt_df,
        geometry=gpd.points_from_xy(pt_df["x"], pt_df["y"], crs=era5_crs),
    )

    # Get just points within polys
    pt_gdf = gpd.sjoin(pt_gdf, poly_gdf, predicate="intersects")
    pt_df = pd.DataFrame(pt_gdf[[id_col, "point_id", "x", "y"]])

    # Get data for points
    res_df = get_era5land_cds_api_time_series(
        pt_df,
        nc_path,
        st_dt,
        end_dt,
        id_col="point_id",
        xcol="x",
        ycol="y",
        crs=era5_crs,
    )

    # Join poly IDs
    res_df = pd.merge(res_df, pt_df[["point_id", id_col]], how="left", on="point_id")

    # Get list of vars in nc file
    with xr.open_dataset(nc_path) as ds:
        pars = list(ds.keys())

    # Aggregate
    stat_dict = {par: ["min", "median", "max", "mean", "std", "count"] for par in pars}
    res_df = res_df.groupby([id_col, "datetime"]).agg(stat_dict).reset_index()

    # Tidy
    res_df = res_df.melt(id_vars=[id_col, "datetime"], var_name=["par", "stat"])
    res_df = (
        res_df.set_index([id_col, "par", "datetime", "stat"])
        .unstack("stat")
        .reset_index()
    )
    res_df.index.name = ""
    res_df.columns = ["_".join(i) for i in res_df.columns.to_flat_index()]
    res_df.rename(
        {
            f"{id_col}_": id_col,
            "datetime_": "datetime",
            "par_": "par",
        },
        axis="columns",
        inplace=True,
    )
    res_df = res_df[
        [
            id_col,
            "par",
            "datetime",
            "value_min",
            "value_median",
            "value_max",
            "value_mean",
            "value_std",
            "value_count",
        ]
    ]

    if len(res_df[id_col].unique()) < len(poly_gdf[id_col].unique()):
        missing = sorted(
            list(set(poly_gdf[id_col].unique()) - set(res_df[id_col].unique()))
        )
        msg = (
            "The following catchments do not contain any grid cell centres:\n"
            f"{missing}\n"
            "Summary statistics for these catchments have not been calculated. "
            "This is a known limitation that will be fixed soon."
        )
        warnings.warn(msg)

    return res_df


def get_data_from_vannmiljo(endpoint, params=None):
    """Vannmiljø endpoints are documented here:

        https://vannmiljowebapi.miljodirektoratet.no/swagger/ui/index#!

    This function can be used with any of the GET endpoints.

    Args
        endpoint: Str. Any of the GET endpoints listed for the API (e.g.
                  'GetMediumList')
        params:   Dict. Valid parameters for the chosen endpoint - see the
                  Swagger UI for details. E.g. {'filter': 'BB'} for the
                  'GetMediumList' endpoint

    Returns
        Dataframe.
    """
    headers = {
        "Content-Type": "application/json",
    }
    url = f"https://vannmiljowebapi.miljodirektoratet.no/api/Public/{endpoint}"
    response = requests.get(url, headers=headers, params=params)

    return pd.DataFrame(response.json())


def get_vannmiljo_api_key():
    """Get API key for POSTing to Vannmiljø."""
    fpath = "/home/jovyan/shared/common/01_datasets/tokens/vannmiljo_api_token.json"
    with open(fpath) as f:
        data = json.loads(f.read())

    return data["VANNMILJO_API_KEY"]


def post_data_to_vannmiljo(endpoint, data=None):
    """Vannmiljø endpoints are documented here:

        https://vannmiljowebapi.miljodirektoratet.no/swagger/ui/index#!

    This function can be used with any of the POST endpoints.

    Args
        endpoint: Str. Any of the POST endpoints listed for the API (e.g.
                  'GetRegistrations')
        data:     Dict. Valid parameters for the chosen endpoint - see the
                  Swagger UI for details. E.g.
                      {'WaterLocationCodeFilter': ['002-58798']}
                  for the 'GetRegistrations' endpoint

    Returns
        Dataframe.
    """
    headers = {
        "vannmiljoWebAPIKey": get_vannmiljo_api_key(),
        "Content-Type": "application/json",
    }
    url = f"https://vannmiljowebapi.miljodirektoratet.no/api/Public/{endpoint}"
    response = requests.post(url, headers=headers, data=json.dumps(data))

    return pd.DataFrame(response.json()["Result"])


def get_data_from_vannnett(wb_id, quality_element):
    """
    Get water quality data from the vann-nett.

    Parameters
        wb_id: Str. The waterbody ID.
        quality_element: Str. The quality element to fetch. Must be one of ['ecological',
            'rbsp', 'swchemical'].

    Returns
        DataFrame of water quality data.
    """
    valid_elements = ["ecological", "rbsp", "swchemical"]
    if quality_element.lower() not in valid_elements:
        raise ValueError(
            "'quality_element' must be one of ['ecological', 'rbsp', 'swchemical']."
        )

    element_dict = {
        "ecological": "ecological",
        "rbsp": "RBSP",
        "swchemical": "swChemical",
    }
    quality_element = element_dict[quality_element.lower()]

    url = f"https://vann-nett.no/service/waterbodies/{wb_id}/qualityElements/{quality_element}"
    response = requests.get(url)
    if response.status_code != 200:
        response.raise_for_status()
    data = response.json()

    par_map = {
        "qualityElementType.parentId": "category",
        "qualityElementType.id": "element",
        "parameterType.text": "parameter",
        "status.text": "status",
        "eqr": "eqr",
        "neqr": "neqr",
        "value": "value",
        "threshold.refValue": "reference_value",
        "threshold.unit": "unit",
        "threshold.statusLimits": "status_limits",
        "yearFrom": "year_from",
        "yearTo": "year_to",
        "sampleCount": "sample_count",
        "otherSource": "source",
        "dataQuality.text": "data_quality",
    }
    par_cols = par_map.keys()
    df_list = []
    cat_data = pd.json_normalize(data)
    for cat_row in cat_data.itertuples():
        ele_data = pd.json_normalize(cat_row.qualityElements)
        for ele_row in ele_data.itertuples():
            par_df = pd.json_normalize(ele_row.parameters)
            if not par_df.empty:
                par_df = par_df[par_cols].rename(columns=par_map)
                df_list.append(par_df)

    df = pd.concat(df_list, ignore_index=True)

    return df