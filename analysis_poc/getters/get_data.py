"""Data getters for the Profiling environments work. The data, including the final dataset is stored in an S3 bucket.

See readme for data sources
"""
import pandas as pd
import geopandas as gpd
from pandas_ods_reader import read_ods
from analysis import PROJECT_DIR

PROFILE_DIR = PROJECT_DIR / "getters/"


def get_haha() -> pd.DataFrame:
    """Returns dataframe of Healthy Assets and Hazards scores for LSOAs"""
    return pd.read_csv(PROFILE_DIR / "allvariableslsoawdeciles.csv")


def get_houseprice() -> pd.DataFrame:
    """Returns dataframe of median house prices sold per LSOA"""
    return pd.read_excel(
        PROFILE_DIR / "hpssadataset46medianpricepaidforresidentialpropertiesbylsoa1.xls"
    )


def get_imd() -> pd.DataFrame:
    """Returns dataframe of English IMD scores for the Health pillar for England for 2019"""
    return pd.read_excel(
        PROFILE_DIR / "IMD_England_2019_Underlying_Indicators.xlsx",
        sheet_name="IoD2019 Health Domain",
    )


def get_ncmp_reception() -> pd.DataFrame:
    """Returns dataframe of Obesity prevalence per LSOA in England for reception children in 2019"""
    return pd.read_ods(PROFILE_DIR / "NCMP_data_Ward_update_2019.ods", sheet=3)[2:]


def get_ward_lookup() -> pd.DataFrame:
    """Returns ward lookup for LSOA conversion"""
    return pd.read_ods(
        PROFILE_DIR
        / "Lower_Layer_Super_Output_Area_(2011)_to_Ward_(2020)_to_LAD_(2020)_Lookup_in_England_and_Wales_V2.csv"
    )


def get_ncmp_year6() -> pd.DataFrame:
    """Returns dataframe of Obesity prevalence per LSOA in England for year 6 children in 2019"""
    return pd.read_ods(PROFILE_DIR / "NCMP_data_Ward_update_2019.ods", sheet=5)[2:]


def get_clean_dataset() -> pd.DataFrame:
    """Load in clean dataset from S3"""
    return pd.read_csv(PROFILE_DIR / "clean_dataset_profiling_envs.csv")


def get_london_boundaries() -> pd.DataFrame:
    """Load in dataset of London geographical boundaries from S3"""
    return gpd.read_csv(PROFILE_DIR / "GIS_London/LSOA_2011_London_gen_MHW.shp")


def get_lsoa_geojson() -> pd.DataFrame:
    """Load in dataset of geojson uk boundaries for lsoas"""
    return gpd.read_csv(PROFILE_DIR / "lsoa_uk.geojson")
