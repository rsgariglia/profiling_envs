"""Cleaning and merging the imported datasets for use in EDA and clustering """

from analysis.getters.get_data import (
    get_haha,
    get_houseprice,
    get_imd,
    get_ncmp_year6,
    get_ncmp_reception,
    get_ward_lookup,
)

import pandas as pd



def clean_imd(df):
    """Delete irrelevant columns for English IMD dataset"""
    df_imd = df[
        [
            "LSOA code (2011)",
            "LSOA name (2011)",
            "Local Authority District code (2019)",
            "Local Authority District name (2019)",
            "Years of potential life lost indicator",
            "Comparative illness and disability ratio indicator",
            "Acute morbidity indicator",
            "Mood and anxiety disorders indicator",
        ]
    ]
    return df_imd


def clean_housep(get_houseprice(df)):
    """Keep only most recent median house price observation"""
    df_house = df[["LSOA code", "Year ending Jun 2021"]]
    return df_house


def clean_haha(get_haha(df)):
    """Remove unnecessary columns HAHA"""
    df.drop(['d_gpp_dist', 'd_ed_dist', 'd_pharm_dist', 'd_dent_dist', 'd_gamb_dist', 'd_ffood_dist',
           'd_pubs_dist', 'd_leis_dist', 'd_blue_dist', 'd_off_dist', 'd_tobac_dist', 'd_green_pas',
           'd_green_act', 'd_no2_mean', 'd_pm10_mean', 'd_so2_mean'],axis = 1,inplace=True)
    return df_haha



def clean_ncmp_reception(get_ncmp_reception(df1)):
    """Fix column names and row structure of reception NCMP datasets"""
    df.reset_index(inplace=True, drop=True)
    df.columns = df.iloc[0]
    df = df.iloc[1:, [0, 41]].reset_index(drop=True)
    df.rename(columns={"%": "reception_obese_%"}, inplace=True)
    return df1


def clean_ncmp_year6(get_ncmp_year6(df2)):
    """Fix column names and row structure of year6 NCMP datasets"""
    df.reset_index(inplace=True, drop=True)
    df.columns = df.iloc[0]
    df = df.iloc[1:, [0, 41]].reset_index(drop=True)
    df.rename(columns={"%": "year6_obese_%"}, inplace=True)
    return df2


def merge_ncmp_clean(df1, df2, get_ward_lookup(df)):
    """Merge the two NCMP dataset and use LSOA/Ward lookup to get LSOA level prevalence"""
    obese_ncmp= (
        pd.merge(
            clean_ncmp_reception(df1),
            clean_ncmp_year6(df2),
            on="Ward code",
            validate="1:1",
        )
        .merge(
            obese_ncmp,
            get_ward_lookup(df),
            how="left",
            left_on="Ward code",
            right_on="WD20CD",
            validate="1:m",
        )
        .dropna(subset=["LSOA11CD"], inplace=True)
    )
    obese_ncmp_test = obese_ncmp_test[
        ["reception_obese_%", "year6_obese_%", "LSOA11CD"]
    ]
    return obese_ncmp


def merge_and_process(df_house, df_haha, df_imd):
    """Join datasets and perform cleaning actions to remove duplicate column variables."""
    final = pd.merge(
        df_house, df_haha, left_on="LSOA code", right_on="lsoa11", indicator=True, validate="1:1"
    ).drop(["rownum", "_merge"], axis=1)
    final.rename(
        columns={"Year ending Jun 2021": "median_house_price_2019"}, inplace=True
    )
    final = pd.merge(
        final[final["LSOA code"].str.startswith("W") == False],
        pd.DataFrame(merge_ncmp_clean(df1, df2, get_ward_lookup(df))),
        how="left",
        left_on="LSOA code",
        right_on="LSOA11CD",
        validate="1:1",
    )
    final = pd.merge(
        final,
        clean_imd(df_imd),
        left_on="LSOA code",
        right_on="LSOA code (2011)",
        indicator="exists",
        validate="1:1",
    )
    final.drop(
        [
            "lsoa11",
            "LSOA code (2011)",
            "LSOA name (2011)",
            "Local Authority District code (2019)",
            "LSOA11CD",
            "exists"
        ],
        axis=1,
        inplace=True,
    )
    return final

def create_nan(df):
    """Replace ':' with np.nan"""
    df = df.replace([':', 'NA'], np.nan)
    return df

