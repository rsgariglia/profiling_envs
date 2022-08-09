"""Visualising the result of the clustering exercise for London."""

import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import geojson
import json
import altair as alt
from getters.get_data import get_london_boundaries, get_lsoa_geojson
from utils.file_management import save_pickle, load_pickle


def nestafont():
    font = "Averta"

    return {
        "config": {
            "title": {"font": font},
            "axis": {"labelFont": font, "titleFont": font},
            "header": {"labelFont": font, "titleFont": font},
            "legend": {"labelFont": font, "titleFont": font},
        }
    }


alt.themes.register("nestafont", nestafont)
alt.themes.enable("nestafont")

range_ = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33"]
domain = [
    "High deprivation, high obesity",
    "High quality of life, low obesity",
    "Less urban, less obese",
    "Affordable but stressful",
    "Greener suburbs",
    "Low deprivation, high obesity",
]


def create_london_data(final):
    """Read in London shp file and merge with exisiting dataset"""
    london = get_london_boundaries()
    london = pd.merge(
        final.iloc[:, 0:-5],
        london[["LSOA11CD", "geometry"]],
        left_on="LSOA code",
        right_on="LSOA11CD",
        how="left",
        validate="one_to_one",
    )
    save_pickle(london, "london_boundaries.csv")
    london_final = GeoDataFrame(
        london[
            [
                "year6_obese_%",
                "Years of potential life lost indicator",
                "clusters",
                "LSOA code",
                "geometry",
            ]
        ]
    )
    return london_final


def preprocess_data_map(get_lsoa_geojson, london_final):
    """Create base layer with geojson file, then convert to geopandas dataframe."""
    get_lsoa_geojson()
    with open('lsoa_uk.geojson') as f:
    gj = geojson.load(f)
    final = gpd.GeoDataFrame.from_features(gj).merge(london_final, on='LSOA11CD', how='inner')
    choro_json = json.loads(final.to_json())
    choro_data = alt.Data(values=choro_json['features'])
    return choro_data


def gen_map(geodata, color_column, title):
    """Generates an interactive Altair plot with a base layer of the LSOA boundaries for London and clusters"""
    # Add Base Layer
    base = alt.Chart(geodata, title = title).mark_geoshape(
        stroke='black',
        strokeWidth=0.5
    ).encode(
    ).properties(
        width=1000,
        height=1000
    )
    # Add Choropleth Layer
    choro = alt.Chart(geodata).mark_geoshape(
        #fill='lightgray',
        stroke='black'
    ).encode(
        alt.Color(color_column, 
                  #type='quantitative', 
                  scale=alt.Scale(range=range_),
                  title = "London clusters"),
        tooltip=[alt.Tooltip('properties.LSOA11NM:N', title="LSOA name"), 
        alt.Tooltip("properties.year6_obese_%:Q", title= "Year 6 obese (%)", format="1.2f")]
    )
    return base + choro


cluster_london_map = gen_map(geodata=choro_data, color_column='properties.clusters:N', title='London clusters')
cluster_london_map.save('london_clusters.html', scale_factor=1.5)