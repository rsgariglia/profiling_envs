"""Performing EDA and dealing with missing values. Outputs and images are stored in the output folder."""
from cleaning import merge_and_process
from cleaning import create_nan
from clustering import pca_preprocess
from clustering import train_km_model
from clustering import train_lgbm_model
from analysis.getters.get_data import get_clean_dataset
from sklearn.preprocessing import MinMaxScaler
from matplotlib import rcParams
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt


alt.renderers.enable("mimetype")
alt.data_transformers.disable_max_rows()

def kde_plot(final):
    """Generate kde plot for variables in the dataset."""
    final_cols = [
        "median_house_price_2019",
        "gpp_dist",
        "ed_dist",
        "dent_dist",
        "pharm_dist",
        "gamb_dist",
        "ffood_dist",
        "pubs_dist",
        "leis_dist",
        "blue_dist",
        "off_dist",
        "tobac_dist",
        "green_pas",
        "green_act",
        "no2_mean",
        "pm10_mean",
        "so2_mean",
        "Years of potential life lost indicator",
        "Comparative illness and disability ratio indicator",
        "Acute morbidity indicator",
        "Mood and anxiety disorders indicator",
        "reception_obese_%",
        "year6_obese_%",
    ]
    fig, ax = plt.subplots(5, 5, figsize=(35, 25))
    for variable, subplot in zip(final_cols, ax.flatten()):
        sns.histplot(createnan(final[variable]), ax=subplot, kde=True, linewidth=0)
    plt.subplots_adjust(hspace=1.2, wspace=0.3)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


def qq_plot(final):
    """Generate qq plot for continuous numerical variables"""
    fig, axes = plt.subplots(ncols=5, nrows=2, sharex=True, figsize=(20, 10))
    for k, ax in zip(
        [print(item) for item in final.columns if final[item].dtype == "float64"][0],
        np.ravel(axes),
    ):
        qqplot(final[k], line="s", ax=ax)
        plt.subplots_adjust(hspace=1.5, wspace=1)
        ax.set_title(f"{k} QQ Plot")
        plt.tight_layout()


def box_plot(final):
    """Generate box plots of some quantitative variables using the urban/rural indicator 'ur' as category"""
    f, axes = plt.subplots(5, 5, figsize=(30, 20))
    ci = 0
    for i in range(0, 5):
        for j in range(0, 5):
        sns.boxplot(ax=axes[i, j], data=final, x=final.ur, y=quant_cols[ci])
        ci += 1;

    

def gen_pairplot(final):
    """Generate pairplot of house prices, obesity prevalence and nutrients"""
    sns.pairplot(final[['median_house_price_2019','Years of potential life lost indicator', 'Comparative illness and disability ratio indicator', 'Acute morbidity indicator',
       'Mood and anxiety disorders indicator', 'reception_obese_%','year6_obese_%', 'ur']], hue="ur");


def gen_heatmap(final):
    """Generate correlation matrix and heatmap"""
    plt.figure(figsize=(20, 10))
    heatmap = sns.heatmap(final.corr(), mask= mask, vmin=-1, vmax=1, annot=True,cmap='BrBG')
    heatmap.set_title('Environment Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
    plt.savefig('heatmap.png', dpi=300, bbox_inches='tight');
    

def corr_matrix_obeseyear6(final):
    "Generate correlations between variables and year 6 obesity prevalence. Display correlations on a heatmap, in descending order"
    plt.figure(figsize=(10, 12))
    heatmap = sns.heatmap(final.corr()[['year6_obese_%']].sort_values(by='year6_obese_%', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Features Correlating with year 6 obesity', fontdict={'fontsize':18}, pad=16)
    plt.savefig('features_obesity.png', dpi=300, bbox_inches='tight');


def plot_obesity_imd(get_clean_dataset):
    """Plot comparative illness indicator against obesity, with years lost as hue"""
    alt.Chart(dataset).mark_point().encode(x='Comparative illness and disability ratio indicator', y='year6_obese_%', color='Years_lost:N').interactive()


def plot_pca(df):
    """Plot loadings of the first 2 PCA components using Altair"""
    df_pca_2d = pca_preprocess(df)
    pca1 = alt.Chart(df_pca_2d.reset_index()).mark_bar().encode(
    y=alt.Y('index:O', title=None),
    x='pca1',
    color=alt.Color('pca1', scale=alt.Scale(scheme='viridis')),
    tooltip = [alt.Tooltip('index', title='Feature'), alt.Tooltip('pca1', format='.2f')]
    )
    pca2 = alt.Chart(df_pca_2d.reset_index()).mark_bar().encode(
    y=alt.Y('index:O', title=None),
    x='pca2',
    color=alt.Color('pca2', scale=alt.Scale(scheme='viridis')),
    tooltip = [alt.Tooltip('index', title='Feature'), alt.Tooltip('pca2', format='.2f')]
    )

    (pca1 & pca2).properties(
        title='Loadings of the first two principal components'
        )



def plot_km_clusters(dataset_scaled):
    """Plot the 6 clusters we obtained by training the Kmeans algorithm"""
    df_km = train_km_model(dataset_scaled)
    domain = [0, 1, 2, 3, 4, 5]
    range_ = ['pink',  'lightblue', 'green', 'black', 'orange', 'yellow']

    selection = alt.selection_multi(fields=['cluster'], bind='legend')

    pca = alt.Chart(df_km).mark_circle(size=20).encode(
        x='pca1',
        y='pca2',
        color=alt.Color('cluster:N', scale=alt.Scale(domain=domain, range=range_)),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
        tooltip = list(all_features)
        ).add_selection(
        selection
        )
    return pca



def shap_summary_plot(X):
    """Calculate SHAP values and generate SHAP summary plot"""
    train_lgbm_model(X)
    explainer = shap.TreeExplainer(mdl)
    shap_values = explainer.shap_values(X)
    return shap.summary_plot(shap_values, X, max_display=23, class_names=list(range(0,6)));


def beeswarm_plots_shap(X):
    """Calculate beeswarm plots for different clusters"""
    for cnr in range(0,6):
        shap.summary_plot(shap_values[cnr], X, max_display=20, show=False)
        plt.title(f'Cluster {cnr}');


def between_cluster_var_plot(df):
    """Calculate between cluster variance and plot"""
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), index= df.index, columns=df.columns)
    km = train_km_model(df_scaled)
    df_scaled['km'] = km.labels_
    df_mean = (df_scaled.groupby('km').mean())
    plt.style.use('ggplot')
    results_var = pd.DataFrame(columns=['Variable', 'Var'])
    for column in df_mean.columns[1:]:
        results_var.loc[len(results_var), :] = [column, np.var(df_mean[column])]
        selected_columns = list(results_var.sort_values(
            'Var', ascending=False,
            ).head(7).Variable.values) + ['km']
    tidy = df_scaled[selected_columns].melt(id_vars='km')
    rcParams['figure.figsize'] = 13,10
    sns.barplot(x='km', y='value', hue='variable', data=tidy)
    plt.title('Variables with highest between cluster variance');


def between_cluster_var_weighted(mdl, df):
    """Calculate between cluster variance weighted by feature importance"""
    data = np.array([mdl.feature_importances_, X.columns]).T
    columns = list(pd.DataFrame(data, columns=['Importance', 'Feature'])
    .sort_values("Importance", ascending=False)
    .head(5).Feature.values)
    tidy = df_scaled[columns+['km']].melt(id_vars='km')
    sns.barplot(x='km', y='value', hue='variable', data=tidy)
    plt.title('Variables with highest between cluster variance weighted by feature importance');


def plot_imd_obese_clusters(df_scaled):
    """Plot deprivation indicators and obesity level in each cluster using scaled data for comparison"""
    return df_scaled.groupby('km')[['year6_obese_%', 'reception_obese_%', 'Years of potential life lost indicator','Mood and anxiety disorders indicator']].mean().plot.bar(title='Deprivation and obesity across clusters');



