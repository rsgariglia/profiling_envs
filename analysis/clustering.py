"Performing clustering using K-means"


from analysis.utils.file_management import save_pickle
from getters.get_data import get_clean_dataset
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



# pickle model so that it can be used in the SHAP section

def pca_preprocess(get_clean_dataset()):
    """Carry out scaling and pre-processing to visualise PCA components"""
    scaler = StandardScaler() 
    dataset_scaled = pd.DataFrame(scaler.fit_transform(get_clean_dataset()))
    pca_2d = PCA(n_components=2)
    dataset_pca = pca_2d.fit_transform(dataset_scaled)
    return pd.DataFrame(data=pca_2d.components_.T, columns=['pca1', 'pca2'], index=dataset_scaled.columns[:-5])


def get_silhouette_score(dataset_scaled):
    """Calculate silhouette score in a loop for a range of 3-10 clusters and fit Kmeans on scaled data"""
    km_scores= []
    km_silhouette = []

    for i in range(3,10):
        km = KMeans(n_clusters=i, random_state=1301).fit(dataset_scaled)
        preds = km.predict(dataset_scaled)
    
        print(f'Score for number of cluster(s) {i}: {km.score(dataset_scaled):.3f}')
        km_scores.append(-km.score(dataset_scaled))
    
        silhouette = silhouette_score(dataset_scaled,preds)
        km_silhouette.append(silhouette)
        print(f'Silhouette score for number of cluster(s) {i}: {silhouette:.3f}')
    
        print('-'*100)


def train_km_model(dataset_scaled):
    """Fit kmeans model to scaled dataset"""
    km = KMeans(n_clusters=6, random_state=1301).fit(dataset_scaled)
    preds = km.predict(dataset_scaled)
    df_km = pd.DataFrame(data={'pca1':dataset_pca[:,0], 'pca2':dataset_pca [:,1], 'cluster':preds})
    return df_km, km


def train_lgbm_model(df):
    """Train a light gradient booster model with the cluster label as output variable."""
    X = train_km_model(get_clean_dataset(df)).drop(columns=['pca1', 'pca2', 'member', 'years_lost_weight', 'Years_lost'])

    params['objective'] = 'multiclass' 
    params['is_unbalance'] = True
    params['n_jobs'] = -1
    params['random_state'] = 1301

    mdl = lgb.LGBMClassifier(**params)
    mdl.fit(X, preds)
    y_pred_km = mdl.predict_proba(X)
    return y_pred_km, mdl



