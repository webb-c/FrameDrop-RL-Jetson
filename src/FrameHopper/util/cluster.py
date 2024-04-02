
"""
clustering for 3-dimension continuous state to (10,3) discrete state 
"""
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans


def get_variant_state(model, d1, d2):
    pass


def get_state(model, diff):
    s = model.predict([diff])
    return s


def init_cluster(state_num=10):
    model = MiniBatchKMeans(n_clusters=state_num, n_init=10, batch_size=30, random_state=42)
    return model


def train_cluster(model, data, visualize=False):
    model.partial_fit(data)

    return model


def save_cluster(model, cluster_path): 
    os.makedirs(os.path.dirname(cluster_path), exist_ok=True)
    
    joblib.dump(model, cluster_path)
    return


def load_cluster(cluster_path):
    model = joblib.load(cluster_path)
    return model







