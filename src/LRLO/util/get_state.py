"""
clustering for 3-dimension continuous state to (10,3) discrete state 
"""
import pickle
import pandas as pd
from sklearn.cluster import KMeans

def cluster_init(state_num=15):
    model = KMeans(n_clusters=state_num, n_init=10, random_state=42)
    return model
    
def cluster_train(model, data, clusterPath, visualize=False):
    print("start clustering for inputVideo")
    model.fit(data)
    with open(clusterPath, 'wb') as file:
        pickle.dump(model, file)
    return model


def cluster_load(clusterPath):
    with open(clusterPath, 'rb') as file:
        model = pickle.load(file)
    return model


def cluster_pred(originState, model):
    originState = [originState]
    s = model.predict(originState)
    return s
