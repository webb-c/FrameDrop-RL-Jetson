import os
import pickle
import joblib


def load_joblib(path):
    model = joblib.load(path)
    return model

def save_pickle(path, model):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(model, file)
    return True
    
def joblib_to_pickle(dir_path):
    file_paths = []
    for root, directories, files in os.walk(dir_path):
        for filename in files:
            file_paths.append(os.path.join(root, filename))
    
    for path in file_paths:
        model = load_joblib(path)
        save_pickle(path, model)

        
        
dir_list = ['model/FrameHopper/cluster/']
for directory in dir_list:
    joblib_to_pickle(directory)