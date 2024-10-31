import pandas as pd

def get_optimal_parameter(conf):
    profile = pd.read_csv(conf['profile_path'])
    filtered_profile = profile[profile['latency'] <= conf['latency_constraint']]
    result_row = filtered_profile.loc[filtered_profile['avg_f1'].idxmax()]

    sf, rf, avg_f1 = result_row['sf'], result_row['rf'], result_row['avg_f1']
    return sf, rf, avg_f1


def cal_fraction(dummy_shape, origin_shape):
    fraction = dummy_shape / origin_shape
    
    return fraction