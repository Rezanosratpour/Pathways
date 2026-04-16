import os 
import numpy as np 
import h5py

from CHI import CHI
import pandas as pd


def X_construct(chi_model , Lags, i , I_start , I_end ) : 
    D= len(Lags)+1
    array = []
    lags = Lags
    lags.append(0)
    land_indices = chi_model.land_indices
    indx = land_indices[i]
    name = f'{Lags[0]}_{Lags[1]}_indx_{(indx)}'
    try: 
        argmax = chi_model.H_memory[name]['argmax_points'][str(D)]
    except:
        argmax = chi_model.H_memory[name]['argmax_points']

    
    for m in range(len(lags)) :
        mem = argmax[m]
        id_0 = I_start + sum(lags[m:])  
        id_1 = I_end - sum(lags[:m]  ) 
        #print(id_0 , ':' , id_1)
        array.append(  np.array ( chi_model.Data[    id_0:id_1   ,  mem[0] , mem[1]   ] ).astype(float)  ) 
        #print(len(array[m]))

    x =  np.array(array)
    m = 0 
    id_0 = I_start + sum(lags[m:])  
    id_1 = I_end - sum(lags[:m]  ) 

    array.append(chi_model.date[ id_0:id_1   ]) 
    array = np.array(array)
    lags.remove(0)
    location = argmax[0]
    return pd.DataFrame(np.transpose(array)    )  , x , location



from tqdm import tqdm

def X_construct_reshape(chi_model , Lags , land_indices  , I_start = None , I_end = None):
    if I_start == None : I_start = chi_model.I_start_time 
    if I_end == None : I_end = chi_model.I_end_time 
    X_ = []
    Y_ = []
    for i in tqdm(range( len(land_indices)) , colour='blue'   ):
        df , x , location = X_construct(chi_model , Lags , i  , I_start = I_start, I_end= I_end ) 
        X_.append(x[1:].T)
        Y_.append(x[0].T)

    X_ = np.array(X_)
    Y_ = np.array(Y_)
    N , row , col = chi_model.Data.shape

    X_reshaped = []

    for t in tqdm(range(X_.shape[1])):
        image = np.zeros( (row, col , 2 ) )
        for i in range(X_.shape[0]):
            kk = chi_model.keys[i]
            image[kk[0]  , kk[1]  , :] = X_[i  , t , : ]
        X_reshaped.append(image)
    X_reshaped = np.array(X_reshaped)


    Y_reshaped = []

    for t in tqdm(range(X_.shape[1])):
        image = np.zeros( (row, col ) )
        for i in range(X_.shape[0]):
            kk = chi_model.keys[i]
            image[kk[0]  , kk[1] ] = Y_[i  , t  ]
        Y_reshaped.append(image)
    Y_reshaped = np.array(Y_reshaped)



    date = chi_model.date[ chi_model.I_start_time + sum(Lags): I_end ]

    return {'X' : X_reshaped   , 'Y'  : Y_reshaped , 'date'  : date   } 





def load_and_merge_chi_models(model_paths):
    """
    Load multiple CHI models and merge their H_memory into the first model.
    """
    if not model_paths:
        raise ValueError("model_paths is empty.")

    valid_paths = [fp for fp in model_paths if os.path.exists(fp)]
    if not valid_paths:
        raise FileNotFoundError("No valid model files were found.")

    base_model = CHI.load(valid_paths[0])

    for fp in valid_paths[1:]:
        temp_model = CHI.load(fp)
        if not hasattr(temp_model, "H_memory"):
            raise AttributeError(f"Loaded model from {fp} has no attribute 'H_memory'.")
        base_model.H_memory.update(temp_model.H_memory)

    return base_model


def build_joint_dict(chi_model, lag_list, land_indices, i_end):
    """
    Construct reshaped X/Y/date dictionaries for each lag combination.
    """
    joint_dict = {}
    for lags in lag_list:
        key = str(lags)
        joint_dict[key] = X_construct_reshape(
            chi_model,
            lags,
            land_indices,
            I_end=i_end
        )
    return joint_dict


def get_common_dates(joint_dict, sort_dates=True):
    """
    Find intersection of dates across all lag combinations.
    """
    date_sets = [set(np.asarray(v["date"])) for v in joint_dict.values()]
    common_dates = set.intersection(*date_sets)

    if sort_dates:
        common_dates = sorted(common_dates)
    else:
        common_dates = list(common_dates)

    return common_dates


def build_date_index_maps(joint_dict):
    """
    Build fast lookup dictionaries: date -> index for each lag key.
    """
    date_index_maps = {}
    for key, data in joint_dict.items():
        dates = np.asarray(data["date"])
        date_index_maps[key] = {date: i for i, date in enumerate(dates)}
    return date_index_maps


def build_X_array(joint_dict, common_dates, date_index_maps):
    """
    Build X array by concatenating features from all lag combinations
    along the last channel axis.
    Output shape will typically be: (N_dates, H, W, total_channels)
    """
    x_list = []

    for date in common_dates:
        features = []
        for key, data in joint_dict.items():
            idx = date_index_maps[key][date]
            feat = data["X"][idx:idx + 1, :, :, :]   # keep batch dim
            features.append(feat)

        stacked = np.concatenate(features, axis=-1)  # concatenate channels
        x_list.append(stacked)

    return np.concatenate(x_list, axis=0)


def build_Y_array(joint_dict, common_dates, date_index_maps, reference_key=None):
    """
    Build Y array using one reference lag entry.
    Assumes Y is identical/aligned across lag combinations after date intersection.
    Output shape will typically be: (N_dates, H, W)
    """
    if reference_key is None:
        reference_key = next(iter(joint_dict))

    y_data = joint_dict[reference_key]["Y"]
    y_map = date_index_maps[reference_key]

    y_list = []
    for date in common_dates:
        idx = y_map[date]
        feat = y_data[idx:idx + 1, :, :]   # keep batch dim
        y_list.append(feat)

    return np.concatenate(y_list, axis=0)


def save_puzzled_dataset(output_file, X_array, Y_array, dates):
    """
    Save output arrays to HDF.
    """
    Dict_to_hdf(
        Dict={
            "X": X_array,
            "Y": Y_array,
            "date": np.array(dates, dtype="S") if isinstance(dates[0], str) else np.array(dates)
        },
        of=output_file
    )


def create_multi_lag_dataset(model_paths, lag_list, shape_0, output_file):
    """
    Full pipeline:
    1. Load and merge models
    2. Build lag-based joint dictionary
    3. Find common dates
    4. Build X and Y arrays
    5. Save to HDF
    """
    chi_model = load_and_merge_chi_models(model_paths)

    land_indices = chi_model.land_indices
    joint_dict = build_joint_dict(
        chi_model=chi_model,
        lag_list=lag_list,
        land_indices=land_indices,
        i_end=shape_0
    )

    common_dates = get_common_dates(joint_dict, sort_dates=True)
    if len(common_dates) == 0:
        raise ValueError("No common dates found across lag combinations.")

    print(f"Number of common dates: {len(common_dates)}")
    print("Common dates:", common_dates)

    date_index_maps = build_date_index_maps(joint_dict)

    X_array = build_X_array(joint_dict, common_dates, date_index_maps)
    Y_array = build_Y_array(joint_dict, common_dates, date_index_maps)

    save_puzzled_dataset(output_file, X_array, Y_array, common_dates)

    return {
        "chi_model": chi_model,
        "joint_dict": joint_dict,
        "common_dates": common_dates,
        "X_array": X_array,
        "Y_array": Y_array,
    }

