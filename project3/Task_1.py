
import sys
from xml.parsers.expat import model

# adding Folder_2/subfolder to the system path
# sys.path.insert(0,r"/home/ec2-user/Project/code/py") #  r"C:\Users\S4055367\code\py"   "/home/ec2-user/Project/code/py"

import warnings
warnings.filterwarnings("ignore", message=".*OMP_NUM_THREADS=1.*")

from pathlib import Path
import sys

try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path.cwd()

sys.path.insert(0, str(script_dir))
sys.path.insert(0,r"D:\projects\Simulated Dataset\code\project")


dataset = 'Datasets3'
parent  = script_dir.parent
import sys

sys.path = [str(p) for p in sys.path]

import tensorflow as tf

from joblib import Parallel, delayed
from tqdm.auto import tqdm

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from itertools import product, islice
import tifffile as tff
import subprocess
import geopandas as gpd
from shapely.geometry import Point , LineString
from cluster_eval import *
from extremals import * 
from generators import *
from CHI import CHI
from chi_chunks import *
from multiprocessing import freeze_support
from Pazzeling_layers import *

from rasterio.mask import mask
import h5py
import numpy as np
import os 
import pandas as pd                                                                                                                                           
from matplotlib import pyplot
import time



def Dict_to_hdf(Dict , of ):
    HDF = h5py.File(of, 'w')
    for k in Dict :    
        test = HDF.create_dataset( k , data = Dict[k] )
    HDF.close()   

def heavy_task(i):
    # put your CPU-heavy code here
    return i ** 2


def build_models(fp_hdf, dir  , lag_memory = [ [1,1] , [1,2] , [2,1] ]  , type = 'raw' , fps = [] )  :

    out_folder  = f'{dir}/outputs/'
    os.makedirs(out_folder, exist_ok=True)

    if type == 'raw' :
        hdf  =  h5py.File(  fp_hdf )['root']
        Data =  np.array(   hdf['data']  ) 
        lat  =   np.array(   hdf['lat']  ) 
        lon  =   np.array(   hdf['lon']  ) 
        date =  np.array(   hdf['time']  ) 
    if type == 'real' :
        hdf =  h5py.File(  fp_hdf ) 
        Data = np.array(hdf['Data'])
        lat = np.array(hdf['lat']) 
        lon = np.array(hdf['lon']) 
        date = np.array(hdf['date']).astype(str)

    Shape = Data.shape[1:]

    keys  = generate(Shape)
    keys = np.array(keys) 

    land_indices = np.array([i for i in range(Shape[0]  * Shape[1])])
    

    for lags in lag_memory :

        lag_0 = lags[0]
        


        model_name = f'{lag_0}.pkl'
        fp = out_folder + model_name

        if fp  not in fps:
            start = time.time()
            chi_model = CHI(Data, lat, lon , date,
                            Start_date  = date[0]  , End_date  = date[-1] , 
                            land_indices= land_indices   )
            
            fps.append(fp)

            end = time.time()
            print(f"⏱️ Time taken: {end - start:.3f} seconds")

            chi_model.save( filepath= fp ) 
        


            #------------------------------build_Graph_V2_parallel--------------------------
            start = time.time()

            chi_model.build_Graph_V2_parallel( lag = lag_0 , out_folder  = out_folder , percentile = 0.95 ) 

            end = time.time()
            print(f"⏱️ build_Graph_V2_parallel Time taken: {end - start:.3f} seconds")

            chi_model.save_update()

            #--------------------------------------------------------
            start = time.time()

            chi_model.generate_max_path2( prcntile_threshold = 0.9 ,out_folder  = out_folder )

            end = time.time()
            print(f"⏱️ generate_max_path2 Time taken: {end - start:.3f} seconds")

            chi_model.save_update()

        else: 
            chi_model =  CHI.load(fp)

        start = time.time()
        lag_0 = chi_model.lag 

        Lags = lags
        Dim =  len(Lags) +1
        i_save = 0

        n_l = len( land_indices)
        high_chi_idx = list(chi_model.highest_chi_idx.keys())

        for i_indx in tqdm( range( n_l)):
            # print(f'{i_indx} of {n_l}')
            
            indx = land_indices[i_indx]
            key_indx = chi_model.keys[indx]
            I_key , J_key = key_indx
            i_save +=1 
            kk = high_chi_idx[i_indx]
            list_high = chi_model.highest_chi_idx[kk] 

            if  len(list_high) != 0  :
                chi_model.explore_for_station_fast( lat_st = chi_model.lat[I_key]   , 
                                    lon_st =  chi_model.lon[J_key] , 
                                    D = Dim , 
                                    lags  =  Lags ,
                                    Sampling_rates  = [1 , 0.2 , 0.2, 0.2 ] , 
                                    Percentile_sampling =0.99 , 
                                    out_folder  = out_folder , 
                                    name = f'{Lags[0]}_{Lags[1]}_indx_{(indx)}')
            else:
                print('--empty file --')
                name = f'{Lags[0]}_{Lags[1]}_indx_{(indx)}'
                chi_model.H_memory[name] = {}
                chi_model.H_memory[name]['argmax_points'] = np.array(  [ [I_key , J_key ] ] * Dim ) 
            

            if i_save ==1000:
                i_save =0
                chi_model.save_update()


        end = time.time()
        print(f"⏱️ Time taken: {end - start:.3f} seconds")

        chi_model.save_update()
    return fps   




def solve_pazzels(fp_models , lag_memory ,of  ):  
    
    # Load the first model as the base
    chi_model = CHI.load(fp_models[0])

    # Load the remaining models and merge their H_memory
    for fp in fp_models[1:]:
        temp_model = CHI.load(fp)
        chi_model.H_memory.update(temp_model.H_memory)

    # Final combined outputs
    land_indices = chi_model.land_indices
    H_memory = chi_model.H_memory

    # Creating a dictionary to store the joint dates for different lag combinations

    Joint_dict = {}
    for Lags in lag_memory :
        Joint_dict[str(Lags)] =  X_construct_reshape(chi_model , Lags  , land_indices , I_end   = chi_model.Data.shape[0] -1)


    sets_list = []
    for k in Joint_dict : 
        sets_list.append( set( Joint_dict[k]['date'])  )
    intesect = set.intersection(*sets_list)


    intesect = list(intesect)

    print( ' ----------- Pazzelling ------------'   )

    X_array = [ ]
    for i in tqdm(range(  len(intesect) )):
        this_date = list(intesect)[i]
        features = []
        for k in Joint_dict :
            i_k = np.where(Joint_dict[k]['date'] == this_date)[0]
            feat = Joint_dict[k]['X'][ i_k  , : , :  , :]
            features.append(feat)
        stacked = np.concatenate(features, axis=3)
        X_array.append(stacked)
    X_array  = np.array(X_array)



    Y_array = [ ]
    for i in tqdm(range(  len(intesect) )):
        this_date = intesect[i]
        features = []
        i_k = np.where(Joint_dict[k]['date'] == this_date)[0]
        feat = Joint_dict[k]['Y'][ i_k  , : , :  ]
        Y_array.append(feat)
        
    Y_array  = np.array(Y_array)
    Dict_to_hdf( Dict= {'X' : X_array   , 'Y'  : Y_array , 'date'  : intesect   }  , of  = of )
    return {'X' : X_array   , 'Y'  : Y_array , 'date'  : intesect   }


#'D:\projects\Simulated Dataset\IMERG'

def main__(dataset_dir  , Format ='.h5' , type = 'raw' ):
    lag_memory = [ [1,1] , [1,2]  , [2,1]  ] 
    for file in os.listdir(dataset_dir):
        if file.endswith(Format):
            fp_hdf = os.path.join(dataset_dir, file)
            dir = f'{dataset_dir}/{file.split(".")[0]}'
            os.makedirs(dir, exist_ok=True)
            files = os.listdir(dir) 
            fps = [ os.path.join(dir, f) for f in files if f.endswith('.pkl') ] 

            fp_models = build_models(fp_hdf = fp_hdf , dir = dir , lag_memory = lag_memory , type = type, fps =fps )
            out = solve_pazzels(fp_models , lag_memory , of = dir + f'/Pazz.h5' )


print(parent)

def main():
    main__(dataset_dir =  str(parent) +f"/{dataset}/" , Format = '.h5'  , type = 'raw'  )


from tensor_help import to_ds , make_pairs

import tensorflow as tf
from tensorflow.data import AUTOTUNE
from Evaluation import  evaluate_metrics 
from Evaluation import evaluate_2D
from Evaluation import *
import tensorflow as tf
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers, models

from Evaluation import *
from UNET import combined_loss ,  ssim_metric , psnr_metric , build_unet_convlstm


def UnetCovLSTM(fp_hdf , type = 'pazzel' , SEQ = 3 , name = 'UnetCovLSTM' ) :

    if type == 'real' :
        # shape: (N, H, W, C)

        hdf =  h5py.File(  fp_hdf ) 
        #frames = np.array(hdf['Data'])
        frames = hdf['Data']
        lat = np.array(hdf['lat']) 
        lon = np.array(hdf['lon']) 
        date = np.array(hdf['date']).astype(str)

        # Ensure frames have shape (N, H, W, C)
        if frames.ndim == 3:  # (N, H, W) -> add channel
            frames = frames[..., None]  # -> (N, H, W, 1)

        SEQ = 3
        X, y = make_pairs(frames, SEQ)

    
    if type == 'pazzel' :
        hdf =  h5py.File(  fp_hdf ) 
        X_  =  np.array(hdf['X']) 
        Y_ =  np.array(hdf['Y'])


        Reshaped_X = []

        for i in range(X_.shape[0]):
            bands = []
            for band in range(X_.shape[4]):
                image = X_[i ,0 ,   : , : , band ]
                bands.append(image)
            bands = np.array(bands)
            Reshaped_X.append(bands)

        Reshaped_X = np.array(Reshaped_X)
        X = np.expand_dims( Reshaped_X  , axis=-1) 


        y = np.expand_dims( Y_[: ,0 ,...]  , axis=-1) 


        X   = X.astype(np.float16)
        y   = y.astype(np.float16)

        SEQ = X.shape[1]



    # RAW
    if type == 'raw' :
        hdf =  h5py.File(  fp_hdf ) 
        frames = hdf['root']['data']
        
        X, y = make_pairs(frames, SEQ)
        X = np.expand_dims( X  , axis=-1) 
        y = np.expand_dims( y  , axis=-1)  





    # ----------   Transform to ds -------------


    # Split
    n = len(X)
    idx = np.arange(n)


    train_idx, val_idx = idx[:int(0.8*n)], idx[int(0.8*n):]

    Xtr, ytr, Xva, yva = X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    print("Xtr shape:", X[train_idx].shape)  # expect (N, SEQ, H, W, C)
    print("ytr shape:", y[train_idx].shape)  # expect (N, H, W, C)

    # tf.data
    BATCH = 50
    AUTOTUNE = tf.data.AUTOTUNE


    #train_ds = to_ds(Xtr, ytr, True)
    #val_ds   = to_ds(Xva, yva, False)


    train_ds = to_ds(X[train_idx], y[train_idx], True)
    val_ds   = to_ds(X[val_idx], y[val_idx], False)

    ds_all = to_ds(X, y, False)

    # If you're using tf.data:
    train_ds = train_ds.map(lambda x,y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))
    val_ds   = val_ds.map(lambda x,y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))
    ds_all   = ds_all.map(lambda x,y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))
    seq_len, H, W, C = Xtr.shape[1], Xtr.shape[2], Xtr.shape[3], Xtr.shape[4]


        
    # =====================================================
    # 3) Build from dataset, compile, callbacks, fit
    # =====================================================

    # Peek at one batch to infer shapes
    x_batch, y_batch = next(iter(train_ds))
    print("x_batch shape:", x_batch.shape)  # (B, T, H, W, C)
    print("y_batch shape:", y_batch.shape)  # (B, H, W, 1) expected

    seq_len = int(x_batch.shape[1])
    H       = int(x_batch.shape[2])
    W       = int(x_batch.shape[3])
    C       = int(x_batch.shape[4])

    # Build model
    model = build_unet_convlstm(
        seq_len=seq_len,
        H=H,
        W=W,
        C=C,
        base=123,
        dropout=0.1,
    )

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=combined_loss,
        metrics=["mae", psnr_metric, ssim_metric],
    )

    model.summary()

    # Callbacks
    checkpoint_path = str(parent) + f"/Models/{name}.keras"
    log_dir = str(parent) + f"/Logs/unet_convlstm"

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor="val_ssim_metric",
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_ssim_metric",
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_ssim_metric",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            write_graph=True,
        ),
    ]
    
    # Fit
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        batch_size= 128 ,
        callbacks=callbacks,
        verbose=1,
    )

    fp_model = str(parent) + f"/Models/Unet.keras"
    model.save(fp_model)

    # Collect all true and predicted values
    y_true_list = []
    y_pred_list = []

    for x_batch, y_batch in val_ds:
        y_pred_batch = model.predict(x_batch, verbose=0)
        y_true_list.append(y_batch.numpy())
        y_pred_list.append(y_pred_batch)

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

        
    metrics = evaluate_metrics(y_true, y_pred)


    # Run the model over the whole period
    y_pred_all = model.predict(ds_all, verbose=1)  # (N, H, W, 1)

    # Convert to 3D: (N, H, W)
    y_pred_3d = y_pred_all[..., 0]  # squeeze last channel
    y_true_3D = y[... , 0 ]
    n , row , col  ,  = y_true_3D.shape

    #date_prediction = pd.to_datetime(hdf['time']) + pd.Timedelta(days=SEQ)
    #date_prediction = date_prediction[:-SEQ]

    forecast ={ 'forecast'  :y_pred_3d , 
                'y_true'  :y_true_3D , 
                       } 
    
    eval_image = evaluate_2D(y_true_3D , y_pred_3d, function= corr)


    return metrics , forecast , eval_image



def compare():
    metrics1 , forecast1 , eval_image1 = UnetCovLSTM(fp_hdf =  str(parent) + f'/{dataset}/simulated_data.h5' , 
                                                     type = 'raw' , 
                                                     name =  'UnetCovLSTM_raw'  )
    
    metrics2 , forecast2 , eval_image2 = UnetCovLSTM(fp_hdf =  str(parent) +  f'/{dataset}/simulated_data/Pazz.h5' , 
                                                     type = 'pazzel' , 
                                                     name =  'UnetCovLSTM_pazzel'  )

    images = [eval_image1 , eval_image2] 
    print("Comparison of two runs:")
    for key in metrics1:
        print(f"{key:15s}: Run 1 = {metrics1[key]:.4f}, Run 2 = {metrics2[key]:.4f}")
    
    df = pd.DataFrame({
        "Metric": list(metrics1.keys()),
        "Run 1": list(metrics1.values()),
        "Run 2": list(metrics2.values())
    })

    vstack = np.vstack(images)
    tff.imshow(vstack)
    print(df)

if __name__ == "__main__":
    freeze_support()   # useful on Windows
    compare()




