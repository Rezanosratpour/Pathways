import math
import sys
import pandas as pd
from shapely.geometry import LineString, Point
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import h5py

from pathlib import Path
# .parent gets the directory containing the file
script_dir = Path(__file__).resolve().parent
sys.path.insert(0,script_dir)

import warnings
warnings.filterwarnings("ignore", message=".*OMP_NUM_THREADS=1.*")
import pickle

# IMPORTANT: keep this at top-level (not nested in a class method),
# so it can be pickled by multiprocessing.

from chi_chunks import *
from transformations import *
from extremals import *
from geo_locators import *
from Pooling_Layers import *
from Pooling import *
from geo_locators import *
from generators import *
from georef import * 


from chi_chunks import _chunk_array, _compute_chi_block, _init_chi_ctx, _chi_worker, _coord_key

from chi_func import * 
from chi_func import _coords_to_flat  , _flat_combo_to_coords , _eval_station_batch ,_unique_combo_batches2

import warnings

# Ignore all FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Ignore all UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

# Ignore all RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)



def Dict_to_hdf(Dict , of ):
    HDF = h5py.File(of, 'w')
    for k in Dict :    
        test = HDF.create_dataset( k , data = Dict[k] )
    HDF.close()   

class CHI:
    def __init__(self , Data, lat, lon , date,Start_date  , End_date , land_indices  ) :
                 
    
        """
        Data: 3D data (time, row, column )

        Start_date :  'YYYY-MM-DD' 
        End_date  : 'YYYY-MM-DD' 

        mask : file path of maked area with data [0 , 1] 
        
        """
        self.lags_memory = [ ]
        
        self.H_memory = {}
        self.I_start_time = int(np.where( date == Start_date )[0][0])
        self.I_end_time = int(np.where( date == End_date )[0][0])
        print(f"Start time index: {self.I_start_time}" , f"End time index: {self.I_end_time}" )
        
        self.grid_date_subset = date[  self.I_start_time : self.I_end_time ] 
        self.cdf_data  = Multi_dim_CDF_transform( Data )
        self.Shape = Data[0].shape   
        self.Row , self.Col  = self.Shape
        self.keys = np.array(generate(self.Shape) ) 
        self.n_keys  = len(self.keys)
        self.array_chi = np.zeros( (self.n_keys  , self.n_keys) )

        self.land_indices  = land_indices
        self.V = np.array([i for i in range(len(self.keys))])
        
        self.save_as_geo_image = Write_georef_image(top_left_lat= lat[0] , top_left_lon= lon[0] , 
                                       pixel_size_x= abs(lat[1]- lat[0])  ,
                                       pixel_size_y=   abs(lon[1]- lon[0]))

        self.lat = lat 
        self.lon = lon
        self.Data =  Data

        self.date  = date   

    def Data_to_probability(self ):
        
        self.Pr_Data = []
        
        for  i in range( self.Row ) :
            for j in range( self.Col ):
                self.Pr_Data.append(  to_empirical_cdf( self.Data[: , i  , j] )) 
        
        self.Pr_Data = np.array( self.Pr_Data ) 
        
    def X_to_Chi(self, key_1 , key_2, lag , percentile ) :
        data_1 = self.cdf_data[key_1 , self.I_start_time +  lag : self.I_end_time   ] 
        data_2 = self.cdf_data[ key_2 , self.I_start_time  : self.I_end_time - lag     ]
        X = np.array( [data_1  , data_2 ]).T
        return Multivar_Chi3( X  , percentile )
        
    def build_Graph( self , lag , out_folder , percentile): 
        self.lag = lag
        self.percentile = percentile      
        
        for  i in tqdm( range(  0 ,  len(self.land_indices)) , desc=" Chi ..." , ncols= 100  ,colour='red')  : 
            
            i_key = self.land_indices[i]
        
            key_1 = self.keys[i_key] 
        
            #data_1 = self.Data[ self.I_start_time +  lag : self.I_end_time  , key_1[0] , key_1[1] ]
            
            data_1 = self.cdf_data[i_key , self.I_start_time +  lag : self.I_end_time   ] 
            
            for j_key , key_2  in  enumerate(self.keys) :
                
                #data_2 = self.Data[self.I_start_time  : self.I_end_time - lag  , key_2[0] , key_2[1]  ]
                data_2 = self.cdf_data[ j_key , self.I_start_time  : self.I_end_time - lag     ]
                
                X = np.array( [data_1  , data_2 ]).T
                
                # df_X = pd.DataFrame(X)
                # chi = Chi.emp_chi_multdim( Data , 0.95)
                # self.array_chi[i_key  , j_key]  = Chi.Multivar_Chi( df_X  , percentile )

                self.array_chi[i_key  , j_key]  = Multivar_Chi3( X  , percentile )
                
            tff.imwrite( out_folder +  f'chi_Lag-{lag}.tiff'  , self.array_chi  )
        self.array_chi[self.array_chi < 0.05] =0 
    def build_Graph_V2( self , lag , out_folder , percentile): 
        self.lag = lag
        self.percentile = percentile    
        

        Lists = [self.land_indices ,  self.V]
        
        self.combinations =  list(product(*Lists))

        
        for  mem in tqdm( range(  0 ,  len(self.combinations)) , desc="Graph Construction" )   :
            
            self.indx_i = self.combinations[mem][0]
            self.indx_j = self.combinations[mem][1]
            data_1 = self.cdf_data[self.indx_i , self.I_start_time +  lag : self.I_end_time   ] 
            data_2 = self.cdf_data[ self.indx_j , self.I_start_time  : self.I_end_time - lag     ]
            X = np.array( [data_1  , data_2 ]).T
            self.array_chi[self.indx_i  , self.indx_j]  =   Multivar_Chi3( X  , percentile ) #self.X_to_Chi( key_1 = self.indx_i  , key_2 = self.indx_j, lag = lag)
            
        tff.imwrite( out_folder +  f'chi_Lag-{lag}.tiff'  , self.array_chi  )  

    def build_Graph_V2_parallel(self, lag, out_folder, percentile, n_jobs=-1, chunk_size=16):
        self.lag = lag
        self.percentile = percentile

        if lag <= 0:
            raise ValueError("lag must be a positive integer.")

        if self.I_end_time - self.I_start_time <= lag:
            raise ValueError("Time window is too short for the selected lag.")

        V = np.asarray(self.V, dtype=np.int64)
        land_indices = np.asarray(self.land_indices, dtype=np.int64)

        # Pre-slice once outside the loop
        # data_left  corresponds to i with shifted start
        # data_right corresponds to j with shifted end
        data_left = self.cdf_data[:, self.I_start_time + lag : self.I_end_time]
        data_right = self.cdf_data[:, self.I_start_time : self.I_end_time - lag]

        if data_left.shape[1] != data_right.shape[1]:
            raise ValueError("Lagged arrays do not have matching time dimensions.")

        # Allocate output if needed
        if not hasattr(self, "array_chi") or self.array_chi is None:
            n_nodes = self.cdf_data.shape[0]
            self.array_chi = np.full((n_nodes, n_nodes), np.nan, dtype=np.float32)

        # Build chunks of source indices
        i_chunks = list(_chunk_array(land_indices, chunk_size))

        # Parallel computation over blocks of rows
        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",      # process-based, safer for CPU-bound work
            verbose=10,
            max_nbytes="256M"    # helps joblib memmap large arrays
        )(
            delayed(_compute_chi_block)(
                i_block=chunk,
                V=V,
                data_left=data_left,
                data_right=data_right,
                percentile=percentile
            )
            for chunk in i_chunks
        )

        # Write results back into the matrix
        for i_block, block in results:
            self.array_chi[np.ix_(i_block, V)] = block

        # Save output
        out_path = os.path.join(out_folder, f"chi_Lag-{lag}.tiff")
        tff.imwrite(out_path, self.array_chi)

    def build_Graph_new(self , lag , out_folder , percentile):
        Theta_array = []
        self.lag = lag
        
        for i in tqdm( range(  0 ,  self.cdf_data.shape[0] ) , desc=" Chi ..." , ncols= 100  ,colour='red')   :

            if i in self.land_indices:
            
                theta_point = Theta2(self.cdf_data[i , self.I_start_time +  lag : self.I_end_time   ], self.cdf_data[ : , self.I_start_time  : self.I_end_time - lag     ] )
            
                Theta_array.append(  theta_point )
            else:
                Theta_array.append(  np.zeros(self.Shape[0] * self.Shape[1] ) +2 )
                
            
            
        self.Theta = np.array(Theta_array)
        self.Theta  = np.minimum( self.Theta , 2) 
        self.array_chi = 2-  self.Theta

    def generate_max_path(self, prcntile_threshold ,out_folder  ):
        # prcntile_threshold : 0.95 , 0.99 
        
        prcntile_threshold = prcntile_threshold *100
        
        of_uv  = out_folder + f"uv_geo_Lag-{self.lag}.tiff"
        of_chi_max = out_folder + f"max_geo_Lag-{self.lag}.tiff"

        # array_chi = tff.imread( out_folder +  f'chi_Lag-{self.lag}.tiff'   ) 

        Shape_uv  =  2 , self.Shape[0]  , self.Shape[1]  

        self.uv = np.zeros(Shape_uv  ) / 0

        self.max_chi  = np.zeros(( 1 , self.Shape[0]  , self.Shape[1]) )

        self.highest_chi_idx = {}

        for I_arr in self.land_indices :

            List_chi = self.array_chi[I_arr]
            i_s , j_s = self.keys[I_arr]

            pcnt99 =  np.percentile(List_chi , 99 )
            h = np.where(List_chi > pcnt99 )[0]
            chi_values = List_chi[h]

            if sum( chi_values) != 0 :
                
                i_c = sum( self.keys[h].T[0] * chi_values) /sum( chi_values)
                j_c = sum( self.keys[h].T[1] * chi_values) /sum( chi_values)
                    
                self.uv[0, i_s, j_s] =  (j_s - j_c )* np.mean(chi_values)
                self.uv[1, i_s, j_s] =  (i_c - i_s)* np.mean(chi_values)
    
                self.max_chi[ 0  , i_s   , j_s]  = np.max(chi_values )
                self.highest_chi_idx[ str(self.keys[I_arr]) ] =  h
                
            else: 
                self.highest_chi_idx[ str(self.keys[I_arr]) ] =  h
        
        self.save_as_geo_image.to_image(image =  self.uv , output_path = of_uv )
        self.save_as_geo_image.to_image(image =  self.max_chi , output_path= of_chi_max )
    
        Dict_to_hdf(Dict   =      self.highest_chi_idx  ,  of =  of_chi_max  + f'_dict_highest_idx_Lag-{self.lag}.hdf'  )  

    def generate_max_path2(self, prcntile_threshold, out_folder):
        """
        prcntile_threshold should be in [0, 1], e.g. 0.95 or 0.99
        """

        if not (0 < prcntile_threshold < 1):
            raise ValueError("prcntile_threshold must be between 0 and 1, e.g. 0.95 or 0.99")

        pct = prcntile_threshold * 100.0

        of_uv = os.path.join(out_folder, f"uv_geo_Lag-{self.lag}.tiff")
        of_chi_max = os.path.join(out_folder, f"max_geo_Lag-{self.lag}.tiff")
        of_dict = os.path.join(out_folder, f"dict_highest_idx_Lag-{self.lag}.hdf")

        # safer than zeros/0
        self.uv = np.full((2, self.Shape[0], self.Shape[1]), np.nan, dtype=float)
        self.max_chi = np.full((1, self.Shape[0], self.Shape[1]), np.nan, dtype=float)

        self.highest_chi_idx = {}

        keys = np.asarray(self.keys)

        for I_arr in self.land_indices:
            List_chi = np.asarray(self.array_chi[I_arr], dtype=float)
            i_s, j_s = keys[I_arr]

            # ignore NaNs
            if np.all(np.isnan(List_chi)):
                self.highest_chi_idx[str(keys[I_arr])] = np.array([], dtype=int)
                continue

            threshold_val = np.nanpercentile(List_chi, pct)
            h = np.where(List_chi >= threshold_val)[0]
            chi_values = List_chi[h]

            # remove NaNs if any slipped in
            valid = ~np.isnan(chi_values)
            h = h[valid]
            chi_values = chi_values[valid]

            self.highest_chi_idx[str(keys[I_arr])] = h

            if chi_values.size == 0:
                continue

            weight_sum = np.sum(chi_values)
            if weight_sum == 0:
                continue

            i_c = np.sum(keys[h, 0] * chi_values) / weight_sum
            j_c = np.sum(keys[h, 1] * chi_values) / weight_sum

            # vector from source to weighted center, scaled by mean chi
            mean_chi = np.mean(chi_values)
            self.uv[0, i_s, j_s] = (j_s - j_c) * mean_chi
            self.uv[1, i_s, j_s] = (i_c - i_s) * mean_chi

            self.max_chi[0, i_s, j_s] = np.max(chi_values)

        self.save_as_geo_image.to_image(image=self.uv, output_path=of_uv)
        self.save_as_geo_image.to_image(image=self.max_chi, output_path=of_chi_max)

        Dict_to_hdf(Dict=self.highest_chi_idx, of=of_dict)
    
    def find_optimum_k( self ,   k_range  , out_folder):  
        
        X =   self.array_chi[self.land_indices] # UV_station.values  
        
        # Lists to store evaluation scores
        
        k_values = range(k_range[0], k_range[1])
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        davies_scores = []
        sum_within_group_dists = []
        sum_between_group_dists = []
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, 
                            init="k-means++",
                            n_init=50,
                            random_state=7 )
            labels = kmeans.fit_predict(X)
            centroids = kmeans.cluster_centers_
        
            # Inertia and metrics
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, labels))
            calinski_scores.append(calinski_harabasz_score(X, labels))
            davies_scores.append(davies_bouldin_score(X, labels))
        
            # Sum of within-group distances
            dist_within = 0
            for i in range(k):
                cluster_points = X[labels == i]
                dist_within += np.sum(cdist(cluster_points, [centroids[i]]))
            sum_within_group_dists.append(dist_within)
        
            # Sum of between-group distances
            global_centroid = np.mean(X, axis=0)
            dist_between = np.sum(cdist(centroids, [global_centroid]) * np.bincount(labels).reshape(-1, 1))
            sum_between_group_dists.append(dist_between)
        
        # Plotting
        plt.figure(figsize=(16, 10))
        
        plt.subplot(3, 2, 1)
        plt.plot(k_values, inertias, 'o-')
        plt.title("Inertia (WSS)")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia (lower is better)")
        
        plt.subplot(3, 2, 2)
        plt.plot(k_values, silhouette_scores, 'o-g')
        plt.title("Silhouette Score")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Score (higher is better)")
        
        plt.subplot(3, 2, 3)
        plt.plot(k_values, calinski_scores, 'o-r')
        plt.title("Calinski-Harabasz Index")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Score (higher is better)")
        
        plt.subplot(3, 2, 4)
        plt.plot(k_values, davies_scores, 'o-m')
        plt.title("Davies-Bouldin Index")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Score (lower is better)")
        
        plt.subplot(3, 2, 5)
        plt.plot(k_values, sum_within_group_dists, 'o-c')
        plt.title("Sum of Within-Group Distances")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Total Distance (lower is better)")
        
        plt.subplot(3, 2, 6)
        plt.plot(k_values, sum_between_group_dists, 'o-k')
        plt.title("Sum of Between-Group Distances")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Total Distance (higher is better)")
        
        plt.tight_layout()
        
        
        plt.savefig( out_folder +  f'Clustering_scores_lag-{self.lag}.png' , dpi=600, bbox_inches='tight', transparent= False ) 
        plt.show()
        
        
        clustering_evaluation_df = pd.DataFrame({  'inertias' : inertias , 
         'silhouette_scores' : silhouette_scores,
            'calinski_scores'  : calinski_scores ,
            'davies_scores'  :  davies_scores , 
            'SDW' : sum_within_group_dists  ,
            'SDB' : sum_between_group_dists ,
            'k'  : np.arange(k_range[0] , k_range[1]) })
        clustering_evaluation_df.to_excel(out_folder + 'clustering_evaluation_df.xlsx' )
        
        
        max_Sil = clustering_evaluation_df.silhouette_scores.max()
        
        self.opt_k = clustering_evaluation_df[ clustering_evaluation_df.silhouette_scores  == max_Sil ].k.values[0]

        self.evaluation = clustering_evaluation_df
        
        of_clustering_results = out_folder  + f'Clustering_scores_Lag-{self.lag}.xlsx'
        clustering_evaluation_df.to_excel(of_clustering_results)
    def find_optimum_k_parallel(
        self,
        k_range,
        out_folder,
        n_jobs=-1,
        n_init=20,
        silhouette_sample_size=2000,
        random_state=7,
        save_plot=True,
        show_plot=False,
    ):
        """
        Optimized and parallel version of find_optimum_k.

        Parameters
        ----------
        k_range : tuple
            Example: (2, 10) -> evaluates k = 2,3,...,9
        out_folder : str
            Output directory
        n_jobs : int
            Number of parallel jobs. -1 uses all CPUs
        n_init : int
            Number of KMeans initializations
        silhouette_sample_size : int or None
            If not None, silhouette is computed on a sample when X is large
        """

        os.makedirs(out_folder, exist_ok=True)

        X = self.array_chi[self.land_indices]

        # Optional: remove rows with NaN/Inf if needed
        mask = np.isfinite(X).all(axis=1)
        X = X[mask]

        # Use float32 to reduce memory if acceptable
        X = np.ascontiguousarray(X, dtype=np.float32)

        k_values = list(range(k_range[0], k_range[1]))

        # Parallel over k
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(_evaluate_one_k)(
                X=X,
                k=k,
                random_state=random_state,
                n_init=n_init,
                silhouette_sample_size=silhouette_sample_size,
            )
            for k in k_values
        )

        clustering_evaluation_df = pd.DataFrame(results).sort_values("k").reset_index(drop=True)

        self.evaluation = clustering_evaluation_df
        self.opt_k = int(
            clustering_evaluation_df.loc[
                clustering_evaluation_df["silhouette_scores"].idxmax(), "k"
            ]
        )

        # Save table once
        excel_path = os.path.join(out_folder, f"Clustering_scores_Lag-{self.lag}.xlsx")
        clustering_evaluation_df.to_excel(excel_path, index=False)

        # Plot
        if save_plot:
            fig, axes = plt.subplots(3, 2, figsize=(16, 10))

            axes[0, 0].plot(clustering_evaluation_df["k"], clustering_evaluation_df["inertias"], "o-")
            axes[0, 0].set_title("Inertia (WSS)")
            axes[0, 0].set_xlabel("Number of Clusters")
            axes[0, 0].set_ylabel("Inertia (lower is better)")

            axes[0, 1].plot(clustering_evaluation_df["k"], clustering_evaluation_df["silhouette_scores"], "o-")
            axes[0, 1].set_title("Silhouette Score")
            axes[0, 1].set_xlabel("Number of Clusters")
            axes[0, 1].set_ylabel("Score (higher is better)")

            axes[1, 0].plot(clustering_evaluation_df["k"], clustering_evaluation_df["calinski_scores"], "o-")
            axes[1, 0].set_title("Calinski-Harabasz Index")
            axes[1, 0].set_xlabel("Number of Clusters")
            axes[1, 0].set_ylabel("Score (higher is better)")

            axes[1, 1].plot(clustering_evaluation_df["k"], clustering_evaluation_df["davies_scores"], "o-")
            axes[1, 1].set_title("Davies-Bouldin Index")
            axes[1, 1].set_xlabel("Number of Clusters")
            axes[1, 1].set_ylabel("Score (lower is better)")

            axes[2, 0].plot(clustering_evaluation_df["k"], clustering_evaluation_df["SDW"], "o-")
            axes[2, 0].set_title("Sum of Within-Group Distances")
            axes[2, 0].set_xlabel("Number of Clusters")
            axes[2, 0].set_ylabel("Total Distance (lower is better)")

            axes[2, 1].plot(clustering_evaluation_df["k"], clustering_evaluation_df["SDB"], "o-")
            axes[2, 1].set_title("Sum of Between-Group Distances")
            axes[2, 1].set_xlabel("Number of Clusters")
            axes[2, 1].set_ylabel("Total Distance (higher is better)")

            fig.tight_layout()

            fig_path = os.path.join(out_folder, f"Clustering_scores_lag-{self.lag}.png")
            fig.savefig(fig_path, dpi=600, bbox_inches="tight", transparent=False)

            if show_plot:
                plt.show()
            else:
                plt.close(fig)
    def set_optimum_k( opt_k ) :
        self.opt_k = opt_k

    def cluster( self , out_folder ) :
        X =   self.array_chi[self.land_indices]
        
        kmeans = KMeans(n_clusters=self.opt_k,
                            init="k-means++",
                            n_init=50,
                            random_state=7 )
        labels = kmeans.fit_predict(X)
        self.centroids = kmeans.cluster_centers_
        Cs = self.centroids.reshape(self.opt_k , self.Shape[0] ,  self.Shape[1]) 
        self.kmeans = kmeans 
        lables_image = np.zeros(self.n_keys) -99 
        
        lables_image[self.land_indices]  = labels
        
        self.image_lbl= lables_image.reshape( self.Shape)
        
        #tff.imshow(image_lbl)
        
        fp_tif_clusters = out_folder + f'Clustered_K({self.opt_k})_Lag{self.lag}.tif'
        fp_tif_cent = out_folder + f'Centroids_K({self.opt_k})_Lag{self.lag}.tif'
        
        self.save_as_geo_image.to_image(np.array([self.image_lbl] ) , fp_tif_clusters)
        self.save_as_geo_image.to_image( Cs  , fp_tif_cent)
        
       
    def explore_hyper_dim( self, D, lags , sampling_rates , out_folder , Percentile_sampling  , tolerance = 0.01 , delta = 0.01  ):
        
        
        self.ERRORS =  {}
        for k in range( self.opt_k) :
            
            image_cent = self.centroids[k].reshape( self.Shape[0] ,  self.Shape[1] )
            
            indices_lbl = np.where( self.image_lbl == k)
            indices_lbl = np.array( [indices_lbl[0]   ,  indices_lbl[1] ]  ).T
            np.random.shuffle(indices_lbl) 
            
            indices_cnt = np.where(image_cent  > np.percentile(image_cent, Percentile_sampling))
            indices_cnt = np.array( [indices_cnt[0]   ,  indices_cnt[1] ]  ).T
            np.random.shuffle(indices_cnt)  
            
            Indices =  {   '0'  :    indices_lbl ,
                           '1'  : indices_cnt   ,      } 
            
            Hk_mean = []
            
            Hk_var = []
            
            Sampling_rates = np.array(sampling_rates)
           

            for d in range(3, D+1):
                Errors = [] 
                len_comb = 0
                mean_err = 10E10


                Chi_list = Chi_list_generator(self.keys)
                my_bag = [ ]
                i_while =0 

                while mean_err  > tolerance  :

                    #print(combinations_array)
                    #print("Shape:", combinations_array.shape)
                    Lists , Indices_sample  , n_Edges_per_pixel = Indices_generate_samples( Indices  ,  Sampling_rates , self.keys)
                    # Generate all possible combinations
                    combinations = list(product(*Lists)) # X master
                    
                    # Convert to numpy array if needed
                    combinations_array = np.array(combinations)
                    len_comb_arr_now = len(combinations_array)

                    for i in tqdm(range(  len_comb , len_comb_arr_now )    , desc=f"Cluster{k+1} for d: {d} "   ) :  
                        if str(combinations_array[i]) not in my_bag:
                            X = []
                            for i_d in range(d):
                                ii , jj = combinations_array[i][i_d][0] , combinations_array[i][i_d][1] 
                                
                                index = ii * self.Col + jj 
                                #
                                #X.append(  self.Data[self.I_start_time + sum(lags[:i_d])  :   self.I_end_time - sum(lags[i_d:])   , ii, jj] ) 
                                X.append(  self.cdf_data[ index , self.I_start_time + sum(lags[:i_d])  :   self.I_end_time - sum(lags[i_d:])      ] ) 
                            X = np.array(X)             
                            # print(X.shape)
                            # dfX = pd.DataFrame( X.T)
                            # Chi_list[  str(  np.array( [ii, jj] )  )  ].append( Chi.Multivar_Chi( dfX , .95 ) )  # Chi.Multivar_Chi( dfX , 0.95 ) )
                            
                            Chi_list[  str( np.array( [ii, jj] )  )  ].append( Multivar_Chi3( X.T , .95 ) ) 
                            my_bag.append( str(combinations_array[i]) )

                    

                    Chi_mean = []
                    Chi_var  = [] 
                    
                    for kk in Chi_list:
                        Chi_mean.append( np.mean(Chi_list[kk]) )
                        Chi_var.append( np.var(Chi_list[kk]) )
                        
                    Chi_mean = np.reshape( Chi_mean , self.Shape) 
                    Chi_var = np.reshape( Chi_var , self.Shape) 

                    try: 
                        error =  np.absolute(Chi_mean - last_chi_mean)
                        mean_err = np.mean(error)
                        print(mean_err)
                        Errors.append( mean_err)
                    except:
                        pass 
                    last_chi_mean = Chi_mean
                    
        
                    Sampling_rates += delta 
                    i_while += 1
                    len_comb = len(combinations_array)
                
                self.ERRORS[f'k{k+1}d{d}'] = Errors

                indices_h_k_q = np.where(Chi_mean  > np.percentile(Chi_mean, Percentile_sampling))
                indices_h_k_q = np.array( [indices_h_k_q[0]   ,  indices_h_k_q[1] ]  ).T
                
                np.random.shuffle(indices_h_k_q)  
                    
                Hk_mean.append( Chi_mean)
                Hk_var.append( Chi_var)

                

                Indices[str(d-1)] = indices_h_k_q
                
                self.save_as_geo_image.to_image( np.array([Chi_mean] ) , out_folder + f'Chi_mean_d({d})_k({k}).tif' )
                self.save_as_geo_image.to_image( np.array([Chi_var] ) , out_folder + f'Chi_var_d({d})_k({k}).tif' )
                
            self.H_memory[str(k)] = {'mean':Hk_mean , 'var': Hk_var}

            del Chi_mean 
            del Chi_var
        self.lags_memory.append(lags) 


    def explore_hyper_dim_fast(self, D,
        lags,
        sampling_rates,
        out_folder,
        Percentile_sampling,
        tolerance=0.01,
        delta=0.01,
        chi_q=0.95,
        max_iter=100,
        show_progress=True,
    ):
        
        """
        Faster version of explore_hyper_dim.

        Main speedups:
        1. No list(product(*Lists)) materialization.
        2. No string-based duplicate checking; uses a set of tuples of flat indices.
        3. Precomputes lag windows once per d.
        4. Uses running count/sum/sumsq instead of storing all chi values in lists.
        5. Uses flat pixel indices directly instead of repeated (i, j) -> string conversions.

        Notes:
        - This version assumes self.cdf_data has shape (n_pixels, time).
        - It uses lags[:d-1] for dimension d, which is usually the intended behavior.
        - It preserves the behavior of carrying increased sampling rates forward across d.
        """
        
        shape = tuple(self.Shape)
        n_rows, n_cols = shape
        n_pix = n_rows * n_cols

        lags = np.asarray(lags, dtype=int)
        if len(lags) < D - 1:
            raise ValueError(
                f"Need at least D-1 lags. Got len(lags)={len(lags)} for D={D}."
            )

        base_sampling_rates = np.asarray(sampling_rates, dtype=float)
        flat_cdf = self.cdf_data  # shape: (n_pix, T)

        out_folder = Path(out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)

        if not hasattr(self, "H_memory"):
            self.H_memory = {}

        self.ERRORS = {}
        rng = np.random.default_rng()

        def coords_to_flat(coords):
            coords = np.asarray(coords)
            if coords.size == 0:
                return np.empty((0,), dtype=np.int64)
            return coords[:, 0].astype(np.int64) * self.Col + coords[:, 1].astype(np.int64)

        for k in range(self.opt_k):
            image_cent = self.centroids[k].reshape(shape)

            indices_lbl = np.argwhere(self.image_lbl == k)
            rng.shuffle(indices_lbl)

            cnt_thr = np.percentile(image_cent, Percentile_sampling)
            indices_cnt = np.argwhere(image_cent > cnt_thr)
            rng.shuffle(indices_cnt)

            Indices = {
                "0": indices_lbl,
                "1": indices_cnt,
            }

            Hk_mean = []
            Hk_var = []

            # Preserve original behavior: sampling rates keep growing across d
            current_sampling_rates = base_sampling_rates.copy()

            for d in range(3, D + 1):
                errors = []
                seen = set()

                # Running stats for target pixel = last node in combo
                chi_count = np.zeros(n_pix, dtype=np.int32)
                chi_sum = np.zeros(n_pix, dtype=np.float64)
                chi_sumsq = np.zeros(n_pix, dtype=np.float64)

                # Precompute lag windows once for this d
                lag_prefix = np.zeros(d, dtype=np.int64)
                lag_prefix[1:] = np.cumsum(lags[: d - 1])
                total_lag = lag_prefix[-1]

                starts = self.I_start_time + lag_prefix
                ends = self.I_end_time - (total_lag - lag_prefix)
                T_eff = int(ends[0] - starts[0])

                if T_eff <= 0:
                    raise ValueError(
                        f"Non-positive effective time length for d={d}. "
                        f"Check I_start_time, I_end_time, and lags."
                    )

                chi_mean = np.full(shape, np.nan, dtype=np.float64)
                chi_var = np.full(shape, np.nan, dtype=np.float64)
                last_chi_mean = None
                mean_err = np.inf

                for _ in range(max_iter):
                    if mean_err <= tolerance:
                        break

                    Lists, _, _ = Indices_generate_samples(
                        Indices,
                        current_sampling_rates,
                        self.keys,
                    )

                    # Convert each sampled coordinate list from (i, j) to flat pixel indices
                    flat_lists = [coords_to_flat(arr) for arr in Lists]

                    if any(len(arr) == 0 for arr in flat_lists):
                        break

                    iterator = product(*flat_lists)
                    if show_progress:
                        iterator = tqdm(
                            iterator,
                            desc=f"Cluster {k+1} | d={d}",
                            leave=False,
                        )

                    n_new = 0
                    X = np.empty((d, T_eff), dtype=flat_cdf.dtype)

                    for combo in iterator:
                        # combo is a tuple of flat pixel indices
                        if combo in seen:
                            continue
                        seen.add(combo)
                        n_new += 1

                        for i_d, pix in enumerate(combo):
                            X[i_d, :] = flat_cdf[pix, starts[i_d]:ends[i_d]]

                        chi_val = Multivar_Chi3(X.T, chi_q)

                        target_pix = combo[-1]
                        chi_count[target_pix] += 1
                        chi_sum[target_pix] += chi_val
                        chi_sumsq[target_pix] += chi_val * chi_val

                    if n_new == 0:
                        break

                    valid = chi_count > 0
                    chi_mean_flat = np.full(n_pix, np.nan, dtype=np.float64)
                    chi_var_flat = np.full(n_pix, np.nan, dtype=np.float64)

                    chi_mean_flat[valid] = chi_sum[valid] / chi_count[valid]
                    chi_var_flat[valid] = (
                        chi_sumsq[valid] / chi_count[valid]
                        - chi_mean_flat[valid] ** 2
                    )

                    chi_mean = chi_mean_flat.reshape(shape)
                    chi_var = chi_var_flat.reshape(shape)

                    if last_chi_mean is not None:
                        mean_err = np.nanmean(np.abs(chi_mean - last_chi_mean))
                        print(mean_err)
                        if np.isfinite(mean_err):
                            errors.append(float(mean_err))
                        else:
                            mean_err = np.inf

                    last_chi_mean = chi_mean.copy()
                    current_sampling_rates = current_sampling_rates + delta

                self.ERRORS[f"k{k+1}d{d}"] = errors

                valid_vals = chi_mean[np.isfinite(chi_mean)]
                if valid_vals.size == 0:
                    indices_h_k_q = np.empty((0, 2), dtype=np.int64)
                else:
                    thr = np.percentile(valid_vals, Percentile_sampling)
                    indices_h_k_q = np.argwhere(chi_mean > thr)
                    rng.shuffle(indices_h_k_q)

                Hk_mean.append(chi_mean)
                Hk_var.append(chi_var)

                Indices[str(d - 1)] = indices_h_k_q

                self.save_as_geo_image.to_image(
                    np.array([np.nan_to_num(chi_mean, nan=0.0)]),
                    str(out_folder / f"Chi_mean_d({d})_k({k}).tif"),
                )
                self.save_as_geo_image.to_image(
                    np.array([np.nan_to_num(chi_var, nan=0.0)]),
                    str(out_folder / f"Chi_var_d({d})_k({k}).tif"),
                )

            self.H_memory[str(k)] = {
                "mean": Hk_mean,
                "var": Hk_var,
            }
        self.lags_memory.append(lags) 


    def explore_hyper_dim_parallel(self,
                        D,
                        lags,
                        sampling_rates,
                        out_folder,
                        Percentile_sampling,
                        tolerance=0.01,
                        delta=0.01,
                        chi_q=0.95,
                        max_iter=100,
                        n_jobs=-1,
                        batch_size=2000,
                        backend="loky",   # "loky" for processes, "threading" if Multivar_Chi3 is mostly NumPy
                        show_progress=True,
                    ):
                        
        """
        Parallel optimized version of explore_hyper_dim.
        """
        
        self.ERRORS = {}
        self.Chi_conv = {}


        if not hasattr(self, "H_memory"):
            self.H_memory = {}
        

        shape = tuple(self.Shape)
        n_rows, n_cols = shape
        n_pix = n_rows * n_cols

        out_folder = Path(out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)

        lags = np.asarray(lags, dtype=int)
        base_sampling_rates = np.asarray(sampling_rates, dtype=float)

        # make sure array is contiguous for faster slicing / better sharing
        flat_cdf = np.ascontiguousarray(self.cdf_data)

        rng = np.random.default_rng()

        for k in range(self.opt_k):

            image_cent = self.centroids[k].reshape(shape)

            indices_lbl = np.argwhere(self.image_lbl == k)
            rng.shuffle(indices_lbl)

            thr_cent = np.percentile(image_cent, Percentile_sampling)
            indices_cnt = np.argwhere(image_cent > thr_cent)
            rng.shuffle(indices_cnt)

            Indices = {
                "0": indices_lbl,
                "1": indices_cnt,
            }

            Hk_mean = []
            Hk_var = []


            current_sampling_rates = base_sampling_rates.copy()

            for d in range(3, D + 1):

                errors = []
                chi_conv = []
                seen = set()

                chi_count = np.zeros(n_pix, dtype=np.int64)
                chi_sum = np.zeros(n_pix, dtype=np.float64)
                chi_sumsq = np.zeros(n_pix, dtype=np.float64)

                # cumulative lag windows for dimension d
                lag_prefix = np.zeros(d, dtype=np.int64)
                lag_prefix[1:] = np.cumsum(lags[:d - 1])
                total_lag = lag_prefix[-1]

                starts = self.I_start_time + lag_prefix
                ends = self.I_end_time - (total_lag - lag_prefix)

                T_eff = int(ends[0] - starts[0])
                if T_eff <= 0:
                    raise ValueError(
                        f"Non-positive effective sample length for d={d}. "
                        f"Check lags / I_start_time / I_end_time."
                    )

                last_chi_mean = None
                mean_err = np.inf

                for it in range(max_iter):
                    if mean_err <= tolerance:
                        break

                    Lists, _, _ = Indices_generate_samples(
                        Indices,
                        current_sampling_rates,
                        self.keys,
                    )

                    flat_lists = [_coords_to_flat(arr, self.Col) for arr in Lists]

                    if any(len(arr) == 0 for arr in flat_lists):
                        print(f"Skipping cluster {k}, d={d}: one sampled list is empty.")
                        break

                    combo_batches = _unique_combo_batches(
                        flat_lists=flat_lists,
                        seen=seen,
                        batch_size=batch_size,
                    )

                    results = Parallel(
                        n_jobs=n_jobs,
                        backend=backend,
                        pre_dispatch="2*n_jobs",
                    )(
                        delayed(_evaluate_combo_batch)(
                            batch, flat_cdf, starts, ends, chi_q
                        )
                        for batch in combo_batches
                    )

                    n_new = 0
                    for targets, chi_vals in results:
                        if len(targets) == 0:
                            continue
                        n_new += len(targets)

                        chi_count += np.bincount(targets, minlength=n_pix)
                        chi_sum += np.bincount(targets, weights=chi_vals, minlength=n_pix)
                        chi_sumsq += np.bincount(
                            targets, weights=chi_vals * chi_vals, minlength=n_pix
                        )

                    if n_new == 0:
                        print(f"Skipping cluster {k}, d={d}: no new unique combinations.")
                        break

                    valid = chi_count > 0

                    chi_mean_flat = np.full(n_pix, np.nan, dtype=np.float64)
                    chi_var_flat = np.full(n_pix, np.nan, dtype=np.float64)

                    chi_mean_flat[valid] = chi_sum[valid] / chi_count[valid]
                    chi_var_flat[valid] = (
                        chi_sumsq[valid] / chi_count[valid]
                        - chi_mean_flat[valid] ** 2
                    )

                    Chi_mean = chi_mean_flat.reshape(shape)
                    Chi_var = chi_var_flat.reshape(shape)
                    chi_conv.append(Chi_mean)
                    if last_chi_mean is not None:
                        err = np.abs(Chi_mean - last_chi_mean)
                        mean_err = np.nanmean(err)
                        errors.append(float(mean_err))
                        
                        if show_progress:
                            print(
                                f"Cluster {k+1}, d={d}, iter={it+1}, "
                                f"new combos={n_new}, mean_err={mean_err:.6f}"
                            )

                    last_chi_mean = Chi_mean.copy()
                    current_sampling_rates = current_sampling_rates + delta

                self.ERRORS[f'k{k+1}d{d}'] = errors
                self.Chi_conv[f'k{k+1}d{d}'] = chi_conv

                valid_vals = Chi_mean[np.isfinite(Chi_mean)]
                if valid_vals.size == 0:
                    print(f"Skipping cluster {k}, d={d}: no valid Chi statistics computed.")
                    indices_h_k_q = np.empty((0, 2), dtype=np.int64)
                else:
                    thr = np.percentile(valid_vals, Percentile_sampling)
                    indices_h_k_q = np.argwhere(Chi_mean > thr)
                    rng.shuffle(indices_h_k_q)

                Hk_mean.append(Chi_mean)
                Hk_var.append(Chi_var)

                Indices[str(d - 1)] = indices_h_k_q

                self.save_as_geo_image.to_image(
                    np.array([np.nan_to_num(Chi_mean, nan=0.0)]),
                    str(out_folder / f'Chi_mean_d({d})_k({k}).tif')
                )
                self.save_as_geo_image.to_image(
                    np.array([np.nan_to_num(Chi_var, nan=0.0)]),
                    str(out_folder / f'Chi_var_d({d})_k({k}).tif')
                )

            self.H_memory[str(k)] = {
                'mean': Hk_mean,
                'var': Hk_var,
            }
        self.lags_memory.append(lags) 

    def explore_for_station( self, lat_st , lon_st , D , lags,  Sampling_rates , Percentile_sampling , out_folder , name = 'Unknown Station'):

        # find the location 
        lags = lags[:D]
        i_st , j_st = np.argmin( (lat_st -self.lat)**2   )  , np.argmin( (lon_st - self.lon)**2   )

        # Extract the data for the station 
        
        self.data_st =  self.Data[  self.I_start_time   :   self.I_end_time  , i_st , j_st ]

        # find the index of station 
        self.st_index =   i_st * self.Shape[1 ] + j_st 

        # chaeck if the grid cell exist on land ?
        on_land = self.st_index in self.land_indices 
        if on_land is False: 
            print( 'the station index : ' , self.st_index , '  is not on land' )
            return 
            
        # find the connected grid cell with the highest tail dependence
        self.highest_idx_station  =  self.highest_chi_idx[  str(self.keys[self.st_index]) ] 

        image_chi_station = self.array_chi[ self.st_index ] 
        self.image_chi_station = np.reshape(image_chi_station , self.Shape)                                                       

        image_cent = np.reshape( self.image_chi_station  , self.Shape) 
       
        indices_lbl = np.array( [ [i_st]   ,
                                 [j_st ] ] ).T

        indices_cnt = np.where(image_cent  > np.percentile(image_cent, Percentile_sampling))
        indices_cnt = np.array( [indices_cnt[0]   ,  indices_cnt[1] ]  ).T
        
        np.random.shuffle(indices_cnt)  
        
        Indices =  {   '0'  :    indices_lbl ,
                       '1'  : indices_cnt   ,      } 
        
        Hd_mean = []
        
        Hd_var = []

        points = [    ]
        lat_lon_points = [  ( lat_st, lon_st  )]
        argmax_points = {}
        max_chi = {}

        for d in range(3, D+1):
            
            Lists , Indices_sample  , n_Edges_per_pixel = Indices_generate_samples( Indices  ,  Sampling_rates , self.keys)
            
            # Generate all possible combinations
            combinations = list(product(*Lists))
            
            # Convert to numpy array if needed
            combinations_array = np.array(combinations)
            
            Chi_list = Chi_list_generator(keys)
            chi_Cal = np.zeros(len(combinations_array))

            for i in tqdm(range( len(combinations_array))    , desc=f" Chi ...{d} Dimensions" , ncols= 100  ,colour='red'  ) :  
                X = []
                for i_d in range(d):
                    ii , jj = combinations_array[i][i_d][0] , combinations_array[i][i_d][1] 
                    index = ii * self.Col + jj 
                    # X.append(  self.Data[self.I_start_time + sum(lags[:i_d])  :   self.I_end_time - sum(lags[i_d:])   , ii, jj] ) 
                    X.append(  self.cdf_data[ index , self.I_start_time + sum(lags[:i_d])  :   self.I_end_time - sum(lags[i_d:])      ] ) 
                    
                X = np.array(X)             
                #print(X.shape)
                #dfX = pd.DataFrame( X.T)
                chi_Cal[i] =  Multivar_Chi3( X.T , self.percentile )

                Chi_list[  str(  np.array( [ii, jj] )  )  ].append( chi_Cal[i] )  # Chi.Multivar_Chi( dfX , 0.95 ) )
        
            Chi_mean = []
            Chi_var  = [] 
            argmax = np.argmax(chi_Cal)
            argmax_points[str(d)] = combinations_array[argmax]
            max_chi[ str(d)] = chi_Cal[argmax]
 

            for k in Chi_list:
                Chi_mean.append( np.mean(Chi_list[k]) )
                Chi_var.append( np.var(Chi_list[k]) )
                
            
            Chi_mean = np.reshape( Chi_mean , Shape) 
            Chi_var = np.reshape( Chi_var , Shape) 
            
            
            indices_h_k_q = np.where(Chi_mean  > np.percentile(Chi_mean, Percentile_sampling))
            indices_h_k_q = np.array( [indices_h_k_q[0]   ,  indices_h_k_q[1] ]  ).T
            
            np.random.shuffle(indices_h_k_q)  
                
            Hd_mean.append( Chi_mean)
            Hd_var.append( Chi_var)
            
            Indices[str(d-1)] = indices_h_k_q
            
            self.save_as_geo_image.to_image( np.array([Chi_mean] ) , out_folder + f'Chi_mean_d({d})_{name}.tif' )
            self.save_as_geo_image.to_image( np.array([Chi_var] ) , out_folder + f'Chi_var_d({d})_{name}.tif' )

            # Flattened index of max element
            flat_index = np.argmax(Chi_mean)
            
            # Convert to row, col
            self.max_i, self.max_j = np.unravel_index(flat_index, self.Shape)

    
            points.append(  (  self.max_i   ,  self.max_j  )    )
            
            lat_lon_points.append(  (  self.lat[self.max_i]   ,  self.lon[self.max_j] ) )

        
        self.H_memory[ name ] = {}
        self.H_memory[ name ]['chi_mean'] =  Hd_mean
        self.H_memory[ name ]['chi_var'] =  Hd_var
        self.H_memory[ name ]['argmax_points'] =  argmax_points
        self.H_memory[ name ]['max_chi'] =  max_chi

        self.H_memory[ name ]['lat_lon_points'] =  lat_lon_points
        self.H_memory[ name ]['lat'] =  lat_st
        self.H_memory[ name ]['lon'] =  lon_st
        self.H_memory[ name ]['D']  = D
        self.H_memory[ name ]['lags']  = lags
        self.H_memory[ name ]['Sampling_rates']  = Sampling_rates
        self.H_memory[ name ]['Percentile_sampling']  = Percentile_sampling
    

        line = LineString([(lon_, lat_) for lat_, lon_ in lat_lon_points])
        gdf_line = gpd.GeoDataFrame(index=[0], geometry=[line], crs="EPSG:4326")
        gdf_line.to_file( out_folder + f'{name}_{lags}.shp'   , driver="ESRI Shapefile")


        df_p = pd.DataFrame( lat_lon_points  , columns = ['lat' , 'lon' ] )
        gdf_points = gpd.GeoDataFrame(df_p, geometry=[Point(xy) for xy in zip(df_p["lon"], df_p["lat"])], crs="EPSG:4326")
        gdf_points.to_file(out_folder + f'{name}_{lags}_points.shp'   , driver="ESRI Shapefile")



    def explore_for_station_fast(
        self,
        lat_st,
        lon_st,
        D,
        lags,
        Sampling_rates,
        Percentile_sampling,
        out_folder,
        name="Unknown Station",
        chi_q=None,
        n_jobs=-1,
        batch_size=2000,
        backend="loky",
        save_shp  = False ,
        save_image = False  # use "threading" if Multivar_Chi3 is mostly NumPy
    ):
        """
        Faster and parallel version of explore_for_station.
        """

        if chi_q is None:
            chi_q = getattr(self, "percentile", 0.95)

        shape = tuple(self.Shape)
        nrows, ncols = shape
        npix = nrows * ncols

        out_folder = Path(out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)

        # For dimension d, we need d-1 lags
        lags = np.asarray(lags[: max(D - 1, 0)], dtype=int)
        if D >= 3 and len(lags) < D - 1:
            raise ValueError(f"Need at least D-1 lags. Got len(lags)={len(lags)} for D={D}.")

        i_st = int(np.argmin((lat_st - self.lat) ** 2))
        j_st = int(np.argmin((lon_st - self.lon) ** 2))

        self.data_st = self.Data[self.I_start_time:self.I_end_time, i_st, j_st]
        self.st_index = i_st * self.Shape[1] + j_st

        # Faster membership check
        if not hasattr(self, "_land_index_set"):
            self._land_index_set = set(self.land_indices)

        if self.st_index not in self._land_index_set:
            print("the station index:", self.st_index, "is not on land")
            return

        self.highest_idx_station = self.highest_chi_idx[str(self.keys[self.st_index])]

        image_chi_station = self.array_chi[self.st_index]
        self.image_chi_station = np.reshape(image_chi_station, self.Shape)
        image_cent = self.image_chi_station

        indices_lbl = np.array([[i_st, j_st]], dtype=np.int64)

        thr0 = np.percentile(image_cent, Percentile_sampling)
        indices_cnt = np.argwhere(image_cent > thr0)

        rng = np.random.default_rng()
        rng.shuffle(indices_cnt)

        Indices = {
            "0": indices_lbl,
            "1": indices_cnt,
        }

        Hd_mean = []
        Hd_var = []

        lat_lon_points = [(lat_st, lon_st)]
        argmax_points = {}
        max_chi = {}

        flat_cdf = np.ascontiguousarray(self.cdf_data)

        for d in range(3, D + 1):
            Lists, _, _ = Indices_generate_samples(Indices, Sampling_rates, self.keys)

            flat_lists = [_coords_to_flat(arr, self.Col) for arr in Lists]

            if any(len(arr) == 0 for arr in flat_lists):
                print(f"Skipping d={d}: one sampled list is empty.")
                break

            # Precompute lag-aligned windows once
            lag_prefix = np.zeros(d, dtype=np.int64)
            lag_prefix[1:] = np.cumsum(lags[: d - 1])
            total_lag = lag_prefix[-1]

            starts = self.I_start_time + lag_prefix
            ends = self.I_end_time - (total_lag - lag_prefix)

            t_eff = int(ends[0] - starts[0])
            if t_eff <= 0:
                raise ValueError(
                    f"Non-positive effective sample length for d={d}. "
                    "Check I_start_time, I_end_time, and lags."
                )

            chi_count = np.zeros(npix, dtype=np.int64)
            chi_sum = np.zeros(npix, dtype=np.float64)
            chi_sumsq = np.zeros(npix, dtype=np.float64)

            best_combo_d = None
            best_chi_d = -np.inf

            combo_batches = _unique_combo_batches2(flat_lists, batch_size)

            results = Parallel(
                n_jobs=n_jobs,
                backend=backend,
                pre_dispatch="2*n_jobs",
            )(
                delayed(_eval_station_batch)(batch, flat_cdf, starts, ends, chi_q)
                for batch in combo_batches
            )

            n_done = 0
            for targets, chi_vals, best_combo_batch, best_chi_batch in results:
                if len(targets) == 0:
                    continue

                n_done += len(targets)

                chi_count += np.bincount(targets, minlength=npix)
                chi_sum += np.bincount(targets, weights=chi_vals, minlength=npix)
                chi_sumsq += np.bincount(targets, weights=chi_vals * chi_vals, minlength=npix)

                if best_chi_batch > best_chi_d:
                    best_chi_d = float(best_chi_batch)
                    best_combo_d = best_combo_batch

            if n_done == 0:
                print(f"Skipping d={d}: no valid combinations were evaluated.")
                break

            valid = chi_count > 0
            Chi_mean_flat = np.full(npix, np.nan, dtype=np.float64)
            Chi_var_flat = np.full(npix, np.nan, dtype=np.float64)

            Chi_mean_flat[valid] = chi_sum[valid] / chi_count[valid]
            Chi_var_flat[valid] = chi_sumsq[valid] / chi_count[valid] - Chi_mean_flat[valid] ** 2

            Chi_mean = Chi_mean_flat.reshape(shape)
            Chi_var = Chi_var_flat.reshape(shape)

            argmax_points[str(d)] = _flat_combo_to_coords(best_combo_d, self.Col)
            max_chi[str(d)] = best_chi_d if np.isfinite(best_chi_d) else np.nan

            valid_vals = Chi_mean[np.isfinite(Chi_mean)]
            if valid_vals.size == 0:
                print(f"Skipping d={d}: no valid Chi statistics computed.")
                break

            thr = np.percentile(valid_vals, Percentile_sampling)
            indices_h_k_q = np.argwhere(Chi_mean > thr)
            rng.shuffle(indices_h_k_q)

            Hd_mean.append(Chi_mean)
            Hd_var.append(Chi_var)

            Indices[str(d - 1)] = indices_h_k_q
            if save_image : 

                self.save_as_geo_image.to_image(
                    np.array([np.nan_to_num(Chi_mean, nan=0.0)]),
                    str(out_folder / f"Chi_mean_d({d})_{name}.tif"),
                )
                self.save_as_geo_image.to_image(
                    np.array([np.nan_to_num(Chi_var, nan=0.0)]),
                    str(out_folder / f"Chi_var_d({d})_{name}.tif"),
                )

            flat_index = int(np.nanargmax(Chi_mean))
            max_i, max_j = np.unravel_index(flat_index, self.Shape)
            lat_lon_points.append((self.lat[max_i], self.lon[max_j]))

        if not hasattr(self, "H_memory"):
            self.H_memory = {}

        self.H_memory[name] = {
            "chi_mean": Hd_mean,
            "chi_var": Hd_var,
            "argmax_points": argmax_points,
            "max_chi": max_chi,
            "lat_lon_points": lat_lon_points,
            "lat": lat_st,
            "lon": lon_st,
            "D": D,
            "lags": lags.tolist(),
            "Sampling_rates": Sampling_rates,
            "Percentile_sampling": Percentile_sampling,
        }

        lag_tag = "_".join(map(str, lags.tolist())) if len(lags) else "nolag"
        if save_shp :
            line = LineString([(lon_, lat_) for lat_, lon_ in lat_lon_points])
            gdf_line = gpd.GeoDataFrame(index=[0], geometry=[line], crs="EPSG:4326")
            gdf_line.to_file(str(out_folder / f"{name}_{lag_tag}.shp"), driver="ESRI Shapefile")

            df_p = pd.DataFrame(lat_lon_points, columns=["lat", "lon"])
            gdf_points = gpd.GeoDataFrame(
                df_p,
                geometry=[Point(xy) for xy in zip(df_p["lon"], df_p["lat"])],
                crs="EPSG:4326",
            )
            gdf_points.to_file(
                str(out_folder / f"{name}_{lag_tag}_points.shp"),
                driver="ESRI Shapefile",
            )

    def explore_for_station_V2(self, lat_st, lon_st, D, lags, Sampling_rates,Percentile_sampling, out_folder, name='Unknown Station' , save_shp = False , save_image = False):

                            
        # --- basics ---
        lags = lags[:D]  # keep your behaviour
        i_st = int(np.argmin((lat_st - self.lat) ** 2))
        j_st = int(np.argmin((lon_st - self.lon) ** 2))

        # Extract station series
        self.data_st = self.Data[self.I_start_time:self.I_end_time, i_st, j_st]

        # Station flat index
        self.st_index = i_st * self.Shape[1] + j_st

        # Check land
        if self.st_index not in self.land_indices:
            print('the station index : ', self.st_index, '  is not on land')
            return

        # Find highest tail dependence neighbour (your code)
        self.highest_idx_station = self.highest_chi_idx[str(self.keys[self.st_index])]

        image_chi_station = self.array_chi[self.st_index]
        self.image_chi_station = np.reshape(image_chi_station, self.Shape)

        image_cent = np.reshape(self.image_chi_station, self.Shape)

        indices_lbl = np.array([[i_st], [j_st]]).T

        indices_cnt = np.where(image_cent > np.percentile(image_cent, Percentile_sampling))
        indices_cnt = np.array([indices_cnt[0], indices_cnt[1]]).T
        np.random.shuffle(indices_cnt)

        Indices = {
            '0': indices_lbl,
            '1': indices_cnt,
        }

        Hd_mean, Hd_var = [], []
        points = []
        lat_lon_points = [(lat_st, lon_st)]
        argmax_points = {}
        max_chi = {}

        Shape = self.Shape  # fix undefined variable in your original
        Col = self.Col      # assumes self.Col == number of columns

        # -------- main loop over dimensions --------
        for d in range(3, D + 1):

            Lists, Indices_sample, n_Edges_per_pixel = Indices_generate_samples(
                Indices, Sampling_rates, self.keys
            )

            # Use only the first (d-1) lags for d-dim construction (this avoids summing unused lags)
            lags_d = lags[:max(0, d - 1)]
            total_lag = int(np.sum(lags_d)) if len(lags_d) else 0

            # Precompute offsets so we don't do sum(lags[:i_d]) inside every task
            start_off = [0] * d
            run = 0
            for k in range(1, d):
                run += int(lags_d[k - 1])
                start_off[k] = run
            end_off = [total_lag - start_off[k] for k in range(d)]

            Chi_list = Chi_list_generator(self.keys)

            # Total number of combinations (for prealloc + tqdm)
            n_comb = math.prod(len(L) for L in Lists)
            chi_Cal = np.zeros(n_comb, dtype=float)

            # generator (no giant list/product materialization)
            comb_iter = product(*Lists)

            # --- parallel compute ---
            max_workers = os.cpu_count()  # or set e.g., 8
            chunksize = 256               # tune: 64..2000

            best_chi = -np.inf
            best_combo = None

            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_chi_ctx,
                initargs=(self.cdf_data, Col, self.I_start_time, self.I_end_time, start_off, end_off, self.percentile)
            ) as ex:

                it = ex.map(_chi_worker, comb_iter, chunksize=chunksize)

                for idx, (key, chi, combo) in enumerate(
                    tqdm(it, total=n_comb, desc=f" Chi ...{d} Dimensions", ncols=100, colour='red')
                ):
                    chi_Cal[idx] = chi
                    Chi_list[key].append(chi)

                    if chi > best_chi:
                        best_chi = chi
                        best_combo = combo

            # store argmax info (same meaning as your original)
            argmax_points[str(d)] = np.array(best_combo, dtype=int) if best_combo is not None else None
            max_chi[str(d)] = float(best_chi) if np.isfinite(best_chi) else None

            # build mean/var images
            Chi_mean = []
            Chi_var = []
            for k in Chi_list:
                Chi_mean.append(np.mean(Chi_list[k]) if len(Chi_list[k]) else np.nan)
                Chi_var.append(np.var(Chi_list[k]) if len(Chi_list[k]) else np.nan)

            Chi_mean = np.reshape(Chi_mean, Shape)
            Chi_var  = np.reshape(Chi_var, Shape)

            indices_h_k_q = np.where(Chi_mean > np.percentile(Chi_mean[~np.isnan(Chi_mean)], Percentile_sampling))
            indices_h_k_q = np.array([indices_h_k_q[0], indices_h_k_q[1]]).T
            np.random.shuffle(indices_h_k_q)

            Hd_mean.append(Chi_mean)
            Hd_var.append(Chi_var)

            Indices[str(d - 1)] = indices_h_k_q
            if save_image :
                self.save_as_geo_image.to_image(np.array([Chi_mean]), out_folder + f'Chi_mean_d({d})_{name}.tif')
                self.save_as_geo_image.to_image(np.array([Chi_var]),  out_folder + f'Chi_var_d({d})_{name}.tif')

            flat_index = int(np.nanargmax(Chi_mean))
            self.max_i, self.max_j = np.unravel_index(flat_index, Shape)

            points.append((self.max_i, self.max_j))
            lat_lon_points.append((self.lat[self.max_i], self.lon[self.max_j]))

        # -------- save memory outputs --------
        self.H_memory[name] = {}
        self.H_memory[name]['chi_mean'] = Hd_mean
        self.H_memory[name]['chi_var'] = Hd_var
        self.H_memory[name]['argmax_points'] = argmax_points
        self.H_memory[name]['max_chi'] = max_chi
        self.H_memory[name]['lat_lon_points'] = lat_lon_points
        self.H_memory[name]['lat'] = lat_st
        self.H_memory[name]['lon'] = lon_st
        self.H_memory[name]['D'] = D
        self.H_memory[name]['lags'] = lags
        self.H_memory[name]['Sampling_rates'] = Sampling_rates
        self.H_memory[name]['Percentile_sampling'] = Percentile_sampling

        # -------- shapefiles --------
        if save_shp :
            line = LineString([(lon_, lat_) for lat_, lon_ in lat_lon_points])
            gdf_line = gpd.GeoDataFrame(index=[0], geometry=[line], crs="EPSG:4326")
            gdf_line.to_file(out_folder + f'{name}_{lags}.shp', driver="ESRI Shapefile")

            df_p = pd.DataFrame(lat_lon_points, columns=['lat', 'lon'])
            gdf_points = gpd.GeoDataFrame(
                df_p,
                geometry=[Point(xy) for xy in zip(df_p["lon"], df_p["lat"])],
                crs="EPSG:4326"
            )
            gdf_points.to_file(out_folder + f'{name}_{lags}_points.shp', driver="ESRI Shapefile")

    def save_shp_points(self , lat_lon_points , file ):
            
            line = LineString([(lon_, lat_) for lat_, lon_ in lat_lon_points])
            gdf_line = gpd.GeoDataFrame(index=[0], geometry=[line], crs="EPSG:4326")
            gdf_line.to_file(file , driver="ESRI Shapefile")

            df_p = pd.DataFrame(lat_lon_points, columns=['lat', 'lon'])
            gdf_points = gpd.GeoDataFrame(
                df_p,
                geometry=[Point(xy) for xy in zip(df_p["lon"], df_p["lat"])],
                crs="EPSG:4326"
            )
            gdf_points.to_file( file + '_points.shp', driver="ESRI Shapefile")

    def predict( self, name , predict_date_start , predict_date_end ):

        lags = self.H_memory[ name ]['lags']
        lat_st= self.H_memory[ name ]['lat'] 
        lon_st = self.H_memory[ name ]['lon'] 
        
        i_st , j_st = np.argmin( (lat_st -self.lat)**2   )  , np.argmin( (lon_st - self.lon)**2   )

        self.prediction =  {} 
        I_Start = np.where( date == predict_date_start )[0][0] 
        I_end = np.where( date == predict_date_end )[0][0] 
        
        max_i , max_j = self.H_memory[ name ]['points'][0][0]  , self.H_memory[ name ]['points'][0][1]
        
        prediction = self.Data[ I_Start - lags[0] : I_end  - lags[0] , max_i , max_j ] 
        
        self.H_memory[ name ]['resamples_lag'] = self.Data[ I_Start - lags[0] : I_end  - lags[0] , i_st , j_st ] 
        
        self.H_memory[ name ]['resamples'] = self.Data[ I_Start : I_end   , i_st , j_st ] 
        
        self.prediction[name]  = prediction
        
        self.H_memory[name]['date'] = self.date[ I_Start - lags[0] : I_end  - lags[0]]
        
        return prediction 
    
    def save(self, filepath):
        self.fp = filepath
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Model saved to {filepath}")

    def save_update(self ):
        with open(self.fp, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Model saved to {self.fp}")

    # Load method (classmethod so we can call RainModel.load(...))
    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        print(f"✅ Model loaded from {filepath}")
        return obj
    
