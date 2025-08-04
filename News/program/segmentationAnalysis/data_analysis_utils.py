import pandas as pd
import numpy as np

from old.resultsAnalysis.numberWithUncertainty import numberWithUncertainty

def reduce_df_dim(df:pd.DataFrame, column:str):
    """
    Data Frames Cells Can Be Filled With Data Structures that ALSO representes Data Frames
    
    If you want to Mount a new Data Frame with the Cells Content You Can remove the column dimension by Selecting Just 1 Column and eval the Data Cell Content Into Columns
    """
    
    # Tranposing To Avoid Data Cells Indexes to Become Columns Labels
    return pd.DataFrame(df[column].to_dict()).T


def mean_w_precision(collection):
    collection = np.asanyarray(collection)
    
    return numberWithUncertainty(
        collection.mean(),
        collection.std()
    )
    
    
def quantify_condo_ds_mean_quality_prop(ds_sample, quality_prop="accuracy"):
    classification_metrics_df = reduce_df_dim(pd.DataFrame(ds_sample), "classification_evaluation_metrics")
    return classification_metrics_df[quality_prop].mean()
    # return mean_w_precision(classification_metrics_df[quality_prop])