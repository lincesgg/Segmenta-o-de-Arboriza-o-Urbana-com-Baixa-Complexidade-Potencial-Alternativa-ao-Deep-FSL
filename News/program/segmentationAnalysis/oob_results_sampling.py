from typing import Callable, Dict
import numpy as np
import pandas as pd
import types as t
from bootstrap_sampling import sample_ds
from data_analysis_utils import reduce_df_dim
from data_analysis_utils import mean_w_precision
from analysisTypes import *

import line_profiler

def no_info_err_rate_4_condo_ds(condo_ds):
    classification_metrics = reduce_df_dim(pd.DataFrame(condo_ds), "classification_evaluation_metrics")
    classification_distribution = reduce_df_dim(classification_metrics, "typesDistribution")
    resubstitution_confusion_matrix = classification_distribution.apply(lambda x: x.sum())
    
    TP, TN, FP, FN = resubstitution_confusion_matrix["TP"], resubstitution_confusion_matrix["TN"], resubstitution_confusion_matrix["FP"], resubstitution_confusion_matrix["FN"]
    actual_classes = {"P": TP + FN,  "N": FP + TN}
    predicted_classes = {"P": TP + FP,  "N": TN + FN}   

    no_info_err_rate = 0
    for crr_class in actual_classes.keys():
        class_proportion_in_ds = actual_classes[crr_class] / resubstitution_confusion_matrix["TOTAL_CLASSIFICATIONS"]
        class_prediction_proportion_in_ds = predicted_classes[crr_class] / resubstitution_confusion_matrix["TOTAL_CLASSIFICATIONS"]
        
        no_info_err_rate += class_proportion_in_ds * (1 - class_prediction_proportion_in_ds)
        
    return no_info_err_rate


def oob_362_correction(oob_err, resubstitution_err):
    return (0.368 * resubstitution_err) + (0.632 * oob_err)


def oob_362_plus_correction(oob_err, resubstitution_err, no_info_err_rate):
    
    relative_overfitting_rate = (oob_err - resubstitution_err) / (no_info_err_rate - resubstitution_err)
    coeff = 0.632 / (1 - 0.368 * relative_overfitting_rate)
    
    return (1 - coeff) * resubstitution_err + coeff * oob_err


@line_profiler.profile
def oob_362_quality_prop_measurement(dataset: t.List[any], bootstrap_samples_idxs:t.List[t.Tuple[t.List[int], t.List[int]]], props_extraction_method:Dict[str, Callable[[pd.DataFrame], pd.DataFrame]]={}):
        
    # Compute OOB 362 Corrected Value 
    # (Each Metric from props_extraction_method; 
    # (Each Train & Test Bootrap Sample from bootstrap_samples_idxs)
    props_resubstitution = {}
    
    props_values = {
        "test": {},
        "train": {}
    }
    
    resubstitution_no_info_err_rate = no_info_err_rate_4_condo_ds(dataset)
    
    for crr_prop, get_crr_prop_from_df in props_extraction_method.items():
        props_resubstitution[crr_prop] = get_crr_prop_from_df(dataset).mean()
        
        props_values["train"][crr_prop] = []
        props_values["test"][crr_prop] = []
        
    
    for train_samples_idxs, test_samples_idxs in bootstrap_samples_idxs:
        samples = {
            "train": sample_ds(dataset, train_samples_idxs),
            "test": sample_ds(dataset, test_samples_idxs)
        }
        
        for type_of_sample in ["train", "test"]:
            for crr_prop, get_crr_prop_from_df in props_extraction_method.items():
                prop_sample_mean = get_crr_prop_from_df(samples[type_of_sample]).mean()
                    
                prop_corrected = oob_362_plus_correction(prop_sample_mean, props_resubstitution[crr_prop], resubstitution_no_info_err_rate)
                props_values[type_of_sample][crr_prop].append(prop_corrected)
                
       
    # Compute Results: OOB 362 Corrected Value Mean
    # (Each Metric from props_extraction_method; 
    # (Each Train & Test Bootrap Sample from bootstrap_samples_idxs)
    results =  {
        "test": {},
        "train": {}
    }
       
    for type_of_sample in ["train", "test"]:
        results[type_of_sample] = {}

        for crr_prop in props_extraction_method.keys():
            oob_values = np.array(props_values[type_of_sample][crr_prop])
            results[type_of_sample][f"{crr_prop}_ds"] = oob_values
            results[type_of_sample][f"mean_{crr_prop}"] = mean_w_precision(oob_values)
                
        
    return results


