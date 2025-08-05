from typing import Callable, Dict
import numpy as np
import pandas as pd
import types as t
from bootstrap_sampling import generate_bootstrap_samples_idxs, sample_ds
from data_analysis_utils import cache_result, reduce_df_dim, mean_w_precision
from analysisTypes import *

def oob_362_correction(oob_err, resubstitution_err):
    return (0.368 * resubstitution_err) + (0.632 * oob_err)

def no_info_err_rate_4_condo_ds(condo_ds:condo_ds_t):
    # Get Confusion Matrix Values From condo_ds
    classification_metrics = reduce_df_dim(pd.DataFrame(condo_ds), "classification_evaluation_metrics")
    classification_distribution = reduce_df_dim(classification_metrics, "typesDistribution")
    resubstitution_confusion_matrix = classification_distribution.apply(lambda col: col.sum())
    
    TP, TN = resubstitution_confusion_matrix["TP"], resubstitution_confusion_matrix["TN"]
    FP, FN = resubstitution_confusion_matrix["FP"], resubstitution_confusion_matrix["FN"]
    TOTAL_CLASSIFICATIONS = resubstitution_confusion_matrix["TOTAL_CLASSIFICATIONS"]
    
    actual_classes = {"P": TP + FN,  "N": FP + TN}
    predicted_classes = {"P": TP + FP,  "N": TN + FN}   

    # Compute no_info_err_rate
    no_info_err_rate = 0
    for crr_class in actual_classes.keys():
        class_proportion = actual_classes[crr_class] / TOTAL_CLASSIFICATIONS
        class_prediction_proportion_to_truth = predicted_classes[crr_class] / TOTAL_CLASSIFICATIONS
        
        no_info_err_rate += class_proportion * (1 - class_prediction_proportion_to_truth)
        
    return no_info_err_rate


def oob_362_plus_correction(oob_err, resubstitution_err, no_info_err_rate):
    relative_overfitting_rate = (oob_err - resubstitution_err) / (no_info_err_rate - resubstitution_err)
    coeff = 0.632 / (1 - 0.368 * relative_overfitting_rate)
    
    return (1 - coeff) * resubstitution_err + coeff * oob_err


def oob_362_quality_prop_measurement(dataset: t.List[any], bootstrap_samples_idxs:t.List[t.Tuple[t.List[int], t.List[int]]], props_extraction_method:Dict[str, Callable[[pd.DataFrame, t.Optional[Dict]], pd.DataFrame]]={}):
        
    # Compute OOB 362 Corrected Value 
    # (Each Metric from props_extraction_method; 
    # (Each Train & Test Bootrap Sample from bootstrap_samples_idxs)
    props_resubstitution = {}
    
    props_values = {
        "test": {},
        "train": {}
    }
    
    resubstitution_no_info_err_rate = no_info_err_rate_4_condo_ds(dataset)
    
    cache = {}
    for crr_prop, get_crr_prop_from_df in props_extraction_method.items():
        props_resubstitution[crr_prop] = get_crr_prop_from_df(dataset, cache).mean()
        
        props_values["train"][crr_prop] = []
        props_values["test"][crr_prop] = []
    del cache
    
    for train_samples_idxs, test_samples_idxs in bootstrap_samples_idxs:
        samples = {
            "train": sample_ds(dataset, train_samples_idxs),
            "test": sample_ds(dataset, test_samples_idxs)
        }
        
        for type_of_sample in ["train", "test"]:
            cache = {}
            for crr_prop, get_crr_prop_from_df in props_extraction_method.items():
                prop_sample_mean = get_crr_prop_from_df(samples[type_of_sample], cache).mean()
                    
                prop_corrected = oob_362_plus_correction(prop_sample_mean, props_resubstitution[crr_prop], resubstitution_no_info_err_rate)
                props_values[type_of_sample][crr_prop].append(prop_corrected)
        del cache
       
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

def compute_oob_362_results_over_methods(color_space_oriented_df, f_to_get_metric_from_condo_ds, sample_amount=300):
    oob_362_plus_results_over_methods = {}
    
    # Use Sample Indexes to Make Samples Lightier to Save
    @cache_result("bootstrap_samples_idxs.data")
    def compute_bootstrap_samples_idxs():
        return generate_bootstrap_samples_idxs(14, sample_amount)
    
    bootstrap_samples_idxs = compute_bootstrap_samples_idxs()
        
    ### Processing OOB Results For Samples (each method in each color space) ---
    for color_space in color_space_oriented_df.keys():
        
        methods = color_space_oriented_df[color_space]
        oob_362_plus_results_over_methods[color_space] = {}
        
        for method_name in methods.keys():
            method_ds: condo_ds_t = color_space_oriented_df[color_space][method_name]
            
            method_oob_corrected_ret = oob_362_quality_prop_measurement(method_ds, bootstrap_samples_idxs, f_to_get_metric_from_condo_ds)
            oob_362_plus_results_over_methods[color_space][method_name] = method_oob_corrected_ret
            
    return oob_362_plus_results_over_methods