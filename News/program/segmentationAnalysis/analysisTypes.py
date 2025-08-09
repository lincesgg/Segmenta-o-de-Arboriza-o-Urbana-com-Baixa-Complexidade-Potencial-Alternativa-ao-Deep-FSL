import typing as t
import enum

class typesDistribution_t(t.TypedDict):
    TN: int
    TP: int
    FN: int
    FP: int
    TOTAL_CLASSIFICATIONS: int

class classification_evaluation_metrics_t(t.TypedDict):
    precision: float
    recall: float
    truePositiveRate: float
    falsePositiveRate: float
    accuracy: float
    balancedAccuracy: float
    jaccardIndex: float
    F1Score: float
    overselectionIndex: float
    typesDistribution: typesDistribution_t
    
    
# Alredy Called Of Data Point, condo data, ...
class condo_results_t(t.TypedDict):
    decimal_places_precision: int
    
    ROI_percentage: float
    ROI_area: float
    
    real_ROI_percentage: float
    real_ROI_area: float
    
    analysis_delimitation_area: float
    real_analysis_delimitation_area: float
    
    delimitation_area_accuracy: float
    ROI_area_accuracy: float
    ROI_percentage_accuracy: float  
    
    classification_evaluation_metrics: classification_evaluation_metrics_t

    kmeans_davies_boudin: t.NotRequired[float] = None #K-means + Inverse-KNN Only
    time_to_analyze: float
    
class valid_color_spaces_enum(enum.Enum):
    SHLS_results = 0
    SHSV_results = 1
    LAB_results = 2
    LUV_results = 3
    RGB_results = 4
    OKLAB_results = 5
valid_color_spaces_t = t.NewType("valid_color_spaces_t", t.Literal[*map(lambda x: x.name, list(valid_color_spaces_enum))]) 
    
# FIXME - How I will Call THe results oof All Condos For a specefic tuple (Color Space, Tecnique)?
# TODO - For now, I was calling it condo ds, method (?)
# But I will prob. Change it to (Config Samples Results) or something like it
condo_ds_t = t.NewType("condo_ds_t", t.List[condo_results_t])

color_spaces_to_condo_ds_t = t.NewType("color_spaces_to_condo_ds_t", t.Dict[valid_color_spaces_t, condo_ds_t])
techniques_to_condo_ds_t = t.NewType("techniques_to_condo_ds_t", t.Dict[str, condo_ds_t])

# FIXME: technique (?)
# Methods Results Collections
technique_oriented_results_t = t.NewType("technique_oriented_results_t", t.Dict[str, color_spaces_to_condo_ds_t])
color_space_oriented_results_t = t.NewType("color_space_oriented_results_t", t.Dict[valid_color_spaces_t, techniques_to_condo_ds_t])


