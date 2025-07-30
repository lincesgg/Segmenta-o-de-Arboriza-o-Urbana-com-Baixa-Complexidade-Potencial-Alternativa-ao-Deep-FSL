import math
from analysisTypes import classification_evaluation_metrics_t

def compute_normPs(confusion_matrix: classification_evaluation_metrics_t):
    TP, TN, FP, FN = confusion_matrix["TP"], confusion_matrix["TN"], confusion_matrix["FP"], confusion_matrix["FN"]
    P = TP + FN
    N = TN + FP 
    FPR = FP / N
    FNR = FN / P
    
    # Old
    if ((FPR + FNR) == 0):
        return 0.5
    return (((FPR - FNR) / (FPR + FNR)) + 1) / 2
    # raise RuntimeError("Actually this is a difference normalization of the errors")
    
# Prevalencia de Sobreseleção
def compute_pS(confusion_matrix: classification_evaluation_metrics_t):
    TP, TN, FP, FN = confusion_matrix["TP"], confusion_matrix["TN"], confusion_matrix["FP"], confusion_matrix["FN"]
    P = TP + FN
    N = TN + FP 
    FPR = FP / N
    FNR = FN / P
    
    if (FPR + FNR == 0):
        # As There's no Error, there is no valid prevalence
        return -1
    
    Ps = FPR / (FPR + FNR)
        
    return Ps
    
def compute_specificity(confusion_matrix: classification_evaluation_metrics_t):
    TP, TN, FP, FN = confusion_matrix["TP"], confusion_matrix["TN"], confusion_matrix["FP"], confusion_matrix["FN"]
    P = TP + FN
    N = TN + FP 
    
    TNR = TN / (TN + FP)
    
    return TNR
    

def compute_MCC(confusion_matrix: classification_evaluation_metrics_t):
    TP, TN, FP, FN = confusion_matrix["TP"], confusion_matrix["TN"], confusion_matrix["FP"], confusion_matrix["FN"]
    
    # Test For Undefined Cases and Provide Adequate Return
    # TODO: Refleta sobre isso depois - O que garante q essas sejam todas as possibilidades possíveis (ou pelo menos todas as possibilidade q geram indefinição)
    # https://stats.stackexchange.com/questions/73000/denominator-is-zero-for-matthews-correlation-coefficient-and-f-measure
    if (TP == 0):
        # Classified All As Negative
        if (FP == 0):
            if (FN == 0):
                return 1.0
            elif (TN == 0):
                return -1.0
            else:
                return 0.0

        # Then, All True Classes are in TN                
        if (FN == 0):
            if (FP == 0):
                return 1.0
            elif (TN == 0):
                return -1.0
            else:
                return 0.0
            
    if (TN == 0):
        # Then, All True Classes are in TP    
        if (FP == 0):
            if (FN == 0):
                return 1.0
            elif (TP == 0):
                return -1.0
            else:
                return 0.0
            
        # Classified All As Positive
        if (FN == 0):
            if (FP == 0):
                return 1.0
            elif (TP == 0):
                return -1.0
            else:
                return 0.0
            
    # If Return Is not Undefined, Use Equation
    return  ((TP * TN) - (FP * FN)) / (math.sqrt( (TP + FP) * (TP+ FN) * (TN + FP) * (TN + FN)))

