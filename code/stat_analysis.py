import os as os
import pandas as pd
import numpy as np
from scipy import stats
from scipy import special
from scipy.optimize import linear_sum_assignment
from MLstatkit.stats import Delong_test

from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import seaborn as sns


def get_bootstrap_sample(X_df, y_df, n_splits):
    """
    Returns a stratified bootstrap sample of X and y,
    preserving class proportions.
    """
    
    df = pd.concat([X_df, y_df], axis=1)
    
    while True:
        sampled_df = df.sample(n=len(df), replace=True).reset_index(drop=True)
        if sampled_df[y_df.name].sum()>=n_splits:
            break;

    return sampled_df[X_df.columns], sampled_df[y_df.name]


def percentile_bootstrap_ci(X_df, y_df, statistic, n_splits, n_iterations, alpha=0.05):
    
    true_statistic = statistic(X_df, y_df)
    
    assert np.isscalar(true_statistic), "the statistic method should return a scalar value"
    
    bootstrap_statistics = []
    
    for i in range(n_iterations):
        
        X_sample, y_sample = get_bootstrap_sample(X_df, y_df, n_splits) 
        estimate = statistic(X_sample, y_sample)
        bootstrap_statistics.append(estimate)


    ci_boot = np.percentile(bootstrap_statistics, [(alpha*100)/2, 100-((alpha*100)/2)])
    
    return true_statistic, ci_boot, bootstrap_statistics

def bca_bootstrap_ci(X_df, y_df, statistic, bootstrap_statistics, alpha=0.05): #bias corrected and accelerated
    """
    Calculate Bias corrected and accelerated confidence intervals for a bootstrap distribution.
    
    https://www.tau.ac.il/~saharon/Boot/10.1.1.133.8405.pdf
    
    equations: https://www.erikdrysdale.com/bca_python/
    
    Parameters:
        X_df the dataframe with features
        y_df the pandas series with values
        statistic: the function that will return a scalar estimate of the statistic of interest
        bootstrapped_statistics: Array of bootstrap statistics.
        alpha: Significance level (default 0.05 for 95% CI).
    
    Returns:
        Lower and upper bounds of the confidence interval.
    """
    # Sort the bootstrap statistics
    
    true_statistic = statistic(X_df, y_df)
    assert np.isscalar(true_statistic), "the statistic function should return a scalar value"
    
    bootstrap_statistics = np.append(bootstrap_statistics, true_statistic) #this is to prevent prop from becoming 0 and is valid
    bootstrap_statistics = np.sort(bootstrap_statistics)

    # Bias correction factor (z0)
    # https://github.com/scipy/scipy/blob/v1.16.2/scipy/stats/_resampling.py#L73
    B = len(bootstrap_statistics)
    prop = ((bootstrap_statistics < true_statistic).sum() + (bootstrap_statistics <= true_statistic).sum()) / (2*B) 
    prop = np.clip(prop, 0.1, 0.9) #this prevent weird answers for extremely skewed bootstrap distributions and division from error issue, the threshold was calibrated for extreme situations
    z0 = special.ndtri(prop)
    
    # Acceleration constant (a)
    # jackknife estimates (leave one out)
    jackknife_estimates = np.array([statistic(X_df.drop(index=i, inplace=False), y_df.drop(index=i, inplace=False)) 
                                    for i in X_df.index])
    mean_jackknife = np.mean(jackknife_estimates)
    
    num = np.sum((mean_jackknife - jackknife_estimates) ** 3)
    den = (6 * np.sum((mean_jackknife - jackknife_estimates) ** 2) ** 1.5)
    a =  num/den if den!=0 else 0.0 

    # Adjusted percentiles
    # z_alpha_low = stats.norm.ppf(alpha / 2)  # Lower z-score
    # z_alpha_high = stats.norm.ppf(1 - alpha / 2)  # Upper z-score
    
    # lower_percentile = stats.norm.cdf(z0 + ((z0 + z_alpha_low) / (1 - a * (z0 + z_alpha_low))))
    # upper_percentile = stats.norm.cdf(z0 + ((z0 + z_alpha_high) / (1 - a * (z0 + z_alpha_high))))
    
    # ci = np.quantile(bootstrap_statistics, [lower_percentile, upper_percentile])
    z_alpha = special.ndtri(alpha)
    z_1alpha = -z_alpha
    
    num1 = z0 + z_alpha
    alpha_1 = special.ndtr(z0 + num1/(1 - a*num1))
    num2 = z0 + z_1alpha
    alpha_2 = special.ndtr(z0 + num2/(1 - a*num2))
    
    ci = np.quantile(bootstrap_statistics, [alpha_1, alpha_2])

   

    # Map percentiles to bootstrap statistics
    

    return true_statistic, ci, bootstrap_statistics

def permutation_test(X_df, y_df, statistic, n_iterations):
    
    true_statistic = statistic(X_df, y_df)
    perm_statistics = [true_statistic]
    
    for i in range(n_iterations-1):
        perm_y_df = y_df.sample(frac=1).reset_index(drop=True)
        estimate = statistic(X_df, perm_y_df)
        
        perm_statistics.append(estimate)
        
    return np.mean(np.abs(perm_statistics)>=np.abs(true_statistic)), perm_statistics    


def compare_aucs(probs1, probs2, targets):

    z_score, p_value = Delong_test(targets, probs1, probs2)
    
    return p_value
    


def cross_val_metric(piped_model, cv, metric):
    def call_fn(X_df, y_df):
        return np.mean(cross_val_score(piped_model, X_df, y_df, scoring=metric, cv=cv))
    return call_fn



if __name__ == "__main__":
    
    n_boot = 20_000
    n_perm = 1000
    
    cv = StratifiedKFold(n_splits=10, shuffle=False)
    metric = "roc_auc"
    num_feats = 3
    
    featuresGTV_df = pd.read_excel("features_GTVSelected.xlsx", index_col=0)
    featuresTRAD_df = pd.read_excel("features_TRADSelected.xlsx", index_col=0)
    
    models = {
    'LR': LogisticRegression(penalty='none', max_iter=25_000),
    'L-SVM': SVC(probability=True),
    'RF': RandomForestClassifier(random_state=0)
    }
    
    selected_signature = np.load("new_results/cv10_selected_signature.npy", allow_pickle=True).item()
    

    perm_dist = {"gtv":{}, "trad":{}}
    boot_dist = {"gtv":{}, "trad":{}}

    stat_df = {"model":[], "roi":[], "signature":[], "estimate":[], "percentile_95%CI":[], "bca_95%CI":[], "p_value":[]}
    for roi_name, df in {"gtv":featuresGTV_df, "trad":featuresTRAD_df}.items():
        
        print("*"*10)
        print(roi_name)
        print("*"*10)
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name} on {roi_name} dataset...")
            piped_model = make_pipeline(StandardScaler(), model)
            selected_features = selected_signature[roi_name][model_name]
            
            print("Features:")
            print(selected_features)
            
            X_df = df[selected_features]
            y_df = df['PD']
       
            true_estimate, percentile_ci, boot_values = percentile_bootstrap_ci(X_df, y_df, statistic=cross_val_metric(piped_model, cv=cv, metric="roc_auc"), n_splits = cv.n_splits, n_iterations=n_boot)
            print("percentile ci:", true_estimate, percentile_ci, np.mean(boot_values))
            true_estimate, bca_ci,boot_values = bca_bootstrap_ci(X_df, y_df, statistic=cross_val_metric(piped_model, cv=cv, metric=metric), bootstrap_statistics=boot_values)
            print("bca ci:", true_estimate, bca_ci)
            
            p_value, perm_values = permutation_test(X_df, y_df, statistic=cross_val_metric(piped_model, cv=cv, metric=metric), n_iterations=n_perm)
            
            print("p_value:", p_value)
            
            stat_df["model"].append(model_name)
            stat_df["roi"].append(roi_name)
            stat_df["signature"].append(selected_features)
            stat_df["estimate"].append(true_estimate)
            stat_df["percentile_95%CI"].append(percentile_ci)
            stat_df["bca_95%CI"].append(bca_ci)
            stat_df["p_value"].append(p_value)
            
            perm_dist[roi_name][model_name] = perm_values
            boot_dist[roi_name][model_name] = boot_values
            
        
    stat_df = pd.DataFrame(stat_df)
    stat_df.to_csv(os.path.join("new_results", "stat_df20K.csv"), index=False)

    np.save(os.path.join("new_results", "perm_dist20K.npy"), perm_dist, allow_pickle=True)
    np.save(os.path.join("new_results", "boot_dist20K.npy"), boot_dist, allow_pickle=True)
