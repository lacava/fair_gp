# train GerryFair model
import copy
from utils import setup_data,evaluate_model,get_hypervolumes
from clean import clean_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import json


def train_gerryfair_model(est, model_name, dataset, attributes, seed=42,
        rdir='results'):
    X_train, X_test, X_prime_train, X_prime_test, y_train, y_test, sens_cols = \
    setup_data(dataset, attributes, seed)
    print('model:',model_name)
#     X_combined_train = np.hstack((X_train, X_prime_train))
#     X_combined_test = np.hstack((X_test, X_prime_test))

#     gamma_list = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    gamma_list = np.linspace(0.001,0.999,100)
    
    performance = []

    dataset_name = dataset.split('/')[-1].split('.')[0]
   
    # loop thru values of gamma
    t0 = time.process_time()
    for g in gamma_list:
#         est.gamma = g
        model = copy.deepcopy(est)
        model.set_options(gamma=g)
        print('gamma:',model.gamma)
        #train
        error, fairness_violation = model.train(X_train, X_prime_train, 
                                                y_train.values)
        # get predictions
        train_predictions = model.predict(X_train, sample=True)
        test_predictions = model.predict(X_test, sample=True)
        train_probabilities = model.predict(X_train, sample=False)
        test_probabilities = model.predict(X_test, sample=False)
        train_perf = evaluate_model(X_train, X_prime_train, y_train, 
                                    train_predictions, train_probabilities)
        train_perf.update({'self_error':error[-1], 
            'self_fairness_violation':fairness_violation[-1]})
        test_perf = evaluate_model(X_test, X_prime_test, y_test, 
                test_predictions, test_probabilities)
        
#         output_train = {**header, 
#                         'model':model_name + str(g),
#                         'train':True,
#                         **train_perf
#                        }
#         output_test = {**header, 
#                         'model':model_name + str(g),
#                         'train':False,
#                         **test_perf
#                        }
        performance.append({
            'method':model_name,
            'dataset':dataset_name,
            'seed':seed,
            'model':model_name +':g=' + str(g),
            'train':train_perf,
            'test':test_perf
        })
        
            
    runtime = time.process_time() - t0
    header = {
            'method':model_name,
            'dataset':dataset_name,
            'seed':seed,
            'time':runtime
    }
    # get hypervolume of pareto front
    hv = get_hypervolumes(performance,dataset_name)
    hv = [{**header, **i} for i in hv]
    df_hv = pd.DataFrame.from_records(hv)
    df_hv.to_csv(
            rdir+'/hv_'+model_name+'_'+str(seed)+'_'+dataset.split('/')[-1],
            index=False)
    
    with open(rdir +'/perf_'+model_name + '_' +dataset_name+'_'
            +str(seed)+'.json', 'w') as fp:
        json.dump(performance, fp, sort_keys=True, indent=4)
    return performance, df_hv

# Train FEAT model
def train_feat_model(est, model_name, dataset, attributes, seed=42, 
        rdir='results'):
    X_train, X_test, X_prime_train, X_prime_test, \
            y_train, y_test, sens_cols = setup_data(dataset, attributes, seed)
    print('model:',model_name)
#     X_combined_train = np.hstack((X_train, X_prime_train))
#     X_combined_test = np.hstack((X_test, X_prime_test))
    protected_groups = [1 if c in sens_cols else 0 for c in X_train.columns]
#     protected_groups = ([False for f in np.arange(X_train.shape[1])] + 
#                        [True for f in np.arange(X_prime_train.shape[1])])
    est.protected_groups = ','.join(                                       
            [str(int(pg)) for pg in protected_groups]).encode() 

    # print('fair_feat.py protected groups:',protected_groups)
    est.feature_names = ','.join(list(X_train.columns)).encode()
    # print('feature names:', feature_names)
    print('est protected_groups: ', est.protected_groups)

    dataset_name = dataset.split('/')[-1].split('.')[0]
    
    t0 = time.process_time()
    
    est.fit(X_train, y_train)
    
#     pdb.set_trace()
    # get predictions
    #TODO: add predict_proba_archive !!
    print('archive size:',est.get_archive_size())
    train_predictions = est.predict_archive(X_train.values)
    test_predictions = est.predict_archive(X_test.values)

    print('getting probabilities')
    train_probs, test_probs = [],[]
    for i in np.arange(est.get_archive_size()):
        train_probs.append(
                np.nan_to_num(
                    est.predict_proba_archive(i,X_train.values).flatten()
                    )
                )
        test_probs.append(
                np.nan_to_num(
                    est.predict_proba_archive(i,X_test.values).flatten()
                    )
                )
#     pdb.set_trace()
    print('getting performance')
    performance = []
    i = 0
    for train_pred, test_pred, train_prob, test_prob \
         in zip(train_predictions,test_predictions, train_probs, test_probs):
        performance.append({
            'method':model_name,
            'model':model_name+':archive('+str(i)+')',
            'dataset':dataset_name,
            'seed':seed,
            'train':evaluate_model(X_train, X_prime_train, y_train, train_pred, 
                train_prob),
            'test':evaluate_model(X_test, X_prime_test, y_test, test_pred, 
                test_prob)
        })
        i = i + 1
    
    # get hypervolume of pareto front
    runtime = time.process_time() - t0
    header = {
            'method':model_name,
            'dataset':dataset_name,
            'seed':seed,
            'time':runtime
    }
    hv = get_hypervolumes(performance,dataset_name)
    hv = [{**header, **i} for i in hv]
    df_hv = pd.DataFrame.from_records(hv)
    df_hv.to_csv(
            rdir+'/hv_'+model_name+'_'+str(seed)+'_'+dataset.split('/')[-1],
            index=False)
    
    with open(rdir +'/perf_'+model_name + '_' +dataset_name+'_'
            +str(seed)+'.json', 'w') as fp:
        json.dump(performance, fp, sort_keys=True, indent=4)
    return performance, hv
