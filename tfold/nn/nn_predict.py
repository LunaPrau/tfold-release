import os
import pickle
import numpy as np
import json

from tfold.nn import pipeline as tfold_pipeline
from tfold.nn import models as tfold_models
from tfold.nn import nn_utils
from tfold.config import seqnn_obj_dir

import pandas as pd

def _create_kd_arrays(cl,l):
    if cl=='I':
        tails=nn_utils.generate_registers_I(l)
    else:
        tails=nn_utils.generate_registers_II(l)    
    return {'tails':tails,'logkds':[]}
    
def predict(df,working_dir,cl,mhc_as_obj=False,model_list=None,params_dir=None,weights_dir=None,keep_all_predictions=False):
    df=df.copy()
    passed_rows=[]
    #prepare data
    for _, row in df.iterrows():
        pmhc_id=row["pmhc_id"]
        try:
            if cl == 'I':
                row['tails_all'] = nn_utils.generate_registers_I(len(row['pep']))
            else:
                row['tails_all'] = nn_utils.generate_registers_II(len(row['pep']))

            row['logkd_all'] = []

            inputs = (tfold_pipeline.pipeline_i if cl == 'I' else tfold_pipeline.pipeline_ii)(
                pd.DataFrame([row]), mhc_as_obj=mhc_as_obj
            )

            passed_rows.append(row)

        except Exception as e:
            print(f"tfold/nn/nn_predict.py [predict] failed row pmhc_id={pmhc_id}: {e}")
            if working_dir:
                outdir = os.path.join(working_dir, "outputs", str(pmhc_id))
                os.makedirs(outdir, exist_ok=True)
                with open(os.path.join(outdir, "failed.txt"), "w") as f:
                    f.write(str(e))
            continue

    if passed_rows:
        df = pd.DataFrame(passed_rows)
    else:
        raise AssertionError("No rows passed the preprocessing step; cannot run prediction.")
    
    # prepare params and such
    params_dir = params_dir or (seqnn_obj_dir + '/params')
    weights_dir = weights_dir or (seqnn_obj_dir + '/weights')

    if not model_list:
        with open(seqnn_obj_dir + f'/model_list_{cl}.pckl','rb') as f:
            model_list = pickle.load(f)

    n_k = len(model_list[0])  # could be 2 or 4 depending on tuple length
    params_all = {}

    for filename in os.listdir(params_dir):
        if not filename.endswith(".json"):
            continue  # skip non-json
        try:
            run_n = int(filename.split('.')[0].split('_')[1])
        except Exception:
            continue  # skip bad filename formats

        with open(os.path.join(params_dir, filename)) as f:
            d = json.load(f)

        for x in d:
            # build key, but be tolerant if some fields missing
            try:
                k = (run_n, x['model_n'], x['split_n'], x['copy_n'])
            except KeyError as e:
                with open("model_parse_failures.log", "a") as logf:
                    logf.write(f"Skipping entry in {filename}: missing field {e}\n")
                continue

            if k[:n_k] in model_list:
                params_all[k] = x
            else:
                with open("model_parse_failures.log", "a") as logf:
                    logf.write(f"Skipping unmatched model {k} (not in model_list)\n")
    #do inference    
    #use logkd, not kd in names!    
    model_list_full=list(params_all.keys())
    print(f'Making Kd predictions for {len(df)} pMHCs.')
    for k in model_list_full:
        params=params_all[k]
        model_func=getattr(tfold_models,params['model'])        
        model=model_func(params)
        weight_path=weights_dir+f'/run_{cl}_'+'_'.join([f'{kk}' for kk in k])
        model.load_weights(weight_path)
        try:
            outputs=model(inputs).numpy()
        except:
            print(f"tfold/nn/nn_predict.py [predict] model inference failed for model {k}")
            continue
        for x,y,z in zip(df['logkd_all'],df['tails_all'],outputs):
            x.append(z[:len(y)])
    # If x is empty (no predictions), np.average([]) raises ValueError
    df['logkd_all'] = df['logkd_all'].map(np.array)
    x=df['logkd_all'].map(lambda arr: np.nan if arr is None or len(arr) == 0 else np.average(arr, axis=0))
    df['seqnn_logkds_all'] = [
        np.array([tuple(c) for c in zip(b, a)],
                 dtype=[('tail', object), ('logkd', float)])
        if a is not None and not (isinstance(a, float) and np.isnan(a)) else np.array([])
        for a, b in zip(x, df['tails_all'])
    ]
    df['seqnn_logkd'] = x.map(lambda val: np.nan if val is None or (hasattr(val, "__len__") and len(val) == 0) else np.min(val))
    df['seqnn_tails'] = x.map(np.argmin)
    df['seqnn_tails'] = df[['seqnn_tails', 'tails_all']].apply(
        lambda x: x['tails_all'][x['seqnn_tails']]
        if x['seqnn_tails'] is not None and x['seqnn_tails'] < len(x['tails_all'])
        else None,
        axis=1
    )
    if not keep_all_predictions:
        df=df.drop(['logkd_all','tails_all'],axis=1)
        return df
    else:
        return df,model_list_full