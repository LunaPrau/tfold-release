#patch by: Victor Mikhaylov, vmikhayl@ias.edu
#Institute for Advanced Study, 2021-2023

# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from tfold_patch.tfold_config import data_dir,mmcif_dir,kalign_binary_path,af_params,alphafold_dir
sys.path.append(alphafold_dir) #path to AlphaFold for import

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

"""Full AlphaFold protein structure prediction script."""
import json
import os
import pathlib
import pickle
import random
import time
from typing import Dict, Union, Optional
import pickle
import pandas as pd

from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants

from alphafold.data import templates
from alphafold.model import config
from alphafold.model import model
import numpy as np

from alphafold.model import data

import jax.numpy as jnp

from collections import Counter
from pprint import pformat

# Internal import (7716).

logging.set_verbosity(logging.WARNING)

import tfold_patch.tfold_pipeline as pipeline
import tfold_patch.postprocessing as postprocessing

flags.DEFINE_string('inputs',None,'path to a .pkl input file with a list of inputs')
flags.DEFINE_string('output_dir',None,'where to put outputs')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS=20            #default 20; later reduced to 4 anyway (?)
MAX_TEMPLATE_DATE='9999-12-31'  #set no limit here

def save_confidences(working_dir, target_id, current_id, rankning_confidences):
    save_path = os.path.join(working_dir, "confidences.csv")

    # Turn timings into a row dict
    row = {"target_id": target_id, "model_id": current_id, **rankning_confidences}

    # If file exists, append row
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        # First time → create new dataframe
        df = pd.DataFrame([row])

    df.to_csv(save_path, index=False)

def save_timings(working_dir, target_id, current_id, timings):
    save_path = os.path.join(working_dir, "timings.csv")

    # Turn timings into a row dict
    row = {"target_id": target_id, "model_id": current_id, **timings}

    # If file exists, append row
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        # First time → create new dataframe
        df = pd.DataFrame([row])

    df.to_csv(save_path, index=False)

def renumber_pdb(pdb,renumber_list):   
    '''
    note: AF numbers residues from 1 sequentially but with interchain shifts    
    '''
    lines=[]      
    i_current=-1
    chain_pdbnum_prev='xxxxxx'
    for line in pdb.split('\n'):
        if line.startswith(('ATOM','TER')):   
            chain_pdbnum=line[21:27]
            if chain_pdbnum!=chain_pdbnum_prev:
                chain_pdbnum_prev=chain_pdbnum
                i_current+=1
            new_chain_pdbnum=renumber_list[i_current]
            line=line[:21]+new_chain_pdbnum+line[27:]
        lines.append(line)    
    return '\n'.join(lines)

def predict_structure(sequences,msas,template_hits,renumber_list,
                      target_id,current_id,parent_output_dir,target_output_dir,
                      data_pipeline,model_runners,benchmark,random_seed,true_pdb=None):  
    
    timings = {}
    ranking_confidences = {}

    os.makedirs(target_output_dir,exist_ok=True)    
    # Get features.
    t_0=time.time()    
    feature_dict=data_pipeline.process(sequences,msas,template_hits)
    t_1=time.time()-t_0
    timings['tfold_features']=t_1 
    features_output_path = os.path.join(target_output_dir, 'features.pkl')
    with open(features_output_path, 'wb') as f:
        pickle.dump(feature_dict, f, protocol=4)
    # Run the models.    
    num_models=len(model_runners)
    for model_index,(model_name,model_runner) in enumerate(model_runners.items()):
        print(f'Predicting for target id {target_id}, model id {current_id}')
        t_3=time.time()
        model_random_seed=model_index+random_seed*num_models
        processed_feature_dict=model_runner.process_features(feature_dict,random_seed=model_random_seed)
        t_4=time.time()-t_3
        timings[f'alphafold_features']=t_4
        outfile = os.path.join(target_output_dir, current_id)
        os.makedirs(outfile, exist_ok=True)
        t_5=time.time()
        prediction_result=model_runner.predict(processed_feature_dict,random_seed=model_random_seed, outfile=outfile)
        t_6=time.time()-t_5
        timings[f'gen_embeddings']=t_6

        # Add the predicted LDDT in the b-factor column.
        # Note that higher predicted LDDT value means higher model confidence.
        plddt=prediction_result['plddt']
        ranking_confidences[current_id] = prediction_result['ranking_confidence']
        plddt_b_factors=np.repeat(plddt[:, None],residue_constants.atom_type_num,axis=-1)
        unrelaxed_protein=protein.from_prediction(features=processed_feature_dict,result=prediction_result,
                                                  b_factors=plddt_b_factors,remove_leading_feature_dimension=True)
        unrelaxed_pdb=protein.to_pdb(unrelaxed_protein)
        t_7=time.time()-t_6
        timings[f'gen_pdb']=t_7
        unrelaxed_pdb_renumbered=renumber_pdb(unrelaxed_pdb,renumber_list)
        #renumber peptide
        unrelaxed_pdb_renumbered,pep_pdbnum,pep_tails,success=postprocessing.renumber_pep(unrelaxed_pdb_renumbered)
        prediction_result['pep_renumbered']=success
        prediction_result['pep_tails']=pep_tails
        prediction_result['pdbnum_list']=['P'+p for p in pep_pdbnum]+renumber_list[len(sequences[0]):]
        #compute rmsd if true structure provided
        if true_pdb:
            rmsds=postprocessing.compute_rmsds(unrelaxed_pdb_renumbered,true_pdb)
            prediction_result={**prediction_result,**rmsds}
        #save results and pdb
        result_output_path=os.path.join(target_output_dir,f'result_{current_id}.pkl')
        with open(result_output_path,'wb') as f:
            pickle.dump(prediction_result, f, protocol=4)
        unrelaxed_pdb_path=os.path.join(target_output_dir,f'structure_{current_id}.pdb')
        with open(unrelaxed_pdb_path,'w') as f:
            f.write(unrelaxed_pdb_renumbered)
        t_8=time.time()-t_7
        timings[f'process_pdb']=t_8
    save_confidences(target_output_dir, target_id, current_id, ranking_confidences)
    print(f"Final timings for target id {target_id}, model id {current_id}:\n{pformat(timings, indent=4)}")
    save_timings(target_output_dir, target_id, current_id, timings)

def main(argv):
    try:
        t_start=time.time()    
        with open(FLAGS.inputs,'rb') as f:
            inputs=pickle.load(f)            #list of dicts [{param_name : value_for_input_0},..]     
        if len(inputs)==0:
            raise ValueError('input list of zero length provided')
        output_dir=FLAGS.output_dir
        parent_output_dir=os.path.dirname(output_dir)
        logging.info(f'processing {len(inputs)} inputs...')
        #set parameters#   
        params=af_params #from tfold.config
        num_ensemble      =params['num_ensemble']   
        model_names       =params['model_names']   
        chain_break_shift =params['chain_break_shift']
        ##################        
        template_featurizer=templates.HhsearchHitFeaturizer(mmcif_dir=mmcif_dir,
                                                            max_template_date=MAX_TEMPLATE_DATE,
                                                            max_hits=MAX_TEMPLATE_HITS,
                                                            kalign_binary_path=kalign_binary_path,
                                                            release_dates_path=None,
                                                            obsolete_pdbs_path=None)
        data_pipeline=pipeline.DataPipeline(template_featurizer=template_featurizer,chain_break_shift=chain_break_shift)
        model_runners={}    
        for model_name in model_names:
            model_config=config.model_config(model_name)
            model_config.data.eval.num_ensemble=num_ensemble
            model_params=data.get_model_haiku_params(model_name=model_name,data_dir=data_dir)
            model_runner=model.RunModel(model_config,model_params)
            model_runners[model_name]=model_runner
        logging.info('Have %d models: %s',len(model_runners),list(model_runners.keys()))
        random_seed=FLAGS.random_seed
        if random_seed is None:
            random_seed = random.randrange(sys.maxsize // len(model_names))
        logging.info('Using random seed %d for the data pipeline',random_seed)
        print("\n")
        target_id = inputs[0]["target_id"]
        print(f'Target id: {target_id}')
        print(f'Number of models to generate for this target: {len(inputs)}')
        for x in inputs:
            sequences        =x['sequences']            #(seq_chain1,seq_chain2,..)
            msas             =x['msas']                 #list of dicts {chain_number:path to msa in a3m format,..}
            template_hits    =x['template_hits']        #list of dicts for template hits
            renumber_list    =x['renumber_list']        #e.g. ['P   1 ','P   2 ',..,'M   5 ',..]
            target_id        =str(x['target_id'])       #id or name of the target
            current_id       =str(x['current_id'])      #id of the run (for a given target, all run ids should be distinct)
            true_pdb         =x.get('true_pdb')         #pdb_id of true structure, for rmsd computation
            target_output_dir=output_dir+'/'+target_id
            predict_structure(sequences=sequences,msas=msas,template_hits=template_hits,renumber_list=renumber_list,target_id=target_id,
                            current_id=current_id,parent_output_dir=parent_output_dir,target_output_dir=target_output_dir,
                            data_pipeline=data_pipeline,model_runners=model_runners,
                            benchmark=FLAGS.benchmark,random_seed=random_seed,true_pdb=true_pdb)
        t_delta=(time.time()-t_start)/60
        print(f'Processed {len(inputs)} inputs in {t_delta} minutes.')
    except AssertionError as e:
        logging.error(f'Error processing target {target_id}, input {x}: {e}')
        log_path = os.path.join(parent_output_dir, "assertion_failures.log")
        with open(log_path, "a") as logf:
                logf.write(f"[target={target_id}, run={current_id}] failed: {e}\n")

        failed_path = os.path.join(output_dir, "failed.txt")
        with open(failed_path, "w") as f:
            f.write(str(e) + "\n")

if __name__ == '__main__':
    flags.mark_flags_as_required(['inputs','output_dir'])
    app.run(main)
