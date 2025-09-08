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
import gc

# Internal import (7716).

logging.set_verbosity(logging.WARNING)

import tfold_patch.tfold_pipeline as pipeline
import tfold_patch.postprocessing as postprocessing

import average_embeddings

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
flags.DEFINE_integer('skip', 0, 'Number of input entries to skip from the start')
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

    # Round float timings to 3 decimal places
    timings_rounded = {
        k: (round(v, 3) if isinstance(v, (int, float)) else v)
        for k, v in timings.items()
    }

    row = {"target_id": target_id, "model_id": current_id, **timings_rounded}

    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(save_path, index=False)

def renumber_pdb(pdb, renumber_list):
    lines = []
    i_current = -1
    chain_pdbnum_prev = 'xxxxxx'
    for line in pdb.split('\n'):
        if line.startswith(('ATOM','TER')):
            chain_pdbnum = line[21:27]
            if chain_pdbnum != chain_pdbnum_prev:
                chain_pdbnum_prev = chain_pdbnum
                i_current += 1
            if i_current < len(renumber_list):
                new_chain_pdbnum = renumber_list[i_current]
                line = line[:21] + new_chain_pdbnum + line[27:]
        lines.append(line)
    return '\n'.join(lines)

def predict_structure(sequences, msas, template_hits, renumber_list,
                      target_id, current_id, parent_output_dir, target_output_dir,
                      data_pipeline, model_runners, benchmark, random_seed, true_pdb=None):

    timings = {}
    ranking_confidences = {}

    os.makedirs(target_output_dir, exist_ok=True)

    # ===== Features =====
    t0 = time.time()
    feature_dict = data_pipeline.process(sequences, msas, template_hits)
    timings['tfold_features'] = time.time() - t0

    # Save features once
    features_output_path = os.path.join(target_output_dir, f'features_{current_id}.pkl')
    with open(features_output_path, 'wb') as f:
        pickle.dump(feature_dict, f, protocol=4)

    num_models = len(model_runners)
    for model_index, (model_name, model_runner) in enumerate(model_runners.items()):
        print("\n")
        print(f'Predicting target={target_id}, run={current_id}, model={model_name}')
        model_random_seed = model_index + random_seed * num_models

        model_outdir = os.path.join(target_output_dir, f"{current_id}", model_name)
        os.makedirs(model_outdir, exist_ok=True)

        try:
            # Alphafold feature processing
            t = time.time()
            processed_feature_dict = model_runner.process_features(
                feature_dict, random_seed=model_random_seed
            )
            timings[f'alphafold_features.{model_name}'] = time.time() - t
            
            # OOM-saving edits: free feature_dict once processed_feature_dict is created
            del feature_dict
            gc.collect()

            # Inference
            t = time.time()
            prediction_result = model_runner.predict(
                processed_feature_dict, random_seed=model_random_seed, outfile=model_outdir
            )
            timings[f'gen_embeddings.{model_name}'] = time.time() - t

            # Build unrelaxed PDB
            plddt = prediction_result['plddt']
            ranking_confidences[model_name] = prediction_result['ranking_confidence']
            plddt_b_factors = np.repeat(plddt[:, None], residue_constants.atom_type_num, axis=-1)

            t = time.time()
            unrelaxed_protein = protein.from_prediction(
                features=processed_feature_dict,
                result=prediction_result,
                b_factors=plddt_b_factors,
                remove_leading_feature_dimension=True
            )
            unrelaxed_pdb = protein.to_pdb(unrelaxed_protein)
            timings[f'gen_pdb.{model_name}'] = time.time() - t

            # Renumber AF PDB
            t = time.time()
            try:
                unrelaxed_pdb_renumbered = renumber_pdb(unrelaxed_pdb, renumber_list)
            except Exception as e:
                # try to carry on without renumbering
                unrelaxed_pdb_renumbered = unrelaxed_pdb
                with open(os.path.join(parent_output_dir, "renumber_failures.log"), "a") as logf:
                    logf.write(f"[target={target_id}, run={current_id}, model={model_name}] renumber_pdb failed: {repr(e)}\n")

            # peptide renumber
            try:
                unrelaxed_pdb_renumbered, pep_pdbnum, pep_tails, success = postprocessing.renumber_pep(unrelaxed_pdb_renumbered)
            except Exception as e:
                pep_pdbnum, pep_tails, success = [], [], False
                with open(os.path.join(parent_output_dir, "pep_renumber_failures.log"), "a") as logf:
                    logf.write(f"[target={target_id}, run={current_id}, model={model_name}] renumber_pep failed: {repr(e)}\n")

            prediction_result['pep_renumbered'] = success
            prediction_result['pep_tails'] = pep_tails

            # Safe pdbnum_list build
            try:
                left = ['P' + p for p in pep_pdbnum] if pep_pdbnum else []
                n0 = len(sequences[0]) if sequences and sequences[0] is not None else 0
                right = renumber_list[n0:] if renumber_list and len(renumber_list) >= n0 else []
                prediction_result['pdbnum_list'] = left + right
            except Exception as e:
                prediction_result['pdbnum_list'] = []
                with open(os.path.join(parent_output_dir, "pdbnum_list_failures.log"), "a") as logf:
                    logf.write(f"[target={target_id}, run={current_id}, model={model_name}] pdbnum_list failed: {repr(e)}\n")

            # RMSDs if true structure provided
            if true_pdb:
                try:
                    rmsds = postprocessing.compute_rmsds(unrelaxed_pdb_renumbered, true_pdb)
                    prediction_result.update(rmsds)
                except Exception as e:
                    with open(os.path.join(parent_output_dir, "rmsd_failures.log"), "a") as logf:
                        logf.write(f"[target={target_id}, run={current_id}, model={model_name}] compute_rmsds failed: {repr(e)}\n")

           # Save results/PDBs
            result_output_path = os.path.join(model_outdir, f'result_{current_id}_{model_name}.pkl')
            with open(result_output_path, 'wb') as f:
                pickle.dump(prediction_result, f, protocol=4)

            unrelaxed_pdb_path = os.path.join(model_outdir, f'structure_{current_id}_{model_name}.pdb')
            with open(unrelaxed_pdb_path, 'w') as f:
                f.write(unrelaxed_pdb)

            timings[f'process_pdb.{model_name}'] = time.time() - t

            # OOM-saving edits: free large intermediates after saving
            del processed_feature_dict, prediction_result, unrelaxed_protein, unrelaxed_pdb
            gc.collect()

        except Exception as e:
            with open(os.path.join(parent_output_dir, "per_model_failures.log"), "a") as logf:
                logf.write(f"[target={target_id}, run={current_id}, model={model_name}] failed: {repr(e)}\n")
            continue

    # Save once at the end
    save_confidences(target_output_dir, target_id, current_id, ranking_confidences)

    # Round timings to 3 decimals before saving
    timings = {k: (round(v, 3) if isinstance(v, (int, float)) else v) for k, v in timings.items()}
    print(f"Final timings for target id {target_id}, run id {current_id}:\n{pformat(timings, indent=4)}")
    save_timings(target_output_dir, target_id, current_id, timings)

def main(argv):
    try:
        t_start=time.time()
        with open(FLAGS.inputs,'rb') as f:
            inputs=pickle.load(f) #list of dicts [{param_name : value_for_input_0},..]
        if len(inputs) == 0:
            raise ValueError('input list of zero length provided')
        
        if FLAGS.skip:
            skip=FLAGS.skip+1
            if skip > 0:
                logging.info(f"Skipping first {skip} inputs...")
                inputs = inputs[skip:]

        # OOM-saving edits: split input into smaller chunks if too many        
        # group inputs by target_id
        input_by_target = {}
        for x in inputs:
            target_id = str(x['target_id'])
            if target_id not in input_by_target:
                input_by_target[target_id] = []
            input_by_target[target_id].append(x)
        
        # turn inputs into chunks of targets
        input_target_chunks = list(input_by_target.values())
        
        output_dir = FLAGS.output_dir
        parent_output_dir = os.path.dirname(output_dir)
        logging.info(f'processing {len(inputs)} inputs grouped into {len(input_target_chunks)} target chunks...')

        # set parameters
        params = af_params
        num_ensemble = params['num_ensemble']
        model_names = params['model_names']
        chain_break_shift = params['chain_break_shift']

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
        
        failed_any = False

        current_target_id = None
        for target_chunk in input_target_chunks:
            # Process one target at a time
            first_entry = target_chunk[0]
            current_target_id = str(first_entry['target_id'])
            print(f'Processing target id: {current_target_id} with {len(target_chunk)} models to be generated.')

            for x in target_chunk:
                try:
                    sequences        = x['sequences']
                    msas             = x['msas']
                    template_hits    = x['template_hits']
                    renumber_list    = x['renumber_list']
                    target_id        = str(x['target_id'])
                    current_id       = str(x['current_id'])
                    true_pdb         = x.get('true_pdb')
                    target_output_dir = os.path.join(output_dir, target_id)

                    predict_structure(
                        sequences=sequences, msas=msas, template_hits=template_hits, renumber_list=renumber_list,
                        target_id=target_id, current_id=current_id, parent_output_dir=parent_output_dir,
                        target_output_dir=target_output_dir, data_pipeline=data_pipeline,
                        model_runners=model_runners, benchmark=FLAGS.benchmark,
                        random_seed=random_seed, true_pdb=true_pdb
                    )
                    # OOM-saving edits: free memory after each input is processed
                    del sequences, msas, template_hits, renumber_list, target_id, current_id, true_pdb
                    gc.collect()
                except Exception as e:
                    failed_any = True
                    # robust logging even if paths missing
                    safe_parent = target_output_dir if 'target_output_dir' in locals() else parent_output_dir if 'parent_output_dir' in locals() else os.path.dirname(output_dir)
                    os.makedirs(safe_parent, exist_ok=True)
                    log_path = os.path.join(safe_parent, "alphafold_failures.log")
                    with open(log_path, "a") as logf:
                        run = x.get('current_id', '?')
                        tgt = x.get('target_id', '?')
                        logf.write(f"[target={tgt}, run={run}] failed: {repr(e)}\n")
                    # also mark this run as failed
                    os.makedirs(output_dir, exist_ok=True)
                    failed_path = os.path.join(output_dir, "failed.txt")
                    with open(failed_path, "a") as f:
                        run = x.get('current_id', '?')
                        tgt = x.get('target_id', '?')
                        f.write(f"{tgt}\t{run}\t{repr(e)}\n")
                    continue  # move on to next run
            
            t_delta = (time.time() - t_start) / 60
            print(f'Processed {len(inputs)} inputs in {t_delta:.2f} minutes.')

            average_embeddings.main(output_dir, current_target_id)
            print(f"Averaged embeddings for {current_target_id}.")

    except Exception as e:
        # Last-chance logger without assuming any locals exist
        base = os.path.dirname(FLAGS.output_dir) if FLAGS.output_dir else "."
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, "alphafold_top_level_failures.log"), "a") as logf:
            logf.write(f"Top-level failure: {repr(e)}\n")

            
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
