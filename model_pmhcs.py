import os,pickle
from argparse import ArgumentParser
import pandas as pd
from tfold.modeling import make_inputs

def safe_seq(x):
    try:
        return x.seq()
    except AttributeError:
        return str(x) if x is not None else None

def safe_int_conversion(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return value
    
def save_info_all(df_to_model, working_dir):
    save_path = os.path.join(working_dir, "info_all.csv")

    # Prepare dataframe to save, keeping pmhc_id as a regular column
    df_to_save = df_to_model[['pmhc_id', 'pep', 'MHC_sequence', 'mhc_a', 'mhc_a_boundary_left', 'mhc_a_boundary_right',
                              'mhc_b', 'mhc_b_boundary_left', 'mhc_b_boundary_right', 'class', 'seqnn_logkd']].copy()
    
    # Convert the 'pmhc_id' column to type int
    df_to_save['pmhc_id'] = df_to_save['pmhc_id'].apply(safe_int_conversion)
    df_to_save['mhc_a_boundary_left'] = df_to_save['mhc_a_boundary_left'].apply(safe_int_conversion)
    df_to_save['mhc_a_boundary_right'] = df_to_save['mhc_a_boundary_right'].apply(safe_int_conversion)
    df_to_save['mhc_b_boundary_left'] = df_to_save['mhc_b_boundary_left'].apply(safe_int_conversion)
    df_to_save['mhc_b_boundary_right'] = df_to_save['mhc_b_boundary_right'].apply(safe_int_conversion)

    df_to_save.loc[:, "mhc_a"] = df_to_save["mhc_a"].apply(safe_seq)
    df_to_save.loc[:, "mhc_b"] = df_to_save["mhc_b"].apply(safe_seq)

    # If file exists, merge instead of overwriting
    if os.path.exists(save_path):
        # Read the existing file without setting an index
        existing = pd.read_csv(save_path)
        combined = pd.concat([existing, df_to_save], ignore_index=True)
        
        # Sort by 'pmhc_id'
        combined = combined.sort_values(by='pmhc_id')

        # Save, resetting the index to create a new unique index column
        combined.to_csv(save_path, index=True, index_label="idx")
    else:
        # Sort by 'pmhc_id'
        df_to_save = df_to_save.sort_values(by='pmhc_id')
        
        # Save with a new index column
        df_to_save.to_csv(save_path, index=True, index_label="idx")

def save_info_processed(af_inputs, working_dir, filename="inputs_summary.csv"):
    """
    Save rows with columns:
      pmhc_id (= target_id), model_id (= current_id), pep, mhc_a, mhc_b
    Works whether each item is:
      - {"sequences": [...], "target_id":..., "current_id":...}   (flat)
      - {"inputs": {"sequences": [...]}, "target_id":..., "current_id":...} (nested)
    """
    os.makedirs(working_dir, exist_ok=True)
    save_path = os.path.join(working_dir, filename)
    log_path  = os.path.join(working_dir, "inputs_summary_failures.log")

    def _extract_sequences(seq):
        """Return (pep, mhc_a, mhc_b) from list, tolerate missing mhc_b."""
        pep   = seq[0] if len(seq) > 0 else None
        mhc_a = seq[1] if len(seq) > 1 else None
        mhc_b = seq[2] if len(seq) > 2 else None
        return pep, mhc_a, mhc_b

    rows = []
    for x in af_inputs:
        try:
            # support flat or nested under 'inputs'
            d = x.get("inputs", x)

            # IDs can be at top level or inside d as well; prefer d if present
            pmhc_id  = d.get("target_id", x.get("target_id"))
            model_id = d.get("current_id", x.get("current_id"))

            seqs = d.get("sequences")
            if seqs is None:
                raise KeyError("missing 'sequences'")

            pep, mhc_a, mhc_b = _extract_sequences(seqs)

            rows.append({
                "pmhc_id": pmhc_id,
                "model_id": model_id,
                "pep": "" if pep is None else str(pep),
                "mhc_a": "" if mhc_a is None else str(mhc_a),
                "mhc_b": "" if mhc_b is None else str(mhc_b),
            })
        except Exception as e:
            continue

    if not rows:
        # nothing to write, but create an empty file with headers for consistency
        pd.DataFrame(columns=["pmhc_id","model_id","pep","mhc_a","mhc_b"]).to_csv(
            save_path, index=True, index_label="idx"
        )
        return

    df = pd.DataFrame(rows)

    # cast IDs to nullable integers, then sort
    df["pmhc_id"]  = pd.to_numeric(df["pmhc_id"], errors="coerce").astype("Int64")
    df["model_id"] = pd.to_numeric(df["model_id"], errors="coerce").astype("Int64")
    df = df.sort_values(["pmhc_id", "model_id"])

    # If file exists, merge and de-dup on (pmhc_id, model_id)
    if os.path.exists(save_path):
        existing = pd.read_csv(save_path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["pmhc_id", "model_id"], keep="last")
        combined = combined.sort_values(["pmhc_id", "model_id"])
        combined.to_csv(save_path, index=True, index_label="idx")
    else:
        df.to_csv(save_path, index=True, index_label="idx")

if __name__=='__main__':               
    parser=ArgumentParser()
    parser.add_argument('input',type=str, 
                         help='Path to input csv file with columns "pep" and "MHC allele" or "MHC sequence", and optionally, "pmhc_id", "pdb_id", and "exclude_pdb". (See details.ipynb for the details.)')    
    parser.add_argument('working_dir',type=str,
                        help='Path to a directory where AlphaFold inputs and outputs will be stored')
    parser.add_argument('--date_cutoff',type=str,default=None,help='Optionally, date cutoff for templates, YYYY-MM-DD.')
    args=parser.parse_args()
    
    date_cutoff=args.date_cutoff

    #make folders
    working_dir=args.working_dir
    input_dir=working_dir+'/inputs'
    output_dir=working_dir+'/outputs'
    os.makedirs(working_dir,exist_ok=True)
    os.makedirs(input_dir,exist_ok=True) 
    os.makedirs(output_dir,exist_ok=True)

    df_to_model=pd.read_csv(args.input)

    required_cols = {"pmhc_id", "pep", "MHC_sequence"}
    if not (required_cols & set(df_to_model.columns)):
        raise ValueError(f"Input file must contain columns {required_cols}, got {list(df_to_model.columns)}")

    print(f'Need to model {len(df_to_model)} pMHCs.')
                   
    #make numbered MHC objects and run seqnn
    df_to_model = make_inputs.preprocess_df(df_to_model,working_dir)
    # df_to_model at this point contains the following columns: ['pep', 'MHC_sequence', 'pmhc_id', 'mhc_a', 'mhc_a_boundary_left','mhc_a_boundary_right', 'mhc_b', 'mhc_b_boundary_left', 'mhc_b_boundary_right', 'class', 'seqnn_logkds_all', 'seqnn_logkd',  'seqnn_tails'
    #make AF inputs
    save_info_all(df_to_model, working_dir)
    af_inputs=make_inputs.make_inputs(df_to_model,working_dir,date_cutoff=date_cutoff,print_stats=False)
    save_info_processed(af_inputs, working_dir)  # in case some rows were filtered out during make_inputs
    print("\n")
    print('Total number of AlphaFold models to be produced across all targets:',len(af_inputs))

    #save AF inputs and input dataframe
    try:
        with open(input_dir+'/input.pckl','wb') as f: 
            pickle.dump(af_inputs,f) 
    except Exception as e:
        print(f"Error saving AlphaFold inputs (contains unpickable objects?): {e}")
    df_to_model.to_pickle(working_dir+'/target_df.pckl')
    