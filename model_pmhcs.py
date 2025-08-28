import os,pickle
from argparse import ArgumentParser
import pandas as pd
from tfold.modeling import make_inputs

def save_info(df_to_model, working_dir):
    save_path = os.path.join(working_dir, "info.csv")

    # Prepare dataframe to save, keeping pmhc_id as a regular column
    df_to_save = df_to_model[['pmhc_id', 'pep', 'MHC_sequence', 'mhc_a', 'mhc_a_boundary',
                              'mhc_b', 'mhc_b_boundary', 'class', 'seqnn_logkd']].copy()
    
    # Convert the 'pmhc_id' column to type int
    try:
        df_to_save['pmhc_id'] = df_to_save['pmhc_id'].astype(int)
    except ValueError as e:
        print(f"Error converting 'pmhc_id' to integer in new data: {e}")
        return

    df_to_save.loc[:, "mhc_a"] = df_to_save["mhc_a"].apply(lambda x: x.seq() if x is not None else None)
    df_to_save.loc[:, "mhc_b"] = df_to_save["mhc_b"].apply(lambda x: x.seq() if x is not None else None)

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

if __name__=='__main__':               
    parser=ArgumentParser()
    parser.add_argument('input',type=str, 
                         help='Path to input csv file with columns "pep" and "MHC allele" or "MHC sequence", and optionally, "pmhc_id", "pdb_id", and "exclude_pdb". (See details.ipynb for the details.)')    
    parser.add_argument('working_dir',type=str,
                        help='Path to a directory where AlphaFold inputs and outputs will be stored')
    parser.add_argument('--date_cutoff',type=str,default=None,help='Optionally, date cutoff for templates, YYYY-MM-DD.')
    args=parser.parse_args() 
    df_to_model=pd.read_csv(args.input)
    print(f'Need to model {len(df_to_model)} pMHCs.')
    working_dir=args.working_dir
    date_cutoff=args.date_cutoff                        
    #make numbered MHC objects and run seqnn
    df_to_model=make_inputs.preprocess_df(df_to_model)
    # df_to_model at this point contains the following columns: ['pep', 'MHC_sequence', 'pmhc_id', 'mhc_a', 'mhc_a_boundary_left','mhc_a_boundary_right', 'mhc_b', 'mhc_b_boundary_left', 'mhc_b_boundary_right', 'class', 'seqnn_logkds_all', 'seqnn_logkd',  'seqnn_tails'
    #make AF inputs
    save_info(df_to_model, working_dir)
    af_inputs=make_inputs.make_inputs(df_to_model,date_cutoff=date_cutoff,print_stats=False)
    print("\n")
    print('Total number of AlphaFold models to be produced across all targets:',len(af_inputs))

    #make folders
    input_dir=working_dir+'/inputs'
    output_dir=working_dir+'/outputs'
    os.makedirs(working_dir,exist_ok=True)
    os.makedirs(input_dir,exist_ok=True) 
    os.makedirs(output_dir,exist_ok=True)

    #save AF inputs and input dataframe
    with open(input_dir+'/input.pckl','wb') as f: 
        pickle.dump(af_inputs,f) 
    df_to_model.to_pickle(working_dir+'/target_df.pckl')
    