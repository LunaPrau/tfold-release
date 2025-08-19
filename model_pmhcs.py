import os,pickle
from argparse import ArgumentParser
import pandas as pd
from tfold.modeling import make_inputs

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
    df_to_save = df_to_model[['pep', 'MHC_sequence', 'pmhc_id', 'mhc_a', 'mhc_a_boundary', 'mhc_b', 'mhc_b_boundary', 'class', 'seqnn_logkd']].copy()
    df_to_save.loc[:, "mhc_a"] = df_to_save["mhc_a"].apply(lambda x: x.seq())
    df_to_save.loc[:, "mhc_b"] = df_to_save["mhc_b"].apply(lambda x: x.seq())
    df_to_save.to_csv(working_dir+'/preprocessed_input.csv', index=False)
    af_inputs=make_inputs.make_inputs(df_to_model,date_cutoff=date_cutoff,print_stats=False)
    print('total AF models to be produced:',len(af_inputs))

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
    