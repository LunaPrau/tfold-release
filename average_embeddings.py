import numpy as np
import os
import shutil
from typing import List
from scipy.special import softmax

def compute_plddt(lddt_logits: np.ndarray) -> np.ndarray:
    """Computes per-residue pLDDT from logits (shape [N, 50]). Returns array of shape [N]."""
    num_bins = lddt_logits.shape[-1]
    bin_width = 1.0 / num_bins
    bin_centers = np.arange(start=bin_width / 2.0, stop=1.0, step=bin_width)
    probs = softmax(lddt_logits, axis=-1)
    predicted_lddt = np.sum(probs * bin_centers[None, :], axis=-1)
    return 100.0 * predicted_lddt

def average_weighted_single_representations(single_reps_list: List[np.ndarray], weights: List[np.ndarray]) -> np.ndarray:
    single_reps = np.stack(single_reps_list)
    sum_weights = np.sum(weights, axis=0)[:, None]
    weighted_sum = np.sum(weights[:, :, None] * single_reps, axis=0)
    avg_single = weighted_sum / np.maximum(sum_weights, 1e-8)
    return avg_single

def average_weighted_pair_representations(pair_reps_list: List[np.ndarray], weights: List[np.ndarray]) -> np.ndarray:
    pair_weights = weights[:, :, None] * weights[:, None, :]
    pair_reps = np.stack(pair_reps_list)
    sum_pair_weights = np.sum(pair_weights, axis=0)[:, :, None]
    weighted_pair_sum = np.sum(pair_weights[:, :, :, None] * pair_reps, axis=0)
    avg_pair = weighted_pair_sum / np.maximum(sum_pair_weights, 1e-8)
    return avg_pair

def main(work_dir, target_id=None):
    target_dir = os.path.join(work_dir, str(target_id))

    single_reps_list = []
    pair_reps_list = []
    lddt_logits_list = []

    current_id_dirs = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
    for current_id in current_id_dirs:
        current_dir = os.path.join(target_dir, current_id)
        model_dir = os.path.join(current_dir, 'model_1')

        if os.path.exists(model_dir):
            # Load embeddings
            lddt_path = os.path.join(model_dir, 'lddt_embeddings.npz')
            pair_path = os.path.join(model_dir, 'pair_embeddings.npz')
            single_path = os.path.join(model_dir, 'single_embeddings.npz')

            if os.path.exists(lddt_path) and os.path.exists(pair_path) and os.path.exists(single_path):
                lddt_data = np.load(lddt_path)
                pair_data = np.load(pair_path)
                single_data = np.load(single_path)

                lddt_logits_list.append(lddt_data['embeddings'])
                pair_reps_list.append(pair_data['embeddings'])
                single_reps_list.append(single_data['embeddings'])

            # Copy structure_...pdb to target_id dir
            pdb_src = os.path.join(model_dir, f'structure_{current_id}_model_1.pdb')
            pdb_dest = os.path.join(target_dir, f'structure_{current_id}.pdb')
            if os.path.exists(pdb_src) and not os.path.exists(pdb_dest):
                shutil.copy(pdb_src, pdb_dest)

            # Copy distogram to target_id dir
            dist_src = os.path.join(model_dir, f'distogram')
            dist_dest = os.path.join(target_dir, f'distogram_{current_id}')
            if os.path.exists(dist_src) and not os.path.exists(dist_dest):
                shutil.copy(dist_src, dist_dest)

            # Copy results to target_id dir
            result_src = os.path.join(model_dir, f'result_{current_id}_model_1.pkl')
            result_dest = os.path.join(target_dir, f'result_{current_id}.pkl')
            if os.path.exists(result_src) and not os.path.exists(result_dest):
                shutil.copy(result_src, result_dest)

            # Delete current_id directory
            shutil.rmtree(current_dir)

    M = len(lddt_logits_list)
    weights = np.stack([compute_plddt(lddt_logits_list[m]) / 100.0 for m in range(M)])
    
    # Calculate weighted averages
    if single_reps_list:
        avg_single = average_weighted_single_representations(single_reps_list, weights)
        np.savez_compressed(os.path.join(target_dir, 'single_embeddings.npz'), avg_single.astype(np.float32))

    if pair_reps_list:
        avg_pair = average_weighted_pair_representations(pair_reps_list, weights)
        np.savez_compressed(os.path.join(target_dir, 'pair_embeddings.npz'), avg_pair.astype(np.float32))

    # Calculate weighted average lddt_embeddings
    if lddt_logits_list:
        # Average the logits directly
        avg_lddt_logits = np.mean(np.stack(lddt_logits_list), axis=0)
        np.savez_compressed(os.path.join(target_dir, 'lddt_embeddings.npz'), avg_lddt_logits.astype(np.float32))

if __name__ == "__main__":
    work_dir = "/home/marta_aikium_com/MHC_peptide_binding_model/src/output/embeddings/tfold/0300-0355/outputs/"
    target_ids = os.listdir(work_dir)
    for target_id in target_ids:
        main(work_dir, target_id)