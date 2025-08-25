import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from timeit import default_timer as timer

from src_utils import create_action_dataset_v2, compare_trajectories, viz_op_traj_with_attention
from src_utils import get_data_split, create_mask, denormalize, visualize_output, visualize_input
from src_utils import see_steplr_trend, simulate_tgt_actions, plot_attention_weights
from utils import read_cfg_file, save_yaml, load_pkl, print_dict, save_object
from custom_models import mySeq2SeqTransformer_v1

import gym
import gym_examples
import sys
import pickle
import wandb
import imageio.v2 as imageio
import time
import argparse
from os.path import join
from datetime import datetime
import numpy as np
from root_path import ROOT
import os
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
OLD_ROOT = "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer"
from main import setup_env

def compute_dist(V_a, V_b):
    """
    Compute the average L2 distance between two sequences of vectors.
    V_a, V_b: (T, D) numpy arrays
    """
    return np.linalg.norm(V_a - V_b)


def get_dataset(dataset_path, split_tr_tst_val, split_ran_seed, random_split):
    # Load and Split dataset
    with open(dataset_path, 'rb') as f:
        traj_dataset = pickle.load(f)

    idx_split, set_split = get_data_split(traj_dataset,
                                        split_ratio=split_tr_tst_val, 
                                        random_seed=split_ran_seed, 
                                        random_split=random_split)
    train_traj_set, test_traj_set, val_traj_set = set_split
    train_idx_set, test_idx_set, val_idx_set = idx_split
    return set_split, idx_split



def get_closest_sample(train_traj_set, test_traj_set, sample_idx, include_obs_diff=False):
    train_Yis = np.array([x[0] for x in train_traj_set])
    test_Yi = test_traj_set[sample_idx][0]

    if include_obs_diff:
        train_obsts = np.array([x[1] for x in train_traj_set])
        test_obs = test_traj_set[sample_idx][1]
        train_Yis = np.concatenate((train_Yis, train_obsts), axis=2)
        test_Yi = np.concatenate((test_Yi, test_obs), axis=1)

    # compute distances between test_Yi and train_Yis
    dist_array = train_Yis - test_Yi
    dist_array = np.linalg.norm(dist_array, axis=(1,2))
    min_idx = np.argmin(dist_array)

    closest_train_sample = train_traj_set[min_idx]
    return min_idx, closest_train_sample, dist_array



def check_topk_dits(dist_array, topk=5, mode='min'):
    if mode=='min':
        topk_idx = np.argsort(dist_array)[:topk]
    elif mode=='max':
        topk_idx = np.argsort(dist_array)[-topk:]
    else:
        raise ValueError("mode should be 'min' or 'max'")
    
    topk_dists = dist_array[topk_idx]
    return topk_idx, topk_dists



def simulate_baseline_actions(test_idx, tr_idx, tr_set, test_set):

    test_timesteps, test_tgt, test_traj_mask, test_target_state, test_env_coef_seq, test_traj_len, test_idx, test_flow_dir, test_rzn = test_set[test_idx]
    tr_timesteps, tr_tgt, tr_traj_mask, tr_target_state, tr_env_coef_seq, tr_traj_len, tr_idx, tr_flow_dir, tr_rzn = tr_set[tr_idx]

    op_traj_dict = {}
    reached_target = False

    # set up environment for simulating the test flow
    # test_flow_dir = test_flow_dir[0]
    test_flow_dir = test_flow_dir.replace(OLD_ROOT, ROOT)
    test_env = setup_env(test_flow_dir, OLD_ROOT)
    test_env.reset()
    test_env.set_rzn(test_rzn)

    # For retriving V*
    # tr_flow_dir = tr_flow_dir[0]
    tr_flow_dir = tr_flow_dir.replace(OLD_ROOT, ROOT)
    tr_env = setup_env(tr_flow_dir, OLD_ROOT)
    tr_env.reset()
    tr_env.set_rzn(tr_rzn)

    # retrieve a* : optimal actions from the closest training sample
    closest_actions = tr_tgt
    max_a_steps = tr_traj_len

    # Bookkeeping
    txy_preds = np.zeros((1, context_len+1, 3),dtype=np.float32,)
    txy_preds[0,0,:] = np.array([0,test_env.start_pos[0],test_env.start_pos[1]])
    a_corrections = []
    corrected_actions = []
    velocities = []
    for i in range(context_len):
        if i >= max_a_steps:
            break
         # execute a*
        a_star = closest_actions[i]*2*np.pi
        u_star, v_star = tr_env.get_velocity(test_env.state)
        u_new, v_new = test_env.get_velocity(test_env.state)
        u_cor = u_star - u_new
        v_cor = v_star - v_new
        velocities.append(((u_star, v_star),(u_new, v_new)))
        a_corrections.append((u_cor, v_cor))
        test_env.set_a_correction((u_cor, v_cor))
        txy, reward ,done, info = test_env.step(a_star)
        corrected_actions.append(test_env.corrected_angle)
       
        txy_preds[0,i+1,:] = txy
        if done:
            if reward > 0:
                reached_target = True
            break

    op_traj_dict['states'] = np.array(txy_preds)
    op_traj_dict['closest_actions'] = closest_actions.cpu()*2*np.pi
    op_traj_dict['actions_corrections'] = a_corrections
    op_traj_dict['corrected_actions'] = corrected_actions
    op_traj_dict['velocities'] = velocities
    op_traj_dict['t_done'] = i+2
    op_traj_dict['n_tsteps'] = i+1
    op_traj_dict['success'] = reached_target
    op_traj_dict['closest_tr_sample_idx'] = tr_idx
    op_traj_dict['test_sample_idx'] = test_idx
    op_traj_dict['has_reached_target'] = info['has_reached_target']
    op_traj_dict['is_outbound'] = info['is_outbound']
    op_traj_dict['is_hitting_obs'] = info['is_hitting_obs']

    return op_traj_dict, test_env

def get_success_rate(results_list):
    n_success = sum([1 for res in results_list if res['success']])
    return n_success/len(results_list)

def get_avg_tdone(results_list):
    t_dones = [res['t_done'] for res in results_list if res['success']]
    return np.mean(t_dones)

def get_stats(results_list):
    n_success = sum([1 for res in results_list if res['success']])
    n_outbound = sum([1 for res in results_list if res['is_outbound']])
    n_hitting_obs = sum([1 for res in results_list if res['is_hitting_obs']])
    avg_a_correction_list = [res['actions_corrections'][0] for res in results_list if len(res['actions_corrections'])>0]
    n_total = len(results_list)
    print( n_total, n_success + n_outbound + n_hitting_obs)
    stats = {
        "n_success": n_success,
        "n_outbound": n_outbound,
        "n_hitting_obs": n_hitting_obs,
        "n_total": n_total,
        "success_rate": n_success/n_total,
        "outbound_rate": n_outbound/n_total,
        "hitting_obs_rate": n_hitting_obs/n_total,
        # "avg_a_correction_list": avg_a_correction_list,
    }
    return stats



def main(dataset_type, unseen_dataset_path, dataset_path, context_len,
          max_test_samples, include_obs_diff,
          use_tr_set_for_stats=False):
    random_split= True
    split_tr_tst_val= [0.8, 0.05, 0.15]
    split_ran_seed= 42

    set_split, idx_split = get_dataset(dataset_path, split_tr_tst_val, split_ran_seed, random_split)
    train_traj_set, test_traj_set, val_traj_set = set_split
    train_idx_set, test_idx_set, val_idx_set = idx_split

    # dataset contains optimal actions for different realizations of the env
    tr_set = create_action_dataset_v2(train_traj_set, 
                            train_idx_set,
                            context_len, 
                                        )

    src_stats = tr_set.get_src_stats()

    if dataset_type == "DOLS":
        test_set = create_action_dataset_v2(test_traj_set, 
                                test_idx_set,
                                context_len, 
                                norm_params_4_val = src_stats
                                            )

    elif dataset_type == "DG3":
        us_set_split, us_idx_split = get_dataset(unseen_dataset_path, split_tr_tst_val, split_ran_seed, random_split)
        us_train_traj_set, us_test_traj_set, us_val_traj_set = us_set_split
        us_train_idx_set, us_test_idx_set, us_val_idx_set = us_idx_split

        if use_tr_set_for_stats:
            used_test_traj_set = us_train_traj_set
        else:
            used_test_traj_set = us_test_traj_set

        test_set = create_action_dataset_v2(used_test_traj_set, 
                                us_test_idx_set,
                                context_len, 
                                norm_params_4_val = src_stats
                                            )

    # sample random number from 0 to len(test_traj_set)
    test_sample_idx = random.randint(0, len(test_traj_set)-1)

    results_list = []
    diff_times_list = []
    sim_times_list = []
    for test_sample_idx in range(max_test_samples):
        start_time = time.time()
        test_sample = test_traj_set[test_sample_idx]

        min_idx_tr, closest_train_sample, dist_array = get_closest_sample(train_traj_set, test_traj_set, test_sample_idx, dataset_type)
        diff_end_time = time.time()
        
        results, test_env = simulate_baseline_actions(test_sample_idx, min_idx_tr, tr_set, test_set)
        sim_end_time = time.time()
        results_list.append(results)

        diff_time = (diff_end_time - start_time)
        sim_time = (sim_end_time - diff_end_time)
        diff_times_list.append(diff_time)
        sim_times_list.append(sim_time)

        if (test_sample_idx+1)%50 == 0:
            print(f"Processed {test_sample_idx+1}/{max_test_samples} test samples") 
            # print(f"Avg. time for finding closest sample: {np.mean(diff_times_list)} sec")
            # print(f"Avg. time for simulating test sample: {np.mean(sim_times_list)} secs")

    # print(f"unseen dataset path: {unseen_dataset_path.split('/')[-1]}")
    print(f"{use_tr_set_for_stats=}")
    print(f"{include_obs_diff=}")
    avg_success_rate = get_success_rate(results_list)
    avg_tdone = get_avg_tdone(results_list)
    print(f"Avg. Success Rate = {avg_success_rate*100:.2f}")
    print(f"Avg. T_done = {avg_tdone:.2f}")
    print(get_stats(results_list))
    print()
    return results_list, test_env






if __name__ == "__main__":

    dataset_type = 'DG3'  # 'DOLS' or 'DG3'

    if dataset_type == 'DOLS':
        dataset_path = '/home/sumanth/rohit/Translation_transformer/my-translat-transformer/data/DOLS_Cylinder/targ_5/gathered_targ_5.pkl'
        context_len = 101
        max_test_samples = 500
        include_obs_diff = False
        # dummy value since not needed for DOLS
        unseen_dataset_path_list = [None]
        use_tr_set_for_stats = True
        files = ["DOLS"]

    elif dataset_type == 'DG3':
        dataset_path = '/home/sumanth/rohit/Translation_transformer/my-translat-transformer/data/GPT_dset_DG3/static_obs/GPTdset_DG3_g100x100x120_r5k_Obsv1_w5_1dataset_v2.pkl'
        unseen_dataset_path = '/home/sumanth/rohit/Translation_transformer/my-translat-transformer/data/GPT_dset_DG3/static_obs/GPTdset_DG3_g100x100x120_r5k_Obsv1_w5_1dataset_single_43475.pkl'
        # unseen_dataset_path = '/home/sumanth/rohit/Translation_transformer/my-translat-transformer/data/GPT_dset_DG3/static_obs/GPTdset_DG3_g100x100x120_r5k_Obsv1_w5_1dataset_single_45565.pkl'
        context_len = 120
        max_test_samples = 1000
        include_obs_diff = True
        use_tr_set_for_stats = True
        # files = [25305,40625,43475,43475,45525,45565]
        files = [43475,45565,25305]

        base_path = '/home/sumanth/rohit/Translation_transformer/my-translat-transformer/data/GPT_dset_DG3/static_obs/GPTdset_DG3_g100x100x120_r5k_Obsv1_w5_1dataset_single_'
        unseen_dataset_path_list = [f"{base_path}{file}.pkl" for file in files]
    
    else:
        raise ValueError("dataset_type should be 'DOLS' or 'DG3'")


    # if DOLS, then unseen_dataset_path is not needed
    for idx, unseen_dataset_path in enumerate(unseen_dataset_path_list):
        result_list, test_env = main(dataset_type, unseen_dataset_path, dataset_path, context_len, max_test_samples, include_obs_diff,
                                     use_tr_set_for_stats=use_tr_set_for_stats)

        env_4_viz = test_env

        save_dir = "/home/sumanth/rohit/Translation_transformer/my-translat-transformer/tmp"
        torch.save(result_list, join(save_dir,f"corrected_simple_baseline_{files[idx]}_results.pt"))

        val_set_txy_preds = [d['states'] for d in result_list]
        path_lens = [d['n_tsteps'] for d in result_list]
        visualize_output(val_set_txy_preds, 
                            path_lens,
                            iter_i = 0, 
                            stats=None, 
                            env=env_4_viz, 
                            log_wandb=False, 
                            plot_policy=False,
                            traj_idx=None,      #None=all, list of rzn_ids []
                            show_scatter=True,
                            at_time=None,
                            color_by_time=True, #TODO: fix tdone issue in src_utils
                            plot_flow=True,
                            wandb_suffix="val",
                            model_name=f"corrected_simple_baseline_obsdiff_{include_obs_diff}_{files[idx]}_",
                           
                            ) 