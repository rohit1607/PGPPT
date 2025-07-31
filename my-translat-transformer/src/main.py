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


wandb.login()
OLD_ROOT = "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer"



def setup_env(flow_dir, old_root):
    flow_specific_cfg = read_cfg_file(cfg_name=join(flow_dir,"cfg_used_in_proc_data.yml"))
    env_name = flow_specific_cfg["env_name"]
    params2 = read_cfg_file(cfg_name=join(flow_dir,"params.yml"))
    env = gym.make(env_name)
    # env.if_scale_velocity = True
    env.setup(flow_specific_cfg, params2, add_trans_noise=False, 
              old_root=old_root, current_root=ROOT)
    return env

def extract_attention_scores(model):
    enc_sa_arr = np.array([layer.enc_avg_att_scores.cpu().detach().numpy() for layer in model.transformer.encoder.layers])
    dec_sa_arr = np.array([layer.dec_avg_att_scores.cpu().detach().numpy() for layer in model.transformer.decoder.layers])
    dec_ga_arr = np.array([layer.dec_avg_cross_att_scores.cpu().detach().numpy() for layer in model.transformer.decoder.layers])

    return enc_sa_arr , dec_sa_arr, dec_ga_arr 

def plot_all_attention_mats(all_att_mats, log_wandb=True, model_name=''):
    enc_sa_arr , dec_sa_arr, dec_ga_arr = all_att_mats
    save_path = "/home/rohit/Documents/Research/Planning_with_transformers/Translation_transformer/my-translat-transformer/tmp/last_exp_figs"

    plot_attention_weights(enc_sa_arr, 
                            layer_idx=0,
                            set_idx=0, 
                            average_across_layers=True,
                            scale_each_row=True, 
                            causal_mask_used=False,
                            log_wandb = log_wandb,
                            fname=join(save_path,'attention_heatmap_'+model_name),
                            info_string = 'enc_sa',
                            wandb_fname = 'enc_sa'   
                            )
    plot_attention_weights(dec_sa_arr, 
                            layer_idx=0,
                            set_idx=0, 
                            average_across_layers=True,
                            scale_each_row=True, 
                            causal_mask_used=True,
                            log_wandb = log_wandb,
                            fname=join(save_path,'attention_heatmap'+model_name),
                            info_string = 'dec_sa',
                            wandb_fname = 'dec_sa'                 
                            )
    plot_attention_weights(dec_ga_arr, 
                            layer_idx=0,
                            set_idx=0, 
                            average_across_layers=True,
                            scale_each_row=True, 
                            causal_mask_used=False,
                            log_wandb = log_wandb,
                            fname=join(save_path,'attention_heatmap'+model_name),
                            info_string = 'dec_ga',
                            wandb_fname = 'dec_ga'                 
                            )
    
    return 

def train_epoch(model, optimizer, tr_set, cfg, args, scheduler=None, log_interval=50):
    model.train()
    # losses = 0
    avg_loss = 0
    # TODO: take it outside
    train_dataloader = DataLoader(tr_set, batch_size=cfg.batch_size, shuffle=True)
    loss = 0
    count=  0
    # for env_coef_seq, tgt in train_dataloader:
    for timesteps, tgt, traj_mask, target_state, env_coef_seq, traj_len, idx, _, _ in train_dataloader:
        timesteps = timesteps.to(cfg.device)
        src = env_coef_seq.to(cfg.device)
        tgt = tgt.to(cfg.device)

        tgt_input = tgt[:, :-1, :]
        tgt_padding_mask = traj_mask.to(cfg.device)
        src_mask, tgt_mask, src_padding_mask, _ = create_mask(src, tgt_input, traj_len, cfg.device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask, timesteps)

        tgt_out = tgt[:, 1:, :]
        mask_for_loss = torch.clone(tgt_padding_mask).to(cfg.device)
        mask_for_loss[:,:-1] = ~tgt_padding_mask[:,1:]
        mask_for_loss[:,-1] = tgt_padding_mask[:,0]
        # mask_for_loss = torch.cat([~tgt_padding_mask[:,1:],tgt_padding_mask[:,0]])
       
        # only consider non padded elements (except one at the end)
        # logits_ =  logits.view(-1,1)[(~tgt_padding_mask).view(-1,)]
        # tgt_out_ = tgt_out.reshape(-1,1)[(~tgt_padding_mask).view(-1,)]

        # only considers purely non-padded elements in predictions
        logits =  logits.view(-1,1)[mask_for_loss.view(-1,)]
        tgt_out = tgt_out.reshape(-1,1)[mask_for_loss.view(-1,)]
        # Consider all elements across context length
        # logits =  logits.view(-1,1)
        # tgt_out = tgt_out.reshape(-1,1)
        # loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        
        # TODO: restrict output to 0-6 or 0-1 if scaled 
        loss = F.mse_loss(logits, tgt_out)
        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 4)

        optimizer.step()
        # if not scheduler == None:
        #     scheduler.step()
        # losses += loss.item()
        avg_loss = avg_loss + ((loss.item() - avg_loss)/(count+1))

        if count%log_interval == 0:
            param_norms = [p.grad.data.norm(2).item() for p in model.parameters()]
            mean_norm = np.mean(param_norms)
            min_norm = np.min(param_norms)
            max_norm = np.max(param_norms)
            wandb.log({f"in_eval/avg_train_loss vs log_intervalth update": avg_loss,
                        "in_eval/MAX_param_norm": max_norm,
                        "in_eval/MIN_param_norm": min_norm,
                        "in_eval/AVG_param_norm": mean_norm,
                       })
        count += 1

    # Note: attention extraction was done by modificatin in the library. 
    # all_att_mats = extract_attention_scores(model)
    all_att_mats = None
    return avg_loss, all_att_mats


def evaluate(model, val_set, cfg, log_interval=10):
    model.eval()
    # losses = 0
    avg_loss = 0
    bs = cfg.batch_size
    val_dataloader = DataLoader(val_set, batch_size=bs, shuffle=True)

    count=  0
    for timesteps, tgt, traj_mask, target_state, env_coef_seq, traj_len, idx, _, _ in val_dataloader:
        timesteps = timesteps.to(cfg.device)
        src = env_coef_seq.to(cfg.device)
        tgt = tgt.to(cfg.device)
        tgt_input = tgt[:, :-1, :]
        tgt_padding_mask = traj_mask.to(cfg.device)
        src_mask, tgt_mask, src_padding_mask, _ = create_mask(src, tgt_input, traj_len, cfg.device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask,timesteps)
        
        tgt_out = tgt[:, 1:, :]
        mask_for_loss = torch.clone(tgt_padding_mask).to(cfg.device)
        mask_for_loss[:,:-1] = ~tgt_padding_mask[:,1:]
        mask_for_loss[:,-1] = tgt_padding_mask[:,0]
        # mask_for_loss = torch.cat([~tgt_padding_mask[:,1:],tgt_padding_mask[:,0]])
       
        # # only consider non padded elements (except one at the end)
        # logits_ =  logits.view(-1,1)[(~tgt_padding_mask).view(-1,)]
        # tgt_out_ = tgt_out.reshape(-1,1)[(~tgt_padding_mask).view(-1,)]

        # only considers purely non-padded elements in predictions
        logits =  logits.view(-1,1)[mask_for_loss.view(-1,)]
        tgt_out = tgt_out.reshape(-1,1)[mask_for_loss.view(-1,)]

        # logits =  logits.view(-1,1)[(~tgt_padding_mask).view(-1,)]
        # tgt_out = tgt_out.reshape(-1,1)[(~tgt_padding_mask).view(-1,)]
        # loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss = F.mse_loss(logits, tgt_out)
        
        # losses += loss.item()
        avg_loss = avg_loss + ((loss.item() - avg_loss)/(count+1))
        if count%log_interval == 0:
            wandb.log({f"in_eval/avg_val_loss vs log_intervalth update": avg_loss})
        count += 1

    # NOTE: Uncomment post re-implementation or library-fix
    # all_att_mats = extract_attention_scores(model)
    all_att_mats = None
    return avg_loss, all_att_mats


def translate(model: torch.nn.Module, test_idx, test_set, tr_set_stats, cfg, earlybreak=10**8):
    model.eval()
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)
    count = 0           # keeps count of total episodes
    success_count = 0   # keeps count of successful episodes
    success_count_ = 0
    op_traj_dict_list = []
    translate_one_rzn_list = []
    with torch.no_grad():
        # for sample in range(len(test_set)):
        # timesteps, tgt, traj_mask, target_state, env_coef_seq, traj_len, idx = test_set[sample]

        for timesteps, tgt, traj_mask, target_state, env_coef_seq, traj_len, idx, flow_dir, rzn in test_dataloader:
            if idx%100==0:
                print(idx)
            # translate_one_rzn = timer()
            # set up environment
            flow_dir = flow_dir[0]
            flow_dir = flow_dir.replace(OLD_ROOT, ROOT)
            env = setup_env(flow_dir, OLD_ROOT)

            op_traj_dict = {}
            reached_target = False
            reached_target_ = False

            env.reset()

            idx = idx[0].item() #initially idx = tensor([0])
            env.set_rzn(rzn)

            timesteps = timesteps.to(cfg.device)
            count += 1
            if count == earlybreak:
                break
            src = env_coef_seq.to(cfg.device)
            dummy_tgt_for_mask = tgt.to(cfg.device)[:, :-1, :]
            src_mask, tgt_mask, src_padding_mask, _ = create_mask(src, dummy_tgt_for_mask, traj_len, cfg.device)
            # memory is the encoder output
            memory =  model.encode(src, src_mask, timesteps)

            preds = torch.zeros((1, cfg.context_len, dummy_tgt_for_mask.shape[2]),dtype=torch.float32, device=cfg.device)
            PREDS_ = torch.zeros((1, cfg.context_len, dummy_tgt_for_mask.shape[2]),dtype=torch.float32, device=cfg.device)
            
            txy_preds = np.zeros((1, cfg.context_len+1, 3),dtype=np.float32,)
            txy_preds[0,0,:] = np.array([0,env.start_pos[0],env.start_pos[1]])
            # NOTE: Find commit where SOS token was used.
            preds[0,0,:] = tgt[0,0,:]
            a = preds[0,0,:].cpu().numpy().copy()
            a = a*2*np.pi
            txy, reward ,done, info = env.step(a)
            txy_preds[0,1,:] = txy     

            for i in range(cfg.context_len-1):
                memory = memory.to(cfg.device)
                out = model.decode(preds, memory, tgt_mask, timesteps)
                gen = model.generator(out)
                preds[0,i+1,:] = gen[0,i,:].detach()
                a = preds[0,i+1,:].cpu().numpy().copy()
                a = a*2*np.pi
                txy, reward ,done, info = env.step(a)
                txy_preds[0,i+2,:] = txy 
                # NOTE: reduce GPU-CPU communication
                if done:
                    if reward > 0:
                        reached_target = True
                        success_count += 1
                    break

            mse = F.mse_loss(preds[0,:i].cpu(),tgt[0,:i].cpu())

            op_traj_dict['states'] = np.array(txy_preds)
            op_traj_dict['actions'] = preds.cpu()*2*np.pi
            op_traj_dict['t_done'] = i+3
            op_traj_dict['n_tsteps'] = i+2
            # op_traj_dict['attention_weights'] = attention_weights
            op_traj_dict['success'] = reached_target
            op_traj_dict['mse'] = mse
            # op_traj_dict['all_att_mat'] = extract_attention_scores(model)
            # op_traj_dict['states_for_action_labels'] = np.array(TXY_PREDS_)
            op_traj_dict['states_for_action_labels'] = None
            op_traj_dict['action_labels'] = tgt.cpu()*2*np.pi
            # op_traj_dict['t_done_fal'] = k+1
            # op_traj_dict['n_tsteps_fal'] = k
            # op_traj_dict['attention_weights'] = attention_weights
            op_traj_dict['success_fal'] = reached_target_
            op_traj_dict_list.append(op_traj_dict)

    # mean_translate_time = np.mean(np.array(translate_one_rzn_list)[1:101])
    # print(f'Avg translate time for 100 rzns: {mean_translate_time}')
    # print(np.array(translate_one_rzn_list)[1:101].shape)
    results = {}
    results['avg_val_loss'] = np.mean([d['mse'] for d in op_traj_dict_list])
    results['translate/avg_ep_len'] = np.mean([d['n_tsteps'] for d in op_traj_dict_list])
    results['translate/success_ratio'] = success_count/count
    results['runs_from_set(count)'] = count

    return op_traj_dict_list, results



"""
Main training function
"""

def train_model(args=None, cfg_name=None):

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%m-%d-%H-%M")
    
    if cfg_name != None:
        cfg = read_cfg_file(cfg_name=cfg_name) 
        config = cfg
        dataset_name = cfg['dataset_name']
    else:
        defaults = dict()
        config = defaults
        dataset_name = NAME_MAP[ARGS_CFG] #for sweep mode
    
    cfg_copy = cfg.copy()

    wandb_exp_name = "env2a_" + dataset_name + "__" + start_time_str
    wandb.init(project="translation-transformer",
        name = wandb_exp_name,
        config=config
        )

    cfg=wandb.config
    # cfg_copy = cfg

    params2 = read_cfg_file(cfg_name=join(ROOT,cfg.params2_name))

    add_trans_noise = cfg.add_transition_noise_during_inf

    env_name = cfg.env_name
    split_tr_tst_val = cfg.split_tr_tst_val
    split_ran_seed = cfg.split_ran_seed
    random_split = cfg.random_split
    max_eval_ep_len = cfg.max_eval_ep_len  # max len of one episode
    num_eval_ep = cfg.num_eval_ep       # num of evaluation episodes


    batch_size = cfg.batch_size           # training batch size
    optimizer_name = cfg.optimizer_name
    lr = cfg.lr    
    final_lr = cfg.final_lr                        # const learning rate
    wt_decay = cfg.wt_decay               # weight decay
    use_scheduler = cfg.use_scheduler
    warmup_steps = cfg.warmup_steps       # warmup steps for lr scheduler

    # total_updates = max_train_iters x num_updates_per_iter
    num_epochs = cfg.num_epochs
    # num_updates_per_iter = cfg.num_updates_per_iter
    # comp_val_loss = cfg.comp_val_loss
    eval_inerval =  cfg.eval_inerval
    target_conditioning = cfg.target_conditioning
    num_encoder_layers = cfg.num_encoder_layers
    num_decoder_layers = cfg.num_decoder_layers
    context_len = cfg.context_len     # K in decision transformer
    embed_dim = cfg.embed_dim          # embedding (hidden) dim of transformer
    n_heads = cfg.n_heads            # num of transformer heads
    dropout_p = cfg.dropout_p         # dropout probability

    # load data from this file
    dataset_path = join(ROOT,cfg.dataset_path)
    dataset_name = cfg.dataset_name
    # saves model and csv in this directory
    log_dir = join(ROOT, cfg.log_dir)
    tt_eb = cfg.translate_earlybreaks

    # training and evaluation device
    device = torch.device(cfg.device)
    

    if ARGS_QR:
        print("\n ---------- Modifying cfg params for quick run --------------- \n")
        num_epochs = 10



    prefix = "my_translat_" + dataset_name

    save_model_name =  prefix + "_model_" + start_time_str + ".pt"
    save_model_path = join(log_dir, save_model_name)
    save_yaml(join(ROOT, "log",f"{save_model_name[:-3]}.yml"), cfg_copy)


    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    print("device set to: " + str(device))
    print("dataset path: " + dataset_path)
    print("model save path: " + save_model_path)

 
    # Load and Split dataset
    with open(dataset_path, 'rb') as f:
        traj_dataset = pickle.load(f)

    idx_split, set_split = get_data_split(traj_dataset,
                                        split_ratio=split_tr_tst_val, 
                                        random_seed=split_ran_seed, 
                                        random_split=random_split)
    train_traj_set, test_traj_set, val_traj_set = set_split
    train_idx_set, test_idx_set, val_idx_set = idx_split


    # dataset contains optimal actions for different realizations of the env
    tr_set = create_action_dataset_v2(train_traj_set, 
                            train_idx_set,
                            context_len, 
                                        )

    src_stats = tr_set.get_src_stats()
    src_stats_path = save_model_path[:-3] +"_src_stats.npy"
    np.save(src_stats_path, src_stats)
    val_set = create_action_dataset_v2(val_traj_set, 
                            val_idx_set,
                            context_len,
                            norm_params_4_val = src_stats
                                        )
    test_set = create_action_dataset_v2(test_traj_set, 
                            val_idx_set,
                            context_len, 
                            norm_params_4_val = src_stats
                                        )


    _, dummy_target, _, _, dummy_env_coef_seq, _,_,dummy_flow_dir,_ = tr_set[0]
    src_vec_dim = dummy_env_coef_seq.shape[-1]
    tgt_vec_dim = dummy_target.shape[-1]
    print(f"src_vec_dim = {src_vec_dim} \n tgt_vec_dim = {tgt_vec_dim}")
    
    # intantiate gym env for vizualization purposes
    # old root is needed because params_in_proc data config file has 
    # references to paths on the machine used for dataset creation
    dummy_flow_dir = dummy_flow_dir.replace(OLD_ROOT, ROOT)
    env_4_viz = setup_env(dummy_flow_dir, OLD_ROOT)
 
    # data visualization for sanity check
    break_at = 200 #for debugging only
    if "DG3" in dataset_name:
        t_viz = 119
    elif "DOLS" in dataset_name:
        t_viz = 100
    else:
        raise NotImplementedError
    # plot xy coords of paths from existing solver
    visualize_input(tr_set, log_wandb=True, at_time=t_viz, env=env_4_viz, break_at=break_at)
    # simulate actions that were processed from optimal paths.
    simulate_tgt_actions(tr_set,
                            env=env_4_viz,
                            log_wandb=True,
                            wandb_fname='simulate_tgt_actions',
                            plot_flow=True,
                            at_time=t_viz,
                            break_at=break_at)
    
    
    # Instantiate the model
    transformer = mySeq2SeqTransformer_v1(num_encoder_layers, num_decoder_layers, embed_dim,
                                 n_heads, src_vec_dim, tgt_vec_dim, 
                                 dim_feedforward=None,     
                                 max_len=context_len,
                                 positional_encoding="simple"
                                 ).to(cfg.device)
    
    # Instantiate the optimizer
    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(
                        transformer.parameters(),
                        lr=lr,
                        weight_decay=wt_decay
                    )
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(transformer.parameters(), 
                                    lr=lr, 
                                    weight_decay=wt_decay,
                                    betas=(0.9, 0.98), 
                                    eps=1e-9, 
                                    )    
    # Set up the scheduler
    gamma, _ = see_steplr_trend(step_size=10, num_epochs=num_epochs, lr=lr, final_lr=final_lr)
    main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)  
    warm_up_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,
                                                          start_factor=0.033,
                                                          total_iters=3
                                                          )
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                      schedulers=[warm_up_scheduler, main_lr_scheduler],
                                                      milestones=[3])

    # Compute number of trainable params
    pytorch_trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    pytorch_total_params = sum(p.numel() for p in transformer.parameters())
    print(f"total params = {pytorch_total_params}")
    print(f"trainable params = {pytorch_trainable_params}")
    wandb.run.summary["total params"] = pytorch_total_params
    wandb.run.summary["trainable params"] = pytorch_trainable_params
    print(f"{len(tr_set)=}")

    # Training Loop
    min_ETA = 10**5
    max_sr = -1
    for epoch in range(0, num_epochs+1):
        print(f"epoch {epoch}")
        epoch_start_time = timer()
        print("training")
        train_loss, tr_all_att_mat = train_epoch(transformer, optimizer, tr_set, cfg, args, scheduler=scheduler)
        epoch_end_time = timer()
        print("evaluating")
        val_loss, val_all_att_mat = evaluate(transformer, val_set, cfg)
        scheduler.step()
        wandb.log({f"in_eval/lr":  scheduler.get_last_lr()[0]
                       })
        
        # # TODO: requires all data to be here
        # # Evalutation by translation   
        if epoch % eval_inerval == 0:
        #     print("plotting attention")
        #     # TODO: using the plot_all_attention_mats requires library changes to extract attention_weights
        #     # plot_all_attention_mats(tr_all_att_mat)
        #     # plot_all_attention_mats(val_all_att_mat)

                        
            val_op_traj_dict_list, val_results = translate(transformer, val_idx_set, val_set, None, 
                                                           cfg, earlybreak=tt_eb[1])
            val_set_txy_preds = [d['states'] for d in val_op_traj_dict_list]
            path_lens = [d['n_tsteps'] for d in val_op_traj_dict_list]
            visualize_output(val_set_txy_preds, 
                                path_lens,
                                iter_i = 0, 
                                stats=None, 
                                env=env_4_viz, 
                                log_wandb=True, 
                                plot_policy=False,
                                traj_idx=None,      #None=all, list of rzn_ids []
                                show_scatter=True,
                                at_time=None,
                                color_by_time=True, #TODO: fix tdone issue in src_utils
                                plot_flow=True,
                                wandb_suffix="val") 

        #Note: val_results get updated after eval_inerval
        translate_avg_ep_len = val_results['translate/avg_ep_len']
        translate_avg_val_loss = val_results['avg_val_loss']
        success_ratio = val_results['translate/success_ratio']
        # tr_success_ratio = tr_results['translate/success_ratio']
        count = val_results['runs_from_set(count)'] 
        log_dict = { "tr_loss_vs_epoch (unpadded elems)": train_loss,
            "val_loss_vs_epoch (unpadded elems)": val_loss,
            "avg_val_loss (across pred len)": translate_avg_val_loss,
            "success_ratio": success_ratio,
            # "succes_ratio (train)": tr_success_ratio,
            'runs_from_set(count)': count,
            "ETA": translate_avg_ep_len,
            # "lr" : scheduler.get_last_lr()[0] if use_scheduler else lr
            }
        wandb.log(log_dict)

    
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)
        print('='*60)
        print(f"Epoch: {epoch}")
        # for key, val in log_dict.items():
        #     print(f"{key}: {format(val,'.4f')}")
        print(f"Epoch runtime = {(epoch_end_time - epoch_start_time):.3f}s")
        print(f"time elapsed: {time_elapsed}")
        print("")
 
        
        # TODO: need to translate to save model. commenting for now
        # if translate_avg_ep_len < min_ETA:
        if success_ratio > max_sr:
            # min_ETA = translate_avg_ep_len
            max_sr = success_ratio
            print("saving current model at: " + save_model_path)

            best_avg_episode_length = translate_avg_ep_len
            best_success_ratio = success_ratio
            best_epoch = epoch
            # "avg_val_loss"= eval_avg_val_loss

            torch.save(transformer, save_model_path)
            # tmp_path = save_model_path[:-1]
            # torch.save(transformer, tmp_path)


    # tmp_path = save_model_path[:-1]
    cfg_copy_path = save_model_path[:-3] + "_wandb.yml"
    save_yaml(cfg_copy_path,cfg)
    wandb.run.summary["best_avg_episode_length"] = best_avg_episode_length
    wandb.run.summary["best_success_ratio"] = best_success_ratio
    wandb.run.summary["total_runs_val_set"] = len(val_op_traj_dict_list)
    wandb.run.summary["best_epoch"] = best_epoch

    print("=" * 60)
    print("finished training!")
    print("=" * 60)



    print(f"\n\n ---- running inference on test set ----- \n\n")
    transformer = torch.load(save_model_path)
    try:
        op_traj_dict_list, results  = translate(transformer,test_idx_set, test_set, 
                                                None, cfg, earlybreak=tt_eb[2])
        test_set_txy_preds = [d['states'] for d in op_traj_dict_list]
        path_lens = [d['n_tsteps'] for d in op_traj_dict_list]  
        visualize_output(test_set_txy_preds, 
                            path_lens,
                            iter_i = 0, 
                            stats=None, 
                            env=env_4_viz, 
                            log_wandb=True, 
                            plot_policy=False,
                            traj_idx=None,      #None=all, list of rzn_ids []
                            show_scatter=True,
                            at_time=None,
                            color_by_time=True, #TODO: fix tdone issue in src_utils
                            plot_flow=True,
                            wandb_suffix="test")
    except:
        print("probably data missing")
        print(f"test_set path = {test_set[0][-2]}")
    
    end_time = datetime.now().replace(microsecond=0)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("saved last updated model at: " + save_model_path)
    print("=" * 60)

    return best_avg_episode_length
    # return



"""
For running manual inference on a checkpoint
"""

class fix_cfg:
    def __init__(self, context_len, device):
        self.context_len = context_len
        self.device = device

def inference_on_ckpt(args):
    wandb_exp_name = "inference_on_ckpt"
    wandb.init(project="translation-transformer",
        name = wandb_exp_name,
        )
    # Load model
    # tmp_path = ROOT + "log/my_translat_GPTdset_DG3_model_08-24-17-28.pt" 
    ckpt_path = args.ckpt_path
    transformer = torch.load(ckpt_path)
    model_name = ckpt_path[:-3].split('/')[-1]
    run_cfg_path = ckpt_path[:-3] + ".yml"
    src_stats_path = ckpt_path[:-3] + "_src_stats.npy"


    if not os.path.exists(run_cfg_path):
        print(f"run_cfg not found at {run_cfg_path}, using default cfg")
        raise NotImplementedError
    
    cfg = read_cfg_file(cfg_name=run_cfg_path)

    # load unseen dataset
    dset = 'test'
    dataset_path = cfg['dataset_path']
    
    traj_dataset = load_pkl(dataset_path)
    dataset_name = cfg['dataset_name']

    src_stats = np.load(src_stats_path)
    src_stats = (src_stats[0], src_stats[1])

    idx_split, set_split = get_data_split(traj_dataset,
                                    split_ratio=cfg['split_tr_tst_val'], 
                                    random_seed=cfg['split_ran_seed'],
                                    random_split=True)
    
    us_train_traj_set, us_test_traj_set, us_val_traj_set = set_split
    us_train_idx_set, us_test_idx_set, us_val_idx_set = idx_split

    us_test_traj_set = create_action_dataset_v2(us_test_traj_set, 
                            idx_set=us_test_idx_set,
                            context_len=cfg['context_len'],
                            norm_params_4_val = src_stats
                                        )
    
    if "DG3" in dataset_name:
        t_viz = 119
        from GPT_paper_plots import paper_plots

    elif "DOLS" in dataset_name:
        t_viz = 100
        from paper_plots import paper_plots

    else:
        raise NotImplementedError

    _, dummy_target, _, _, dummy_env_coef_seq, _,_,dummy_flow_dir,_ = us_test_traj_set[0]
    src_vec_dim = dummy_env_coef_seq.shape[-1]
    tgt_vec_dim = dummy_target.shape[-1]
    print(f"src_vec_dim = {src_vec_dim} \n tgt_vec_dim = {tgt_vec_dim}")
    
    # intantiate gym env for vizualization purposes
    dummy_flow_dir = dummy_flow_dir.replace(OLD_ROOT, ROOT)
    env_4_viz = setup_env(dummy_flow_dir, OLD_ROOT)
    simulate_tgt_actions(us_test_traj_set,
                            env=env_4_viz,
                            log_wandb=True,
                            wandb_fname='simulate_tgt_actions',
                            plot_flow=True,
                            at_time=t_viz,
                            break_at=50)
    
    cfg_ = fix_cfg(context_len=cfg['context_len'], device=cfg['device'])
    op_traj_dict_list, results = translate(transformer, us_test_idx_set, us_test_traj_set, 
                                            None, cfg_, earlybreak=100)

    
    os.makedirs(os.path.dirname(ROOT + f"paper_plots/{model_name}/{dataset_name}/{dset}_op_traj_dict_list.pkl"),exist_ok=True)
    os.makedirs(os.path.dirname(ROOT + f"paper_plots/{model_name}/{dataset_name}/{dset}_results.pkl"), exist_ok=True)
    save_object(op_traj_dict_list, os.path.join(ROOT, f"paper_plots/{model_name}/{dataset_name}/{dset}_op_traj_dict_list.pkl"))
    save_object(results,os.path.join(ROOT, f"paper_plots/{model_name}/{dataset_name}/{dset}_results.pkl"))

    op_traj_dict_list = load_pkl(os.path.join(ROOT, f"paper_plots/{model_name}/{dataset_name}/{dset}_op_traj_dict_list.pkl"))
    results = load_pkl(os.path.join(ROOT, f"paper_plots/{model_name}/{dataset_name}/{dset}_results.pkl"))
    

    test_set_txy_preds = [d['states'] for d in op_traj_dict_list]
    path_lens = [d['n_tsteps'] for d in op_traj_dict_list]
    # all_att_mat_list =  [d['all_att_mat'] for d in op_traj_dict_list]
    success_list = [d['success'] for d in op_traj_dict_list]
    actions = [d['actions'] for d in op_traj_dict_list]
    

    # taken from vis_traj_with_attention.py in my decision transformer project
    print(f"model_name = {model_name}")
    # save_dir = "paper_plots/"  + model_name + f"/DOLS_targ_{targ}/increased_cbar"
    save_dir = "paper_plots/"  + model_name + f"/{dataset_name}/translation"    
    save_dir = join(ROOT,save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    paper_plot_info = {"trajs_by_arr": {"fname":"T_arr"},
                    "trajs_by_att": {"ts":[17,46, 70],"fname":"att"},
                    "att_heatmap":{"fname":"heatmap"},
                    "plot_val_ip_op":{"fname":"test_ip_op"},
                    "plot_trajs_ip_op":{"fname":"test_trajs_ip_op"},
                    "plot_actions":{"fname":"test_actions_ip_op"},
                    "plot_train_val_ip_op":{"fname":"plot_train_val_ip_op"},
                    "loss_avg_returns":{"fname":"loss"},
                    "vel_field":{"fname":"vel_field", "ts":[1,60,119]}
                        }
    
    if "DG3" in dataset_name:
        from GPT_paper_plots import paper_plots
        pp = paper_plots(env_4_viz, op_traj_dict_list, src_stats,
                        paper_plot_info=paper_plot_info,
                        save_dir=save_dir)
        pp.plot_traj_by_arr(us_test_traj_set,set_str="_us_test_")

    elif "DOLS" in dataset_name:
        from paper_plots import paper_plots
        pp = paper_plots(env_4_viz, op_traj_dict_list, src_stats,
                        paper_plot_info=paper_plot_info,
                        save_dir=save_dir)
        pp.plot_val_ip_op(us_test_traj_set, test_set_txy_preds, path_lens, success_list)
        pp.plot_trajs_ip_op(us_test_traj_set, test_set_txy_preds, path_lens, success_list)
        pp.plot_velocity_obstacle()
        pp.plot_actions(us_test_traj_set, test_set_txy_preds, path_lens, success_list, actions)

    else:
        raise NotImplementedError    
    

    # pp.plot_train_val_ip_op(us_train_traj_set, us_test_traj_set)
    # pp.plot_traj_by_arr(val_traj_dataset, set_str="_val")
    # pp.plot_att_heatmap(100)
    # pp.plot_traj_by_att("a_a_attention")
    # pp.plot_traj_by_att("a_s_attention")
    
    #  visualize_output(test_set_txy_preds, 
    #                     path_lens,
    #                     iter_i = 0, 
    #                     stats=None, 
    #                     env=env_4_viz, 
    #                     log_wandb=False, 
    #                     plot_policy=False,
    #                     traj_idx=None,      #None=all, list of rzn_ids []
    #                     show_scatter=False,
    #                     at_time=None,
    #                     color_by_time=True, #TODO: fix tdone issue in src_utils
    #                     plot_flow=True,
    #                     wandb_suffix="test_on_unseen",
    #                     model_name=model_name+"_on_"+dataset_name)
    
    # plot_all_attention_mats(all_att_mat_list[0],
    #                         log_wandb=False, 
    #                         model_name=model_name+"_on_"+dataset_name)
    # for t in [i*10 for i in range(1,11)]:
    #     viz_op_traj_with_attention(test_set_txy_preds,
    #                         all_att_mat_list, 
    #                         path_lens,
    #                         mode='dec_sa',       #or 'a_s_attention'
    #                         average_across_layers=True,
    #                         stats=None, 
    #                         env=env_4_viz, 
    #                         log_wandb=False, 
    #                         scale_each_row=True,
    #                         plot_policy=False,
    #                         traj_idx=None,      #None=all, list of rzn_ids []
    #                         show_scatter=False,
    #                         plot_flow=True,
    #                         at_time=t,
    #                         model_name=model_name+"_on_"+dataset_name
    #                         )
    # viz_op_traj_with_attention(test_set_txy_preds,
    #                     all_att_mat_list, 
    #                     path_lens,
    #                     mode='dec_ga',       #or 'a_s_attention'
    #                     average_across_layers=True,
    #                     stats=None, 
    #                     env=env_4_viz, 
    #                     log_wandb=False, 
    #                     scale_each_row=True,
    #                     plot_policy=False,
    #                     traj_idx=None,      #None=all, list of rzn_ids []
    #                     show_scatter=False,
    #                     plot_flow=True,
    #                     at_time=None,
    #                     model_name=model_name+"_on_"+dataset_name
    #                     )   
    # print(f" Results on unseen test")
    # print_dict(results)  
    # # visualize_input(us_test_traj_set, at_time=119, 
    # #                                   env=env_4_viz,
    # #                                   log_wandb=False,
    # #                                   data_name=dataset_name
    # #                                         )

    # print(f" Results on unseen test")
    # print_dict(results)
    # return        





ARGS_QR = False
ARGS_CFG = "_--_"
NAME_MAP = {
            "v5": "DG3",
            "v5_HW": "HW",
            "v5_GPT_DG3": "GPT_DG3",
            "v5_DOLS": "DOLS"
            }


"""
run script with:
To train model:
    For Flow past cylindrical island scenario:
        python main.py --mode train --CFG v5_DOLS
    For Double gyre:
        python main.py --mode train --CFG v5_GPT_DG3

To run inference:
    python main.py --mode inference_on_ckpt --ckpt_path <path_to_model_checkpoint.pt>
"""

if __name__ == "__main__":

    print(f"cuda available: {torch.cuda.is_available()}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--quick_run', type=bool, default=True)
    parser.add_argument('--CFG', type=str, default='v5_DOLS') #v5_GPT_DG3 #v5_DOLS
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help="Path to the checkpoint for inference")
    args = parser.parse_args()

    cfg_name = "cfg/contGrid_" + args.CFG
    sweep_cfg_name = cfg_name + "_sweep"
    cfg_name =  cfg_name + ".yaml"
    sweep_cfg_name =  sweep_cfg_name + ".yaml"
    cfg_name = join(ROOT,cfg_name)
    sweep_cfg_name = join(ROOT,sweep_cfg_name)

    ARGS_QR = args.quick_run # for sweep mode
    ARGS_CFG = args.CFG
    print(f'args.mode = {args.mode}')
    print(f"ARGS_QR={ARGS_QR}")

    if args.mode == 'inference_on_ckpt':
        print("----- beginning inference_on_ckpt ------")
        inference_on_ckpt(args)        

    if args.mode == 'train':
        print("----- beginning train ------")
        train_model(args, cfg_name)

    elif args.mode == 'sweep':
        print("----- beginning sweep ------")
        sweep_cfg = read_cfg_file(cfg_name=sweep_cfg_name)
        sweep_id = wandb.sweep(sweep_cfg)
        wandb.agent(sweep_id, function=train_model)
