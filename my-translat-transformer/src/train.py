import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from timeit import default_timer as timer
from src_utils import create_waypoint_dataset, create_action_dataset, get_data_split, create_mask, denormalize, visualize_output, visualize_input
from utils import read_cfg_file
# from std_models import Seq2SeqTransformer
from custom_models import mySeq2SeqTransformer_v1
# from custom_models import LSTMModel

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
import sys

wandb.login()


def train_epoch(model, optimizer, tr_set, cfg, args):
    model.train()
    losses = 0
    train_dataloader = DataLoader(tr_set, batch_size=cfg.batch_size)

    # for env_coef_seq, tgt in train_dataloader:
    for timesteps, tgt, traj_mask, target_state, env_coef_seq, traj_len in train_dataloader:
        # print(f"### verify env_coef_seq: {env_coef_seq.shape}, \ntgt: {tgt.shape}, {traj_mask.shape}")
        timesteps = timesteps.to(cfg.device)
        src = env_coef_seq.to(cfg.device)
        tgt = tgt.to(cfg.device)
        # print(f"verify src = {src[0::2,0:10,:]} \n")
        # print(f"verify tgt = {tgt[0::2,0:10,:]} \n")
        tgt_input = tgt[:, :-1, :]
        tgt_padding_mask = traj_mask.to(cfg.device)
        src_mask, tgt_mask, src_padding_mask, _ = create_mask(src, tgt_input, traj_len, cfg.device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask, timesteps)
        # print(f" logits.shape = {logits.shape}")
        # print(f" logits,isnan = {torch.isnan(logits)}")

        optimizer.zero_grad()

        tgt_out = tgt[:, 1:, :]
        # loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss = F.mse_loss(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1,tgt_out.shape[-1]))
        # print("model predictions:", logits.reshape(-1, logits.shape[-1]),)
        # print("tgt_out:", tgt_out.reshape(-1,tgt_out.shape[-1]))
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

        optimizer.step()
        losses += loss.item()
        # wandb.log({"loss_vs_batch": loss,
        #             })
    return losses / len(train_dataloader)


def evaluate(model, val_set, cfg):
    model.eval()
    losses = 0

    val_dataloader = DataLoader(val_set, batch_size=cfg.batch_size)

    for timesteps, tgt, traj_mask, target_state, env_coef_seq, traj_len in val_dataloader:
        timesteps = timesteps.to(cfg.device)
        src = env_coef_seq.to(cfg.device)
        tgt = tgt.to(cfg.device)
        tgt_input = tgt[:, :-1, :]
        tgt_padding_mask = traj_mask.to(cfg.device)
        src_mask, tgt_mask, src_padding_mask, _ = create_mask(src, tgt_input, traj_len, cfg.device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask,timesteps)
        
        tgt_out = tgt[:, 1:, :]
        loss = F.mse_loss(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1,tgt_out.shape[-1]))
        losses += loss.item()

    return losses / len(val_dataloader)


def translate(model: torch.nn.Module, test_set, tr_set_stats, cfg, env):
    model.eval()
    test_dataloader = DataLoader(test_set, batch_size=1)
    test_set_a_preds = []
    test_set_txy_preds = []
    path_lens = []
    test_idx = 0
    for timesteps, tgt, traj_mask, target_state, env_coef_seq, traj_len in test_dataloader:
        env.reset()
        timesteps = timesteps.to(cfg.device)
        # print(f"env.state = {env.state}")
        test_idx += 1
        if test_idx ==50:
            break
        src = env_coef_seq.to(cfg.device)
        dummy_tgt_for_mask = tgt.to(cfg.device)[:, :-1, :]
        src_mask, tgt_mask, src_padding_mask, _ = create_mask(src, dummy_tgt_for_mask, traj_len, cfg.device)
        # memory is the encoder output
        memory =  model.encode(src, src_mask, timesteps)
        preds = torch.zeros((1, cfg.context_len, dummy_tgt_for_mask.shape[2]),dtype=torch.float32, device=cfg.device)
        txy_preds = np.zeros((1, cfg.context_len, 3),dtype=np.float32,)
        # print(f" preds.shape = {preds.shape}, tgt.shape = {tgt.shape}")
        preds[0,0,:] = tgt[0,0]

        for i in range(cfg.context_len-1):
            memory = memory.to(cfg.device)
            out = model.decode(preds, memory, tgt_mask, timesteps)
            # print(f"out.shape = {out.shape}")
            gen = model.generator(out)
            # print(f"gen[0,{i}:{i+5},:] = {gen[0,i:i+5,:]}")
            preds[0,i+1,:] = gen[0,i,:].detach()
            a = preds[0,i+1,:].cpu().numpy().copy()
            # print(f"a = {a}")
            txy, reward ,done, info = env.step(a)
            txy_preds[0,i+1,:] = txy 
            # TODO: ***IMP*****: reduce GPU-CPU communication
            # print(f"-------- {txy}, {target_state}, {tr_set_stats}")
            # print(f"**** check : norm: {np.linalg.norm((txy[1:]-target_state[0,1:].numpy())) }")
            if done:
                # print(f"when done env.state = {env.state}")

                # print(f" ########### DONE ################")
                path_lens.append(i)
                break

        if test_idx in [2, 4, 12, 14, 22, 24]:
            # print(f"preds = {preds[0]} ")
            # print(f"tgy = {tgt} ")
            mse = F.mse_loss(preds[0].cpu(),tgt[0,0].cpu())
            print(f"mse = {mse}")
        test_set_a_preds.append(preds.cpu())
        test_set_txy_preds.append(np.array(txy_preds))
    return test_set_a_preds,test_set_txy_preds, path_lens


            




def train_model(args, cfg_name):

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%m-%d-%H-%M")

    cfg = read_cfg_file(cfg_name=cfg_name)
    cfg_copy = cfg.copy()
    dataset_name = cfg['dataset_name']
    wandb_exp_name = "env2wp_" + dataset_name + "__" + start_time_str
    wandb.init(project="my_translation_transformer",
        name = wandb_exp_name,
        config=cfg
        )
    cfg=wandb.config
    params2 = read_cfg_file(cfg_name=join(ROOT,cfg.params2_name))

    add_trans_noise = cfg.add_transition_noise_during_inf

    env_name = cfg.env_name
    split_tr_tst_val = cfg.split_tr_tst_val
    split_ran_seed = cfg.split_ran_seed
    random_split = cfg.random_split
    max_eval_ep_len = cfg.max_eval_ep_len  # max len of one episode
    num_eval_ep = cfg.num_eval_ep       # num of evaluation episodes

    batch_size = cfg.batch_size           # training batch size
    use_scheduler = cfg.use_scheduler
    lr = cfg.lr                            # const learning rate
    wt_decay = cfg.wt_decay               # weight decay
    warmup_steps = cfg.warmup_steps       # warmup steps for lr scheduler

    # total_updates = max_train_iters x num_updates_per_iter
    num_epochs = cfg.num_epochs
    # num_updates_per_iter = cfg.num_updates_per_iter
    # comp_val_loss = cfg.comp_val_loss

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

    # training and evaluation device
    device = torch.device(cfg.device)
    
    at_pl_t = 50
    if args.quick_run:
        print("\n ---------- Modifying cfg params for quick run --------------- \n")
        num_epochs = 2
        # num_updates_per_iter = 10
        num_eval_ep = 10
        at_pl_t = 4


    prefix = "my_translat_" + dataset_name

    save_model_name =  prefix + "_model_" + start_time_str + ".pt"
    save_model_path = join(log_dir, save_model_name)
    save_best_model_path = save_model_path[:-3] + "_best.pt"


    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    print("device set to: " + str(device))
    print("dataset path: " + dataset_path)
    print("model save path: " + save_model_path)

    env = gym.make(env_name)
    env.setup(cfg, params2, add_trans_noise=add_trans_noise)

    # Load and Split dataset
    with open(dataset_path, 'rb') as f:
        traj_dataset = pickle.load(f)

    idx_split, set_split = get_data_split(traj_dataset,
                                        split_ratio=split_tr_tst_val, 
                                        random_seed=split_ran_seed, 
                                        random_split=random_split)
    train_traj_set, test_traj_set, val_traj_set = set_split
    train_idx_set, test_idx_set, val_idx_set = idx_split


    # # dataset contains optimal waypoints for different realizations of the env
    # tr_set = create_waypoint_dataset(train_traj_set, 
    #                         train_idx_set,
    #                         env, 
    #                         context_len, 
    #                         norm_params_4_val=None)
    # tr_set_stats = tr_set.state_mean, tr_set.state_std

    # val_set = create_waypoint_dataset(val_traj_set, 
    #                         val_idx_set,
    #                         env, 
    #                         context_len, 
    #                         norm_params_4_val=tr_set_stats)
    # test_set = create_waypoint_dataset(test_traj_set, 
    #                         val_idx_set,
    #                         env, 
    #                         context_len, 
    #                         norm_params_4_val=tr_set_stats)


    # dataset contains optimal actions for different realizations of the env
    tr_set = create_action_dataset(train_traj_set, 
                            train_idx_set,
                            env, 
                            context_len, 
                                        )

    # tr_set_stats = tr_set.state_mean, tr_set.state_std

    val_set = create_action_dataset(val_traj_set, 
                            val_idx_set,
                            env, 
                            context_len, 
                                        )

    test_set = create_action_dataset(test_traj_set, 
                            val_idx_set,
                            env, 
                            context_len, 
                                        )


    # TODO: Take it outside the function and try
    train_dataloader = DataLoader(tr_set, batch_size=batch_size)
    # visualize_input(tr_set, stats=tr_set_stats, log_wandb=True, at_time=50)
    # visualize_input(val_set, stats=None, log_wandb=True, at_time=119, info_str='val', color_by_time=False)
    # visualize_input(test_set, stats=None, log_wandb=True, at_time=119, info_str='test', color_by_time=False)


    _, dummy_target, _, _, dummy_env_coef_seq, _ = tr_set[0]
    src_vec_dim = dummy_env_coef_seq.shape[-1]
    tgt_vec_dim = dummy_target.shape[-1]
    print(f"src_vec_dim = {src_vec_dim} \n tgt_vec_dim = {tgt_vec_dim}")

    transformer = mySeq2SeqTransformer_v1(num_encoder_layers, num_decoder_layers, embed_dim,
                                 n_heads, src_vec_dim, tgt_vec_dim, 
                                 dim_feedforward=None,     # TODO: add dim_ffn to cfg
                                 max_len=context_len).to(cfg.device)
   
    # optimizer = torch.optim.Adam(transformer.parameters(), lr=0.00001, betas=(0.9, 0.98), eps=1e-9)
    optimizer = torch.optim.AdamW(
                    transformer.parameters(),
                    lr=lr,
                    weight_decay=wt_decay
                )

    pytorch_trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    pytorch_total_params = sum(p.numel() for p in transformer.parameters())
    print(f"total params = {pytorch_total_params}")
    print(f"trainable params = {pytorch_trainable_params}")
    wandb.run.summary["total params"] = pytorch_total_params
    wandb.run.summary["trainable params"] = pytorch_trainable_params

    for epoch in range(1, num_epochs+1):
        print(f"epoch {epoch}")
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, tr_set, cfg, args)
        end_time = timer()
        val_loss = evaluate(transformer, val_set, cfg)
        if epoch%500 == 0:
            tr_set_a_preds, tr_set_txy_preds, path_lens = translate(transformer, tr_set, None, cfg, env)
            visualize_output(tr_set_txy_preds, 
                                path_lens,
                                iter_i = 0, 
                                stats=None, 
                                env=env, 
                                log_wandb=True, 
                                plot_policy=False,
                                traj_idx=None,      #None=all, list of rzn_ids []
                                show_scatter=True,
                                at_time=None,
                                color_by_time=True, #TODO: fix tdone issue in src_utils
                                plot_flow=True,)
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}") 
        print(f"Val loss: {val_loss:.3f}, ")     
        print(f"Epoch time = {(end_time - start_time):.3f}s")
        wandb.log({"tr_loss_vs_epoch": train_loss,
                    "val_loss_vs_epoch": val_loss
                    })


    save_model_path = join(log_dir, wandb_exp_name)
    torch.save(transformer, save_model_path)


    # test_set_preds, path_lens = translate(transformer, test_set, tr_set_stats , cfg)
    tr_set_a_preds, tr_set_txy_preds, path_lens = translate(transformer, tr_set, None, cfg, env)
    visualize_output(tr_set_txy_preds, 
                        path_lens,
                        iter_i = 0, 
                        stats=None, 
                        env=env, 
                        log_wandb=True, 
                        plot_policy=False,
                        traj_idx=None,      #None=all, list of rzn_ids []
                        show_scatter=True,
                        at_time=None,
                        color_by_time=True, #TODO: fix tdone issue in src_utils
                        plot_flow=True,)
    
    # tr_set_a_preds, tr_set_txy_preds, path_lens = translate(transformer, tr_set, None, cfg, env)
    # visualize_output(tr_set_txy_preds, 
    #                     path_lens,
    #                     iter_i = 0, 
    #                     stats=None, 
    #                     env=env, 
    #                     log_wandb=True, 
    #                     plot_policy=False,
    #                     traj_idx=None,      #None=all, list of rzn_ids []
    #                     show_scatter=True,
    #                     at_time=None,
    #                     color_by_time=True, #TODO: fix tdone issue in src_utils
    #                     plot_flow=True,)


ARGS_QR = False
if __name__ == "__main__":

    print(f"cuda available: {torch.cuda.is_available()}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='single_run')
    parser.add_argument('--quick_run', type=bool, default=False)
    parser.add_argument('--env', type=str, default='v5_HW')
    args = parser.parse_args()

    cfg_name = "cfg/contGrid_" + args.env + ".yaml"
    cfg_name = join(ROOT,cfg_name) 
    # sweeep_cfg_name = join(ROOT,"cfg/contGrid_v6_sweep.yaml")
    # print(f'args.mode = {args.mode}')

    if args.mode == 'single_run':
        print("----- beginning single_run ------")
        train_model(args, cfg_name)


    # elif args.mode == 'sweep':
    #     print("----- beginning sweeeeeeeeeeeeep ------")
    #     ARGS_QR = args.quick_run
    #     print(f"ARGS_QR={ARGS_QR}")
    #     sweep_cfg = read_cfg_file(cfg_name=sweeep_cfg_name)
    #     sweep_id = wandb.sweep(sweep_cfg)
    #     wandb.agent(sweep_id, function=sweep_train)
