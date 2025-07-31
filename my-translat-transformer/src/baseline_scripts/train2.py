import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from timeit import default_timer as timer
from src_utils import create_waypoint_dataset, create_action_dataset, get_data_split, create_mask, denormalize, visualize_output, visualize_input
from utils import read_cfg_file, save_yaml
# from std_models import Seq2SeqTransformer
from custom_models import mySeq2SeqTransformer_v1
from custom_models import TransRNN

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

def rnn_train_epoch(model, optimizer, tr_set, cfg):
    model.train()
    losses = 0
    train_dataloader = DataLoader(tr_set, batch_size=cfg.batch_size)
    for timesteps, states, traj_mask, target_state, env_coef_seq, traj_len in train_dataloader:
        src = env_coef_seq.to(cfg.device)
        tgt = states.to(cfg.device)
        tgt_input = tgt[:, :-1, :]
        logits = model(src, tgt_input)
        optimizer.zero_grad()
        tgt_out = tgt[:,1:,:]
        loss = F.mse_loss(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1, tgt_out.shape[-1]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        losses += loss.item()
        wandb.log({"loss_vs_batch": loss,})
    
    return losses / len(train_dataloader)

def rnn_evaluate(model, val_set, cfg):
    model.eval()
    losses = 0
    val_dataloader = DataLoader(val_set, batch_size=cfg.batch_size)

    for timesteps, states, traj_mask, target_state, env_coef_seq, traj_len in val_dataloader:
        src = env_coef_seq.to(cfg.device)
        tgt = states.to(cfg.device)
        tgt_input = tgt[:, :-1, :]
        logits = model(src, tgt_input)
        tgt_out = tgt[:, 1:, :]
        loss = F.mse_loss(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1,tgt_out.shape[-1]))
        losses += loss.item()

    return losses / len(val_dataloader)

def rnn_translate(model, test_set, tr_set_stats, cfg):
    test_dataloader = DataLoader(test_batch_size=cfg.batch_size)
    final_pred_list = []
    final_path_lens = []
    for timesteps, states, traj_mask, target_state, env_coef_seq, traj_len in test_dataloader:
        src = env_coef_seq.to(cfg.device)
        tgt = states.to(cfg.device)
        tgt_input = tgt[:, :-1, :]
        final_pred, path_length = model.translate(src,tgt_input, tr_set_stats, target_state)
        final_pred = final_pred.reshape(1, final_pred.shape[-1])
        pred_list = [final_pred[i,:] for i in range(path_length)]
        final_pred_list.append(pred_list)
        final_path_lens.append(path_length)

    return final_pred_list, final_path_lens

def train_epoch(model, optimizer, tr_set, cfg, args):
    model.train()
    losses = 0
    train_dataloader = DataLoader(tr_set, batch_size=cfg.batch_size)

    # for env_coef_seq, tgt in train_dataloader:
    for timesteps, tgt, traj_mask, target_state, env_coef_seq, traj_len, idx in train_dataloader:
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


        tgt_out = tgt[:, 1:, :]

        # only consider non padded elements
        logits =  logits.view(-1,1)[(~tgt_padding_mask).view(-1,)]
        tgt_out = tgt_out.reshape(-1,1)[(~tgt_padding_mask).view(-1,)]

        # loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        loss = F.mse_loss(logits, tgt_out)
        # print("model predictions:", logits.reshape(-1, logits.shape[-1]),) 
        # print("tgt_out:", tgt_out.reshape(-1,tgt_out.shape[-1]))
        optimizer.zero_grad()

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

    for timesteps, tgt, traj_mask, target_state, env_coef_seq, traj_len, idx in val_dataloader:
        timesteps = timesteps.to(cfg.device)
        src = env_coef_seq.to(cfg.device)
        tgt = tgt.to(cfg.device)
        tgt_input = tgt[:, :-1, :]
        tgt_padding_mask = traj_mask.to(cfg.device)
        src_mask, tgt_mask, src_padding_mask, _ = create_mask(src, tgt_input, traj_len, cfg.device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask,timesteps)
        
        tgt_out = tgt[:, 1:, :]
        logits =  logits.view(-1,1)[(~tgt_padding_mask).view(-1,)]
        tgt_out = tgt_out.reshape(-1,1)[(~tgt_padding_mask).view(-1,)]
        # loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss = F.mse_loss(logits, tgt_out)
        
        losses += loss.item()

    return losses / len(val_dataloader)


def translate(model: torch.nn.Module, test_idx, test_set, tr_set_stats, cfg, env, earlybreak=10**8):
    model.eval()
    test_dataloader = DataLoader(test_set, batch_size=1)
    count = 0           # keeps count of total episodes
    success_count = 0   # keeps count of successful episodes
    op_traj_dict_list = []
    with torch.no_grad():
        # for sample in range(len(test_set)):
        # timesteps, tgt, traj_mask, target_state, env_coef_seq, traj_len, idx = test_set[sample]
        for timesteps, tgt, traj_mask, target_state, env_coef_seq, traj_len, idx in test_dataloader:
            op_traj_dict = {}
            reached_target = False
            env.reset()
            idx = idx[0].item() #initially idx = tensor([0])
            rzn = test_idx[idx]
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
            txy_preds = np.zeros((1, cfg.context_len, 3),dtype=np.float32,)
            # print(f" preds.shape = {preds.shape}, tgt.shape = {tgt.shape}")
            txy_preds[0,0,:] = np.array([0,env.start_pos[0],env.start_pos[1]])
            preds[0,0,:] = tgt[0,0,:]

            for i in range(cfg.context_len-1):
                memory = memory.to(cfg.device)
                out = model.decode(preds, memory, tgt_mask, timesteps)
                gen = model.generator(out)
                preds[0,i+1,:] = gen[0,i,:].detach()
                a = preds[0,i+1,:].cpu().numpy().copy()
                txy, reward ,done, info = env.step(a)
                txy_preds[0,i+1,:] = txy 
                # TODO: ***IMP*****: reduce GPU-CPU communication
                if done:
                    if reward > 0:
                        reached_target = True
                        success_count += 1
                    break

            # loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            mse = F.mse_loss(preds[0,:i].cpu(),tgt[0,:i].cpu())
            # print(f"rough mse for sample {count} = {mse}")

            op_traj_dict['states'] = np.array(txy_preds)
            op_traj_dict['actions'] = preds.cpu()
            op_traj_dict['t_done'] = i+1
            op_traj_dict['n_tsteps'] = i
            # op_traj_dict['attention_weights'] = attention_weights
            op_traj_dict['success'] = reached_target
            op_traj_dict['mse'] = mse
            op_traj_dict_list.append(op_traj_dict)

    results = {}
    results['avg_val_loss'] = np.mean([d['mse'] for d in op_traj_dict_list])
    results['translate/avg_ep_len'] = np.mean([d['n_tsteps'] for d in op_traj_dict_list])
    results['translate/success_ratio'] = success_count/count
    results['runs_from_set(count)'] = count
    return op_traj_dict_list, results



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
    
    _, dummy_target, _, _, dummy_env_coef_seq, _,_ = tr_set[0]
    src_vec_dim = dummy_env_coef_seq.shape[-1]
    tgt_vec_dim = dummy_target.shape[-1]
    print(f"src_vec_dim = {src_vec_dim} \n tgt_vec_dim = {tgt_vec_dim}")

    transformer = mySeq2SeqTransformer_v1(num_encoder_layers, num_decoder_layers, embed_dim,
                                 n_heads, src_vec_dim, tgt_vec_dim, 
                                 dim_feedforward=None,     # TODO: add dim_ffn to cfg
                                 max_len=context_len).to(cfg.device)

    rnn = TransRNN(src_vec_dim, 2*src_vec_dim, tgt_vec_dim, device, context_len).to(cfg.device)
   
    # optimizer = torch.optim.Adam(transformer.parameters(), lr=0.00001, betas=(0.9, 0.98), eps=1e-9)
    optimizer = torch.optim.AdamW(
                    transformer.parameters(),
                    lr=lr,
                    weight_decay=wt_decay
                )

    pytorch_trainable_params = sum(p.numel() for p in rnn.parameters() if p.requires_grad)
    pytorch_total_params = sum(p.numel() for p in rnn.parameters())
    print(f"total params = {pytorch_total_params}")
    print(f"trainable params = {pytorch_trainable_params}")
    wandb.run.summary["total params"] = pytorch_total_params
    wandb.run.summary["trainable params"] = pytorch_trainable_params

    min_ETA = 10**5
    for epoch in range(0, num_epochs+1):
        print(f"epoch {epoch}")
        epoch_start_time = timer()
        train_loss = rnn_train_epoch(rnn, optimizer, tr_set, cfg)
        epoch_end_time = timer()
        val_loss = rnn_evaluate(rnn, val_set, cfg)
        # if epoch % eval_inerval == 0:
        #     tr_op_traj_dict_list, tr_results = translate(transformer, train_idx_set, tr_set, None, 
        #                                            cfg, env, earlybreak=50)
        #     tr_set_txy_preds = [d['states'] for d in tr_op_traj_dict_list]
        #     path_lens = [d['n_tsteps'] for d in tr_op_traj_dict_list]
        #     visualize_output(tr_set_txy_preds, 
        #                         path_lens,
        #                         iter_i = 0, 
        #                         stats=None, 
        #                         env=env, 
        #                         log_wandb=True, 
        #                         plot_policy=False,
        #                         traj_idx=None,      #None=all, list of rzn_ids []
        #                         show_scatter=True,
        #                         at_time=None,
        #                         color_by_time=True, #TODO: fix tdone issue in src_utils
        #                         plot_flow=True,
        #                         wandb_suffix="train")
            
        #     val_op_traj_dict_list, val_results = translate(transformer, val_idx_set, val_set, None, 
        #                                                    cfg, env, earlybreak=200)
        #     val_set_txy_preds = [d['states'] for d in val_op_traj_dict_list]
        #     path_lens = [d['n_tsteps'] for d in val_op_traj_dict_list]
        #     visualize_output(val_set_txy_preds, 
        #                         path_lens,
        #                         iter_i = 0, 
        #                         stats=None, 
        #                         env=env, 
        #                         log_wandb=True, 
        #                         plot_policy=False,
        #                         traj_idx=None,      #None=all, list of rzn_ids []
        #                         show_scatter=True,
        #                         at_time=None,
        #                         color_by_time=True, #TODO: fix tdone issue in src_utils
        #                         plot_flow=True,
        #                         wandb_suffix="val")


        # translate_avg_ep_len = val_results['translate/avg_ep_len']
        # translate_avg_val_loss = val_results['avg_val_loss']
        # success_ratio = val_results['translate/success_ratio']
        # count = val_results['runs_from_set(count)'] 
        log_dict = { "tr_loss_vs_epoch (unpadded elems)": train_loss,
            "val_loss_vs_epoch (unpadded elems)": val_loss,
            # "avg_val_loss (across pred len)": translate_avg_val_loss,
            # "success_ratio": success_ratio,
            # 'runs_from_set(count)': count,
            # "ETA": translate_avg_ep_len,
            # "lr" : scheduler.get_last_lr()[0] if use_scheduler else lr
            }
        wandb.log(log_dict)

        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)
        print('='*60)
        print(f"Epoch: {epoch}")
        for key, val in log_dict.items():
            print(f"{key}: {format(val,'.4f')}")
        print(f"Epoch runtime = {(epoch_end_time - epoch_start_time):.3f}s")
        print(f"time elapsed: {time_elapsed}")
        print("")
 
        
        # TODO: save model and best metrics after completing evaluation
        # if translate_avg_ep_len < min_ETA:
        #     min_ETA = translate_avg_ep_len
        #     print("saving current model at: " + save_model_path)

        #     best_avg_episode_length = translate_avg_ep_len
        #     best_success_ratio = success_ratio
        #     best_epoch = epoch
        #     # "avg_val_loss"= eval_avg_val_loss

        #     torch.save(transformer, save_model_path)
        #     tmp_path = save_model_path[:-1]
        #     torch.save(transformer, tmp_path)
        

    sys.exit()  ######## Shubham
    # save_model_path = join(log_dir, wandb_exp_name)
    # torch.save(transformer, save_model_path)
    cfg_copy_path = save_model_path[:-2] + "yml"
    save_yaml(cfg_copy_path,cfg_copy)
    print(f"cfg_copy_path = {cfg_copy_path}")
    wandb.run.summary["best_avg_episode_length"] = best_avg_episode_length
    wandb.run.summary["best_success_ratio"] = best_success_ratio
    wandb.run.summary["total_runs_val_set"] = len(val_op_traj_dict_list)
    wandb.run.summary["best_epoch"] = best_epoch

    print("=" * 60)
    print("finished training!")
    print("=" * 60)

    print(f"\n\n ---- running inference on test set ----- \n\n")
    transformer = torch.load(tmp_path)
    op_traj_dict_list, results  = translate(transformer,test_idx_set, test_set, None, cfg, env)
    test_set_txy_preds = [d['states'] for d in op_traj_dict_list]
    path_lens = [d['n_tsteps'] for d in op_traj_dict_list]  
    visualize_output(test_set_txy_preds, 
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
                        plot_flow=True,
                        wandb_suffix="test")
    
    end_time = datetime.now().replace(microsecond=0)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("saved last updated model at: " + save_model_path)
    print("=" * 60)


ARGS_QR = False
if __name__ == "__main__":

    print(f"cuda available: {torch.cuda.is_available()}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='single_run')
    parser.add_argument('--quick_run', type=bool, default=False)
    parser.add_argument('--env', type=str, default='v5_DOLS')
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
