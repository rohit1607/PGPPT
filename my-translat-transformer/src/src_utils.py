import torch
import numpy as np
import random
from torch.utils.data import Dataset
import pickle
import wandb
from sklearn.model_selection import train_test_split
from operator import itemgetter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from os.path import join
from root_path import ROOT
import sys
import time

def get_data_split(traj_dataset, split_ratio=[0.6, 0.2, 0.2], random_seed=42, random_split=True):
    """"

    traj_dataset: contains traj infor for all rzns
                    list of dictionaries. dictionary contains states, actions, rtgs
    returns:
    idx_split: test_idx, train_idx, val_idx
    set_split:
    """
    n_trajs =  len(traj_dataset)
    all_idx = [i for i in range(n_trajs)]
    # split all idx to train and (test+val)
    test_val_size = split_ratio[1] + split_ratio[2]
    if random_split:
        train_idx, test_val_idx = train_test_split(all_idx, 
                                            test_size=test_val_size,
                                            random_state=random_seed,
                                            shuffle=True)
        # split (test+val into test and val)
        val_size = split_ratio[2]/test_val_size
        test_idx, val_idx = train_test_split(test_val_idx, test_size=val_size)
        idx_split = (train_idx, test_idx, val_idx)
    else:
        train_idx, test_val_idx = train_test_split(all_idx, 
                                    test_size=test_val_size,
                                    shuffle=False)
        # split (test+val into test and val)
        val_size = split_ratio[2]/test_val_size
        test_idx, val_idx = train_test_split(test_val_idx, test_size=val_size, shuffle=False)
        idx_split = (train_idx, test_idx, val_idx)

    train_traj_set = itemgetter(*train_idx)(traj_dataset)
    test_traj_set = itemgetter(*test_idx)(traj_dataset)
    val_traj_set = itemgetter(*val_idx)(traj_dataset)
    
    set_split = (train_traj_set, test_traj_set, val_traj_set)

    return idx_split, set_split


class create_waypoint_dataset(Dataset):
    def __init__(self, dataset, 
                        idx_set,
                        env,        #env object
                        context_len, 
                        norm_params_4_val=None):
        """
        Computes Dones, Normalizes trajs, masks trajs based on their lengths wrt context_len
        dataset: list of experience dictionaries 
        context_len: context lenght of the transformer
        env_info: (String) env name 

        Note: Written for a particular env field
        """


        self.context_len = context_len
        self.n_trajs = len(dataset)
        self.dataset = [traj for traj in dataset if traj['done']]
        print(f"\n Making dataset out of successful trajectories. \n \
                No. of successful trajs / total trajs = {len(self.dataset)} / {self.n_trajs } \n")
        self.Yi = env.Yi    #Yi are env coeffs
        
        # store waypoints across realizations
        states = [] 
        for traj in self.dataset:
            states.append(traj['states'])
        print(f" states[0].shape:  {states[0].shape}")
        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        # normalize states
        if norm_params_4_val == None:
            for traj in self.dataset:
                traj['states'] = (traj['states'] - self.state_mean) / self.state_std        
        else:
            tr_state_mean,  tr_state_std =  norm_params_4_val
            for traj in self.dataset:
                traj['states'] = (traj['states'] - tr_state_mean) / tr_state_std

    def get_state_stats(self):
        return (self.state_mean, self.state_std)


    # TODO: Verify it returns the no. of trajectories
    def __len__(self):
        return len(self.dataset)
        

    def __getitem__(self, idx):
        traj = self.dataset[idx]
        traj_len = traj['states'].shape[0]
        env_coef_seq = self.Yi[:self.context_len, idx, :] # Yi.shape = (nT, nrzns, nmodes)
        # print(f"****** VERIFY: traj_len = {traj_len}")
        padding_len = None
        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['states'][si : si + self.context_len])
            # NOTE: add extra padde
            states = torch.cat([states, torch.zeros(1,states.shape[1:])], dim=0)
            # actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.zeros(self.context_len, dtype=torch.long).to(torch.bool)
            target_state = torch.from_numpy(traj['target_pos'])
            sys.exit()
        else:
            padding_len = self.context_len - traj_len 

            # padding with zeros
            states = torch.from_numpy(traj['states'])
            # print(f"+++ in cwg: states.shape = {states.shape}")

            # NOTE: paddding_len + 1 for tgt i/o offset for translation tasks
            states = torch.cat([states,
                                torch.zeros(([padding_len+1] + list(states.shape[1:])),
                                dtype=states.dtype)],
                               dim=0)
            # print(f" in cwg: states.shape = {states.shape}")

            # sys.exit()
            # actions = torch.from_numpy(traj['actions'])
            # actions = torch.cat([actions,
            #                     torch.zeros(([padding_len] + list(actions.shape[1:])),
            #                     dtype=actions.dtype)],
            #                    dim=0)

            # returns_to_go = torch.from_numpy(traj['returns_to_go'])
            # returns_to_go = torch.cat([returns_to_go,
            #                     torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
            #                     dtype=returns_to_go.dtype)],
            #                    dim=0)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.zeros(traj_len, dtype=torch.long),
                                   torch.ones(padding_len, dtype=torch.long)],
                                  dim=0).type(torch.bool)
            target_state = torch.from_numpy(traj['target_pos'])

        # print(f"### verify: target_state = {target_state}")
        # sys.exit()
        dummy_t = float(int(0.8*self.context_len)) # time stamp for target state.
        target_state = np.insert(target_state,0,dummy_t,axis=0)
        # print(f"### verify: target_state = {target_state}")
        return  timesteps, states, traj_mask, target_state, env_coef_seq, traj_len


class create_action_dataset(Dataset):
    def __init__(self, dataset, 
                        idx_set,
                        env,        #env object
                        context_len, 
                        norm_params_4_val=None):
        """
        Computes Dones, Normalizes trajs, masks trajs based on their lengths wrt context_len
        dataset: list of experience dictionaries 
        context_len: context lenght of the transformer
        env_info: (String) env name 

        Note: Written for a particular env field
        """

        self.env = env
        self.context_len = context_len
        self.n_trajs = len(dataset)
        self.dataset = [traj for traj in dataset if traj['done']]
        print(f"\n Making dataset out of successful trajectories. \n \
                No. of successful trajs / total trajs = {len(self.dataset)} / {self.n_trajs } \n")
        self.Yi = env.Yi    #Yi are env coeffs
        
        # store actions across realizations
        # TODO: try action normalization

        # actions = []
        # for traj in self.dataset:
        #     actions.append(torch.from_numpy(traj['actions']))
        # print(f" actions[0].shape:  {actions[0].shape}")

        # states = [] 
        # for traj in self.dataset:
        #     states.append(traj['states'])
        # print(f" states[0].shape:  {states[0].shape}")
        # # used for input normalization
        # states = np.concatenate(states, axis=0)
        # self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        # # normalize states
        # if norm_params_4_val == None:
        #     for traj in self.dataset:
        #         traj['states'] = (traj['states'] - self.state_mean) / self.state_std        
        # else:
        #     tr_state_mean,  tr_state_std =  norm_params_4_val
        #     for traj in self.dataset:
        #         traj['states'] = (traj['states'] - tr_state_mean) / tr_state_std

    def get_state_stats(self):
        return (self.state_mean, self.state_std)


    # TODO: Verify it returns the no. of trajectories
    def __len__(self):
        return len(self.dataset)
        

    def __getitem__(self, idx):
        traj = self.dataset[idx]
        traj_len = traj['states'].shape[0]
        env_coef_seq = self.Yi[:self.context_len, idx, :] # Yi.shape = (nT, nrzns, nmodes)
        # print(f"****** VERIFY: traj_len = {traj_len}")
        padding_len = None
        if traj_len >= self.context_len:
            # TODO: correcly write if condition
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            # states = torch.from_numpy(traj['states'][si : si + self.context_len])
            # # NOTE: add extra padde
            # states = torch.cat([states, torch.zeros(1,states.shape[1:])], dim=0)
            try:
                actions = torch.from_numpy(traj['actions'][si : si + self.context_len + 1])
            except:
                actions = torch.cat([actions,
                                    torch.zeros(([1] + list(actions.shape[1:])),
                                    dtype=actions.dtype)],
                                    dim=0)
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # # all ones since no padding
            traj_mask = torch.zeros(self.context_len, dtype=torch.long).to(torch.bool)
            # target_state = torch.from_numpy(traj['target_pos'])
            print(f" entering if condition - should not happen for basic cases")
            print(f"actions.shape = {actions.shape}")
            sys.exit()
        else:
            padding_len = self.context_len - traj_len 

            # padding with zeros
            actions = torch.from_numpy(traj['actions'])
            # print(f"+++ in cwg: states.shape = {states.shape}")

            # NOTE: paddding_len + 1 for tgt i/o offset for translation tasks
            actions = torch.cat([actions,
                                torch.zeros(([padding_len+1] + list(actions.shape[1:])),
                                dtype=actions.dtype)],
                               dim=0)
            # print(f" in cwg: states.shape = {states.shape}")

            # sys.exit()
            # actions = torch.from_numpy(traj['actions'])
            # actions = torch.cat([actions,
            #                     torch.zeros(([padding_len] + list(actions.shape[1:])),
            #                     dtype=actions.dtype)],
            #                    dim=0)

            # returns_to_go = torch.from_numpy(traj['returns_to_go'])
            # returns_to_go = torch.cat([returns_to_go,
            #                     torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
            #                     dtype=returns_to_go.dtype)],
            #                    dim=0)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.zeros(traj_len, dtype=torch.long),
                                   torch.ones(padding_len, dtype=torch.long)],
                                  dim=0).type(torch.bool)
        try:
            target_state = torch.from_numpy(traj['target_pos'])
        except:
            target_state = torch.from_numpy(self.env.target_pos)

        # print(f"### verify: target_state = {target_state}")
        # sys.exit()
        dummy_t = float(int(traj_len)) # time stamp for target state.
        target_state = np.insert(target_state,0,dummy_t,axis=1)
        # print(f"### verify: target_state = {target_state}")
        return  timesteps, actions, traj_mask, target_state, env_coef_seq, traj_len, idx


"""
https://pytorch.org/tutorials/beginner/translation_transformer.html
"""
# mask for sz = 3. It is an ADDITIVE mask
    # [[0., -inf, -inf],
    #         [0., 0., -inf],
    #         [0., 0., 0.]]

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, padding_len, device):
    src_seq_len = src.shape[1] #(BS, context_len, hdim)
    tgt_seq_len = tgt.shape[1]
    assert(src_seq_len==tgt_seq_len)
    batch_size = src.shape[0]
    # print(f" in create mask: src_seq_len.shape = {src_seq_len}")
    # print(f" in create mask: tgt_seq_len.shape = {tgt_seq_len}")

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    # src_padding_mask = (src == src).transpose(0, 1) # (context_len, BS, hdim)
    
    src_padding_mask = torch.zeros((batch_size,src_seq_len),device=device).type(torch.bool)

    # print(f" in create mask: src_padding_mask.shape = {src_padding_mask.shape}")
    # print(f" in create mask: src_padding_mask = {(src_padding_mask.type(torch.int))}")

    tgt_padding_mask = torch.zeros((batch_size,src_seq_len),device=device).type(torch.bool)
    for i in range(batch_size):
        tgt_padding_mask[i,:padding_len[i]] =  True
    # print(f" in create mask: tgt_padding_mask.shape = {tgt_padding_mask.shape}")
    # print(f" in create mask: tgt_padding_mask = {(tgt_padding_mask.type(torch.int))}")
    # print(f" verify sum {torch.sum(tgt_padding_mask, axis=1)}")
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def plot_vel_field(env,t,r=0, g_strmplot_lw=1, g_strmplot_arrowsize=1):
    # Make modes the last axis
    Ui = np.transpose(env.Ui,(0,2,3,1))
    Vi = np.transpose(env.Vi,(0,2,3,1))
    # print(f"**** verify: t = {t}")
    vx_grid = env.U[t,:,:] + np.dot(Ui[t,:,:,:],env.Yi[t,r,:])
    vy_grid = env.V[t,:,:] + np.dot(Vi[t,:,:,:],env.Yi[t,r,:])
    vx_grid = np.flipud(vx_grid)
    vy_grid = np.flipud(vy_grid)
    Xs = np.arange(0,env.xlim) + (env.dxy/2)
    Ys = np.arange(0,env.ylim) + (env.dxy/2)
    X,Y = np.meshgrid(Xs, Ys)
    plt.streamplot(X, Y, vx_grid, vy_grid, color = 'grey', zorder = 0,  linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
    v_mag_grid = (vx_grid**2 + vy_grid**2)**0.5
    plt.contourf(X, Y, v_mag_grid, cmap = "Blues", alpha = 0.5, zorder = -1e5)

def denormalize(txy_norm,tr_stats):
    mean, std = tr_stats
    return (txy_norm*std) + mean

def discrete_frechet_distance(P, Q):
    N = P.shape[0]
    M = Q.shape[0]
    inf = float('inf')
    d = torch.zeros((N, M), dtype=torch.float32)
    d[0, 0] = torch.norm(P[0] - Q[0], 2)
    d[1:, 0] = inf
    d[0, 1:] = inf
        
    for i in range(1, N):
        for j in range(1, M):
            d[i, j] = min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1]) + torch.norm(P[i] - Q[j], 2)
    return d[N - 1, M - 1]

def visualize_output(preds_list, 
                        path_lens,
                        iter_i = 0, 
                        stats=None, 
                        env=None, 
                        log_wandb=True, 
                        plot_policy=False,
                        traj_idx=None,      #None=all, list of rzn_ids []
                        show_scatter=False,
                        at_time=None,
                        color_by_time=True,
                        plot_flow=True,

                        ):
 
    print(f"path_lens = {path_lens}")
    path = join(ROOT, "tmp/last_exp_figs/")
    fname = path + "pred_traj_epoch_" + str(iter_i) + ".png" 
    fig = plt.figure()
    plt.cla()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    # plt.title(f"Policy execution after {iter_i} epochs")

    if stats!=None:
        print("===== Note: rescaling states to original scale for viz=====")

    if traj_idx==None:
        traj_idx = [k for k in range(len(preds_list))]

    # print(f" ***** Verify: traj_idx = {traj_idx}")
    if color_by_time:
        # t_dones = []
        # for preds in preds_list:
        #     t_dones.append(op_traj_dict['t_done'])
        vmin = min(path_lens) if len(path_lens)>0 else 0
        vmax = max(path_lens) if len(path_lens)>0 else 70
        # Make a user-defined colormap.
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('YlOrRd')
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)


    for idx,traj in enumerate(preds_list):
        if idx in traj_idx:
            states = preds_list[idx]
            # print(f"states.shape = {states.shape}\n ")
            # print(f"states = {states}")
            t_done = 50 #TODO: change 
            # print(f"******* Verify: visualize_op: states.shape= {states.shape}")
            if at_time != None:
                assert(at_time >= 1), f"Can only plot at_time >= 1 only"
                # if at_time > t_done, just plot for t_done
                t_done = min(at_time, t_done)
            else:
                at_time = t_done

            if stats!=None:
                mean, std = stats
                print(f"{states.shape}, {mean.shape}")

                states = (states*std) + mean
        
            # Plot sstates
            # shape: (eval_batch_size, max_test_ep_len, state_dim)
            if color_by_time:
                plt.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=scalarMap.to_rgba(t_done), alpha=0.2)
                plt.plot
            else:
                plt.plot(states[0,:t_done+1,1], states[0,:t_done+1,2])

            if show_scatter:
                plt.scatter(states[0,:,1], states[0,:,2],s=1)

            # Plot policy at visites states
            _, nstates,_ = states.shape
            # if plot_policy:
            #     for i in range(nstates):
            #         plt.arrow(states[0,i,1], states[0,i,2], np.cos(actions[0,i,0]), np.sin(actions[0,i,0]))
    if color_by_time:
        cbar = plt.colorbar(scalarMap, label="Arrival Time")
    # TODO: remove hardcode
    plt.xlim([0., 100.])
    plt.ylim([0., 100.])
    # plot target area and set limits
    if env != None:
        plt.xlim([0, env.xlim])
        plt.ylim([0, env.ylim])
        # print("****VERIFY: env.target_pos: ", env.target_pos)
        if env.target_pos.ndim == 1:
            target_circle = plt.Circle(env.target_pos, env.target_rad, color='r', alpha=0.3)
            ax.add_patch(target_circle)
        elif env.target_pos.ndim > 1:
            for target_pos in env.target_pos:
                target_circle = plt.Circle(target_pos, env.target_rad, color='r', alpha=0.3)
                ax.add_patch(target_circle)
        if plot_flow and at_time!=None:
            plot_vel_field(env,at_time-1)
    plt.savefig(fname, dpi=300)

    if log_wandb:
        wandb.log({"pred_traj_fig": wandb.Image(fname)})


    return fig


def visualize_input(traj_dataset, 
                    stats=None, 
                    env=None, 
                    log_wandb=True,
                    traj_idx=None,      #None=all, list of rzn_ids []
                    wandb_fname='input_traj_fig',
                    info_str='',
                    at_time=None,
                    color_by_time=True,
                    plot_flow=True,
                    ):
 
    print(" ---- Visualizing input ---- ")

    path = join(ROOT, "tmp/last_exp_figs/")
    fname = path + "input_traj" + info_str  + ".png"

    fig = plt.figure()
    plt.cla()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    title = "Input_traj_" + info_str
    plt.title(title)

    # print(f" **** Verify: len of first traj = {len(traj_dataset[0][1])}")
    # print(f" **** Verify: first traj = {traj_dataset[0][1]}")
    # print(f" **** Verify: first traj_mask = {traj_dataset[0][4]}")

    if stats!=None:
        print("===== Note: rescaling states to original scale for viz (in visualize_input)=====")

    if traj_idx==None:
        traj_idx = [k for k in range(len(traj_dataset))]
        # TODO: verify wgy 320 out 400 are here
        # print(f" traj_idx= {traj_idx}")
    if color_by_time:
        t_dones = []
        for idx, traj in enumerate(traj_dataset):
            if idx in traj_idx:
                timesteps, states, actions, returns_to_go, traj_mask, _ = traj
                t_done = int(np.sum(traj_mask))   # no. of points to plot. No need to plot masked data.
                # print(f"traj_mask = {traj_mask}")
            t_dones.append(t_done)
        vmin = min(t_dones)
        vmax = max(t_dones)
        # Make a user-defined colormap.
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('YlOrRd')
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)

    for idx, traj in enumerate(traj_dataset):
        if idx in traj_idx:
            timesteps, states, actions, returns_to_go, traj_mask, _ = traj
            t_done = int(np.sum(traj_mask))   # no. of points to plot. No need to plot masked data.
            t_done = 119
            # if at_time != None:
            #     assert(at_time >= 1), f"Can only plot at_time >= 1 only"
            #     # if at_time > t_done, just plot for t_done
            #     t_done = min(at_time, t_done)
            # else:
            #     at_time = t_done
           
           
            if stats != None:
                mean, std = stats
                states = (states*std) + mean
                # states = states*(traj_mask.reshape(-1,1))

            if color_by_time:
                plt.plot(states[:t_done,1], states[:t_done,2], color=scalarMap.to_rgba(t_done) )
            else:
                plt.plot(states[:t_done,1], states[:t_done,2], linewidth=0.1)
            plt.scatter(states[-1,1], states[-1,2], alpha=0.5, zorder=10000, s=5)
            # print(f"bcords: {states[-1,1], states[-1,2]}")
    plt.xlim([0, 100.])
    plt.ylim([0, 100.])

    if env != None:
        plt.xlim([0, env.xlim])
        plt.ylim([0, env.ylim])
        print("****VERIFY: env.target_pos: ", env.target_pos)
        print(f"**** verify: {len(env.target_pos)}")
        if env.target_pos.ndim == 1:
            target_circle = plt.Circle(env.target_pos, env.target_rad, color='r', alpha=0.3)
            ax.add_patch(target_circle)
        elif env.target_pos.ndim > 1:
            for target_pos in env.target_pos:
                target_circle = plt.Circle(target_pos, env.target_rad, color='r', alpha=0.3)
                ax.add_patch(target_circle)


        if plot_flow and at_time!=None:
            plot_vel_field(env,at_time-1)    
    plt.savefig(fname, dpi=300)

    if log_wandb:
        wandb.log({wandb_fname: wandb.Image(fname)})
    plt.cla()

