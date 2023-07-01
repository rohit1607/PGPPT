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


            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.zeros(traj_len, dtype=torch.long),
                                   torch.ones(padding_len, dtype=torch.long)],
                                  dim=0).type(torch.bool)
            target_state = torch.from_numpy(traj['target_pos'])

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
        if len(target_state.shape) == 2:
            target_state = np.insert(target_state,0,dummy_t,axis=1)
        elif len(target_state.shape) == 1:
            target_state = np.insert(target_state,0,dummy_t,axis=0)

        # print(f"### verify: target_state = {target_state}")
        return  timesteps, actions, traj_mask, target_state, env_coef_seq, traj_len, idx


class create_action_dataset_v2(Dataset):
    def __init__(self, dataset, 
                        idx_set,
                        context_len, 
                        norm_params_4_val=None):
        """
        Different from v1: 
            - deals with a combined dataset made from multiple
              datasets with varying obstacle configs
            - dataset is cleaned.

        dataset: [(Yi_r, obs_r, actions_r, states_r,
                     timesteps_r, dones_r, success_r, 
                     target_pos_r, start_pos_r, flow_dir, rzn), (..), ...]

        Computes Dones, Normalizes trajs, masks trajs based on their lengths wrt context_len
        dataset: list of experience dictionaries 
        context_len: context lenght of the transformer
        env_info: (String) env name 

        Note: Written for a particular env field
        """

        self.context_len = context_len
        self.n_trajs = len(dataset)
        self.dataset = dataset
        # self.X = np.array([np.concatenate((item[0], item[1]), axis=-1) for item in self.dataset])
        self.X = np.array([item[0] for item in self.dataset])
        self.X_mean = np.mean(self.X, axis=0)
        self.X_std = np.std(self.X, axis=0)

        # extract actions (tgt) and scale them to range [0,1)
        self.Y = [item[2]/(2*np.pi) for item in self.dataset]

        # normalzise
        if norm_params_4_val == None:
            for i in range(len(self.X)):
                self.X[i] = self.X[i] - self.X_mean
                self.X[i] = np.divide(self.X[i], self.X_std)
                # self.Y[i] = self.Y[i] - self.Y_mean
                # self.Y[i] = np.divide(self.Y[i], self.Y_std)
        else:
            tr_X_mean, tr_X_std= norm_params_4_val
            for i in range(len(self.X)):
                self.X[i] = self.X[i] - tr_X_mean
                self.X[i] = np.divide(self.X[i], tr_X_std)            
                # self.Y[i] = self.Y[i] - tr_Y_mean
                # self.Y[i] = np.divide(self.Y[i], tr_Y_std)
        # store actions across realizations
        # TODO: try action normalization




    def get_src_stats(self):
        return (self.X_mean, self.X_std)


    # TODO: Verify it returns the no. of trajectories
    def __len__(self):
        return len(self.dataset)
        

    def __getitem__(self, idx):
        _, _, _, _, _, _, success, target_pos, _, flow_dir, rzn = self.dataset[idx]
        actions = self.Y[idx]
        traj_len = len(actions)
        env_coef_seq = self.X[idx, :self.context_len, :] # X.shape = (B(r), ETA, coefs+obs_tok)
        padding_len = None
        if traj_len > self.context_len:
            # TODO: correcly write if condition
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            # states = torch.from_numpy(traj['states'][si : si + self.context_len])
            # # NOTE: add extra padde
            # states = torch.cat([states, torch.zeros(1,states.shape[1:])], dim=0)
            try:
                actions = torch.from_numpy(actions[si : si + self.context_len + 1])
            except:
                actions = torch.cat([actions,
                                    torch.zeros(([1] + list(actions.shape[1:])),
                                    dtype=actions.dtype)],
                                    dim=0)
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # # all ones since no padding
            traj_mask = torch.zeros(self.context_len, dtype=torch.long).to(torch.bool)
            print(f" entering if condition - should not happen for basic cases")
            print(f"actions.shape = {actions.shape}")
            print(f"traj_len = {traj_len}")
            sys.exit()
        else:
            padding_len = self.context_len - traj_len 

            # padding with zeros
            actions = torch.from_numpy(actions)

            # NOTE: paddding_len + 1 for tgt i/o offset for translation tasks
            actions = torch.cat([actions, # shape (nactions, 1)
                                # [padding_len+1] is seq_len axis, list(actions.shape[1:] is for everythin apart from seq_len axis
                                torch.zeros(([padding_len+1] + list(actions.shape[1:])), #[4]+[1]=[4,1]
                                dtype=actions.dtype)],
                               dim=0)


            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.zeros(traj_len, dtype=torch.long),
                                   torch.ones(padding_len, dtype=torch.long)],
                                  dim=0).type(torch.bool)
        
        target_state = torch.tensor(target_pos)


        return  timesteps, actions, traj_mask, target_state, env_coef_seq, traj_len, idx, flow_dir, rzn


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
    # assert(src_seq_len==tgt_seq_len)
    batch_size = src.shape[0]
    # print(f" in create mask: src_seq_len.shape = {src_seq_len}")
    # print(f" in create mask: tgt_seq_len.shape = {tgt_seq_len}")

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    # src_padding_mask = (src == src).transpose(0, 1) # (context_len, BS, hdim)
    
    src_padding_mask = torch.zeros((batch_size,src_seq_len),device=device).type(torch.bool)

    tgt_padding_mask = torch.zeros((batch_size,src_seq_len),device=device).type(torch.bool)
    for i in range(batch_size):
        tgt_padding_mask[i,:padding_len[i]] =  True
    # print(f" in create mask: tgt_padding_mask.shape = {tgt_padding_mask.shape}")
    # print(f" in create mask: tgt_padding_mask = {(tgt_padding_mask.type(torch.int))}")
    # print(f" verify sum {torch.sum(tgt_padding_mask, axis=1)}")
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def plot_vel_field(env,t,r=0, g_strmplot_lw=1, g_strmplot_arrowsize=1):
    # Make modes the last axis T m H W
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
    # dummmy

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

def DOLS_obstacle():
    s=15
    return plt.Circle([4.5*s, 3*s], 0.5*s, color='k', alpha=0.3)
    
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
                        wandb_suffix="",
                        model_name = ""
                        ):
 
    print(f"path_lens = {path_lens}")
    path = join(ROOT, "tmp/last_exp_figs/")
    fname = path + model_name + "_pred_on_unseen" + ".png" 
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
            if idx%100==0:
                print(idx)
            states = preds_list[idx]
            t_done = path_lens[idx] #TODO: change 
            # print(f"******* Verify: visualize_op: states.shape= {states.shape}")
            if at_time != None:
                assert(at_time >= 1), f"Can only plot at_time >= 1 only"
                # if at_time > t_done, just plot for t_done
                at_time = min(at_time, t_done)
            else:
                at_time = t_done

            if stats!=None:
                mean, std = stats
                print(f"{states.shape}, {mean.shape}")
                states = (states*std) + mean
        
            # shape: (eval_batch_size, max_test_ep_len, state_dim)
            if color_by_time:
                plt.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=scalarMap.to_rgba(t_done), alpha=0.2)
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
        obstacle = DOLS_obstacle()
        ax.add_patch(obstacle)
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
        wandb.log({"pred_traj_fig_"+wandb_suffix: wandb.Image(fname)})


    return fig

def compare_trajectories(tr_op_traj_dict_list,
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
                            wandb_suffix="",
                            ):
    
    tr_set_txy_preds = [d['states'] for d in tr_op_traj_dict_list]
    tr_set_txy_PREDS_ = [d['states_for_action_labels'] for d in tr_op_traj_dict_list]
    actions = [d['actions'] for d in tr_op_traj_dict_list]
    ACTIONS_ = [d["action_labels"] for d in tr_op_traj_dict_list]
    mses = [d['mse'].item() for d in tr_op_traj_dict_list]
    
    path = join(ROOT, "tmp/last_exp_figs/")
    fname = path + "compare_traj" + str(iter_i) + ".png" 
    fig = plt.figure()
    plt.cla()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    if stats!=None:
        print("===== Note: rescaling states to original scale for viz=====")

    if traj_idx==None:
        traj_idx = [k for k in range(len(tr_set_txy_preds))]

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


    for idx in traj_idx:
        states = tr_set_txy_preds[idx]
        STATES_ = tr_set_txy_PREDS_[idx]
        mse = np.round(mses[idx],6)
        if stats!=None:
            mean, std = stats
            states = (states*std) + mean
            STATES_ = (STATES_*std) + mean
    
        if show_scatter:
            plt.scatter(states[0,:,1], states[0,:,2],s=1, c='r', label=f"id_{idx}: {mse}")
            plt.scatter(STATES_[0,:,1], STATES_[0,:,2],s=1, c='g')

        # Plot policy at visites states
        _, nstates,_ = states.shape
        if plot_policy:
            action = actions[idx]
            ACTION_ =  ACTIONS_[idx]
            for i in range(nstates-1):
                plt.arrow(states[0,i,1], states[0,i,2], np.cos(action[0,i,0]), np.sin(action[0,i,0]))
                plt.arrow(STATES_[0,i,1], STATES_[0,i,2], np.cos(ACTION_[0,i,0]), np.sin(ACTION_[0,i,0]))
    plt.legend()
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
        wandb.log({"Compare"+wandb_suffix: wandb.Image(fname)})


def simulate_tgt_actions(traj_dataset,
                           env=None,
                           log_wandb=True,
                           wandb_fname='simulate_tgt_actions',
                           plot_flow=True,
                           at_time=119,
                           break_at=None,
                           ):
    
    path = join(ROOT, "tmp/last_exp_figs/")
    fname = path + "simulated_input_traj"  + ".png"

    fig = plt.figure()
    plt.cla()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    title = "Input_traj_"
    plt.title(title)
    success_count_ = 0
    action_list = [item[2] for item in traj_dataset.dataset]
    rzn_list = [item[-1] for item in traj_dataset.dataset]
    flow_dir_list = [item[-2] for item in traj_dataset.dataset]
    for i in range(int(0.1*len(traj_dataset))):
        if i == break_at:
            break
        rzn = rzn_list[i]
        flow_dir = flow_dir_list[i]
        env.set_rzn(rzn)
        env.reset()
        actions =  action_list[i]
        txy_array = np.zeros((121,3))
        txy_array[:,:] = None
        txy_array[0,:] = np.array([0,env.start_pos[0],env.start_pos[1]])
        reached_target_ = False

        for k in range(len(actions)):
            a = actions[k]
            txy, reward ,done, info = env.step(a)
            txy_array[k+1,:] = txy
            if done:
                if reward > 0:
                    reached_target_ = True
                    success_count_ += 1
                break

        plt.plot(txy_array[:,1], txy_array[:,2])
        plt.scatter(txy_array[len(actions),1], txy_array[len(actions),2], s=1.5, zorder=1000)

    plt.xlim([0, 100.])
    plt.ylim([0, 100.])

    if env != None:
        plt.xlim([0, env.xlim])
        plt.ylim([0, env.ylim])
        obstacle = DOLS_obstacle()
        ax.add_patch(obstacle)
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
    
    print("success_count = ", success_count_)
    return success_count_
    
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
                    data_name='',
                    break_at=None
                    ):
 
    print(" ---- Visualizing input ---- ")

    path = join(ROOT, "tmp/last_exp_figs/")
    fname = path + "input_traj" + info_str + data_name + ".png"

    fig = plt.figure()
    plt.cla()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    title = "Input_traj_" + info_str
    plt.title(title)

    states_list =[item[3] for item in traj_dataset.dataset]


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
                states = states_list[idx]
                t_done = len(states)  # no. of points to plot. No need to plot masked data.
            t_dones.append(t_done)
        vmin = min(t_dones)
        vmax = max(t_dones)
        # Make a user-defined colormap.
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('YlOrRd')
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)

    for idx, traj in enumerate(traj_dataset):
        if idx == break_at:
            break
        if idx in traj_idx:
            states = states_list[idx]
            t_done = t_dones[idx]

            if stats != None:
                mean, std = stats
                states = (states*std) + mean
                # states = states*(traj_mask.reshape(-1,1))

            if color_by_time:
                plt.plot(states[:,1], states[:,2], color=scalarMap.to_rgba(t_done), alpha= 0.2)
            else:
                plt.plot(states[:,1], states[:,2], linewidth=0.1)
            plt.scatter(states[-1,1], states[-1,2], alpha=0.5, zorder=10000, s=5)
            # print(f"bcords: {states[-1,1], states[-1,2]}")
    plt.xlim([0, 100.])
    plt.ylim([0, 100.])
    if color_by_time:
        cbar = plt.colorbar(scalarMap, label="Arrival Time")

    if env != None:
        plt.xlim([0, env.xlim])
        plt.ylim([0, env.ylim])
        obstacle = DOLS_obstacle()
        ax.add_patch(obstacle)
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

def see_steplr_trend(step_size: int, num_epochs: int, lr=0.001, final_lr=None, gamma=None, show_plot=False):
    nsteps = int(num_epochs/step_size)
    lr = lr
    if gamma == None and final_lr != None:
        final_lr = final_lr
        gamma = (final_lr/lr)**(1/nsteps)
        print(f"use gamma = {gamma}")

    if gamma != None and final_lr == None:
        final_lr = lr*gamma**nsteps
        print(f"final_lr= {final_lr}")

    y = np.zeros(num_epochs,)
    for i in range(nsteps):
        y[i*step_size:(i+1)*step_size] = lr*gamma**i
    if show_plot:
        plt.yscale("log")  
        plt.plot(y)
        plt.show()

    return gamma, final_lr



def plot_attention_weights(weight_mat, 
                            layer_idx=0,
                            set_idx=0, 
                            average_across_layers=False,
                            scale_each_row=False, 
                            causal_mask_used=True,
                            cmap=cm.Reds, 
                            log_wandb = False,
                            fname='attention_heatmap.png',
                            info_string = '',
                            wandb_fname = ''                 
                            ):

    """
    weight_mat: (avgd across heads) weight matrix expected shape = [1-6],32,120,120 or L,B,T,T where T is 3*context_len
    set_idx: sample index of batch
    layer_idx: layer index
    scale_each_row: scales each row INDEPENDENTLY to lie between 0 and 1 for visualization
    """
    plt.cla()
    plt.clf()
    plt.close()
    weights = weight_mat
    shape = weights.shape
    
    # Plot attenetion scores for the ith trajectory/sample among the batch (for training batch)

    
    scale_str = ''
    layer_str = ''
    if average_across_layers:
        layer_str = 'avg'
        weights = np.mean(weights[:,set_idx,:,:],axis=0)
    else:
        layer_str = str(layer_idx)
        weights = weights[layer_idx,set_idx,:,:]

    if scale_each_row:
        scale_str = '_scaled_rows_'
        weights = scale_attention_rows(weights, causal_mask_used=causal_mask_used)


    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_ticks(np.arange(0,120,20))
    ax.yaxis.set_ticks(np.arange(0,120,20))
    shw = ax.imshow(weights, cmap=cmap)
    bar = plt.colorbar(shw)
    bar_fontsize = 12
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    # cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), shrink=0.65)
    bar.set_label("Attention Weights (scaled)", fontsize=bar_fontsize)

    title =  scale_str + info_string  +"_L" + layer_str + "_smp" +  str(set_idx)
    # plt.title(title)
    # plt.figure(figsize=(12,10))
    # ax = sns.heatmap(weights, linewidth=0.05)
    fname=fname + title + ".png"
    plt.savefig(fname, bbox_inches="tight", dpi=600)
    if log_wandb:
        if wandb_fname == None:
            wandb_fname =  "attention map"  
        wandb.log({wandb_fname: wandb.Image(fname)})



def scale_attention_rows(at_weights, causal_mask_used=False):
    """
    at_weights: 2d matrix of attntion weights
    """
    shape = at_weights.shape
    # print(f"**** Verify type= {type(at_weights)}")
    # print(f"**** Verify shape= {shape}")

    assert(len(shape)==2), f"Invalid shape < {len(shape)} > of weight matrix"
    if causal_mask_used:
        for i in range(shape[0]):   # shape[0] is T (no. of rows)
            min_wi = np.min(at_weights[i,0:i+1])
            del_wi = np.max(at_weights[i,0:i+1]) - min_wi
            at_weights[i,0:i+1] -= min_wi
            if del_wi != 0:
                at_weights[i,0:i+1] /= del_wi
    else:
        for i in range(shape[0]):   # shape[0] is T (no. of rows)
            min_wi = np.min(at_weights[i,0:])
            del_wi = np.max(at_weights[i,0:]) - min_wi
            at_weights[i,0:] -= min_wi
            if del_wi != 0:
                at_weights[i,0:] /= del_wi
            
    return at_weights

def viz_op_traj_with_attention(txy_preds_list,
                        all_at_mat_list, # could be enc_sa, dec_sa, dec_ga
                        path_lens,
                        mode='dec_sa',       #or 'a_s_attention'
                        layer_idx=0,
                        batch_idx=0, 
                        average_across_layers=False,
                        scale_each_row=True,
                        stats=None, 
                        env=None, 
                        log_wandb=True, 
                        plot_policy=False,
                        traj_idx=None,      #None=all, list of rzn_ids []
                        show_scatter=False,
                        plot_flow=False,
                        at_time=None,
                        model_name=''
                        ):
    
    path = join(ROOT, "tmp/last_exp_figs/")
    fname = path + "att" + mode + model_name + "_@t"+str(at_time)+ ".png" 
    fig = plt.figure()
    plt.cla()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    # plt.title(f"{mode}")
    # enc_sa_arr , dec_sa_arr, dec_ga_arr = all_att_mats
    # 0             1           2
    if mode == 'enc_sa':
        weights_list = [item[0] for item in all_at_mat_list]
    elif mode == 'dec_sa':
        weights_list = [item[1] for item in all_at_mat_list]
    elif mode == 'dec_ga':
        weights_list = [item[2] for item in all_at_mat_list]
    else:
        raise ValueError(f'no such mode :{mode}')
    
    

        
    if stats!=None:
        print("===== Note: rescaling states to original scale for viz=====")

    if traj_idx==None:
        traj_idx = [k for k in range(len(txy_preds_list))]

    # print(f" ***** Verify: traj_idx = {traj_idx}")
    for idx,traj in enumerate(txy_preds_list):
        if idx in traj_idx:
            states = txy_preds_list[idx]
            t_done =  path_lens[idx]
            weights = weights_list[idx]
            if average_across_layers:
                layer_str = 'avg'
                weights = np.mean(weights[:,batch_idx,:,:],axis=0)
            else:
                layer_str = str(layer_idx)
                weights = weights[layer_idx,batch_idx,:,:]
            if scale_each_row:
                scale_str = '_scaled_rows_'
                weights = scale_attention_rows(weights)
                
            if at_time != None:
                assert(at_time >= 1), f"Can only plot at_time >= 1 only"
                # if at_time > t_done, just plot for t_done
                at_time = min(at_time, t_done)
            else:
                at_time = t_done
            t_done = at_time

            # print(f"******* Verify: visualize_op: states.shape= {states.shape}")
            if stats!=None:
                mean, std = stats
                states = (states*std) + mean
        
            # Plot states
            # shape: (eval_batch_size, max_test_ep_len, state_dim)
            for t in range(t_done):
                plt.plot(states[0,t:t+2,1], states[0,t:t+2,2], 
                            c=cm.Reds(weights[t_done-1,t])) #TODO: verify weights idx


            if show_scatter:
                plt.scatter(states[0,:,1], states[0,:,2],s=0.5)

    # plot target area and set limits
    if env != None:
        setup_ax(ax,env)
        plt.xlim([0, env.xlim])
        plt.ylim([0, env.ylim])
        # print("****VERIFY: env.target_pos: ", env.target_pos)
        obstacle = DOLS_obstacle()
        ax.add_patch(obstacle)
        if env.target_pos.ndim == 1:
            target_circle = plt.Circle(env.target_pos, env.target_rad, color='r', alpha=0.3)
            ax.add_patch(target_circle)
        elif env.target_pos.ndim > 1:
            for target_pos in env.target_pos:
                target_circle = plt.Circle(target_pos, env.target_rad, color='r', alpha=0.3)
                ax.add_patch(target_circle)

        if plot_flow and at_time!=None:
            plot_vel_field(env,at_time-1)

    plt.savefig(fname, bbox_inches="tight", dpi=600)
    wandbfname = "pred_trajs_with_" + mode
    if log_wandb:
        wandb.log({wandbfname: wandb.Image(fname)})

    plt.close()
    return fname

def setup_ax(ax,env, show_xlabel= True, 
                            show_ylabel=True, 
                            show_states=True,
                            show_xticks=True,
                            show_yticks=True,
                            lab_fs = 15,
                            tick_fs = 14,
                        ):
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([0,100])
    ax.set_ylim([0,100])
    xticks = np.arange(0,100+25,25)
    yticks = xticks.copy()
    ax.xaxis.set_ticks(xticks)
    ax.yaxis.set_ticks(yticks)
    ax.set_xticklabels(xticks, fontsize=tick_fs)
    ax.set_yticklabels(yticks, fontsize=tick_fs)

    if not show_xticks:
        ax.tick_params(axis='x',       
                        which='both',      
                        bottom=False,      
                        labelbottom=False,
                        labelsize=tick_fs)
    if not show_yticks:
        ax.tick_params(axis='y',       
                which='both',      
                left=False,      
                labelleft=False,
                labelsize=tick_fs)
    xlabel = f"X "
    ylabel = f"Y "
    xlabel += "(Non-Dim)"
    ylabel += "(Non-Dim)"
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=lab_fs)
    if show_ylabel:
        ax.set_ylabel(ylabel,fontsize=lab_fs)
    if show_states:
        ax.scatter(env.start_pos[0], env.start_pos[1], color='k', marker='o')
    
        if env.target_pos.ndim == 1:
            ax.scatter(env.target_pos[0], env.target_pos[1], color='k', marker='*')
            target_circle = plt.Circle(env.target_pos, env.target_rad, color='r', alpha=0.3)
            ax.add_patch(target_circle)
        elif env.target_pos.ndim > 1:
            for target_pos in env.target_pos:
                ax.scatter(target_pos[0], target_pos[1], color='k', marker='*')
                target_circle = plt.Circle(target_pos, env.target_rad, color='r', alpha=0.3)
                ax.add_patch(target_circle)