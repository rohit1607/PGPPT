import sys
import numpy as np

from root_path import ROOT
from os.path import join

sys.path.insert(0, ROOT)
import seaborn as sns
import wandb
from datetime import datetime
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcol
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from src_utils import scale_attention_rows
import matplotlib.patches as patches


class paper_plots:
    def __init__(self, env,  op_traj_dict_list, stats, paper_plot_info, non_dim_plots=True, save_dir='../tmp/'):
        self.env = env
        self.op_traj_dict_list = op_traj_dict_list
        self.stats = stats  #training mean and variance for normalization
        self.paper_plot_info = paper_plot_info
        self.save_dir = save_dir
        self.non_dim_plots = non_dim_plots

    def plot_all(self):
        self.plot_traj_by_arr()
        self.plot_traj_att()
        self.plot_att_heatmap()


    def plot_vel_field(self,ax, t,r=0, g_strmplot_lw=1, g_strmplot_arrowsize=1):
        # Make modes the last axis
        Ui = np.transpose(self.env.Ui,(0,2,3,1))
        Vi = np.transpose(self.env.Vi,(0,2,3,1))
        vx_grid = self.env.U[t,:,:] + np.dot(Ui[t,:,:,:],self.env.Yi[t,r,:])
        vy_grid = self.env.V[t,:,:] + np.dot(Vi[t,:,:,:],self.env.Yi[t,r,:])
        vx_grid = np.flipud(vx_grid)
        vy_grid = np.flipud(vy_grid)
        Xs = np.arange(0,self.env.xlim) + (self.env.dxy/2)
        Ys = np.arange(0,self.env.ylim) + (self.env.dxy/2)
        X,Y = np.meshgrid(Xs, Ys)
        ax.streamplot(X, Y, vx_grid, vy_grid, color = 'grey', zorder = 0,  linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
        v_mag_grid = (vx_grid**2 + vy_grid**2)**0.5
        im = ax.contourf(X, Y, v_mag_grid, cmap = "Blues", alpha = 0.5, zorder = -1e5)
        return im

    

    def plot_traj_by_arr(self, traj_dataset, set_str=""):
        info = self.paper_plot_info["trajs_by_arr"]
        t_dones = []
        for traj in traj_dataset:
            _,_,_,_, traj_mask,_ = traj
            t_done = int(np.sum(traj_mask.numpy()))   # no. of points to plot. No need to plot masked data.
            t_dones.append(t_done)
        vmin = min(t_dones)
        vmax = max(t_dones)
        # vmax = 51
        # print(f"---- Info: tdone: min={vmin}, max={vmax} ")
        plt.hist(t_dones)
        plt.savefig("/home/rohit/Documents/Research/Planning_with_transformers/Decision_transformer/my-dec-transformer/tmp/tdone-hist")
        fig, ax = plt.subplots()

        # Make a user-defined colormap.
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('YlOrRd')
        sm = cm.ScalarMappable(norm=cNorm, cmap=cmap)

        self.setup_ax(ax)       
        im = self.plot_vel_field(ax,t=vmax, r=199)
        # traj_dataset=random.shuffle(traj_dataset)
        for idx, traj in enumerate(traj_dataset):
            timesteps, states, actions, returns_to_go, traj_mask, _ = traj
            t_done = int(np.sum(traj_mask.numpy()))   # no. of points to plot. No need to plot masked data
           
            mean, std = self.stats
            states = (states*std) + mean
            states = states*(traj_mask.reshape(-1,1))

            ax.plot(states[:t_done,1], states[:t_done,2], color=sm.to_rgba(t_done), alpha=1 )
            ax.scatter(states[-1,1], states[-1,2], alpha=0.5, zorder=10000, s=5)
            # if idx>10:
            #     break


        cbar_fontsize = 12
        cbar = fig.colorbar(sm, ax=ax, ticks=[i for i in range(vmin, vmax+1, 3)])
        cbar.set_label("Arrival Time (non-dim)", fontsize=cbar_fontsize)
        
        cbarv = fig.colorbar(im, ax=ax)
        cbarv.set_label("Velocity Magnitude (non-dim)", fontsize=cbar_fontsize)
        fname = info["fname"] + set_str
        save_name = join(self.save_dir,fname)
        plt.savefig(save_name, bbox_inches = 'tight', dpi=600)
        return ax

    def f(self, obs_token):

        ru,cl,w = obs_token
        w = w*100
        y_l = 100*(1-ru) + 0.5 + w
        x_l = 100*cl - 0.5
        return x_l, y_l, w
    
    def plot_obstacle(self, ax, obs_token=None, xyw=None):
        if obs_token != None:
            x_ll, y_ll, w = self.f(obs_token)
        elif xyw != None:
            x_ll, y_ll, w = xyw
        else:
            raise ValueError('provide iether token or xyw')
        rect = patches.Rectangle((x_ll, y_ll), w, w, linewidth=1, edgecolor='k', facecolor='grey')
        ax.add_patch(rect)
        return
        
        
    def plot_val_ip_op(self, traj_dataset,
                       preds_list,
                       path_lens,
                       success_list,
                        at_time=None):
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10,5))

        info = self.paper_plot_info["plot_val_ip_op"]
        
        ip_states_list =[item[3] for item in traj_dataset.dataset]
        ip_path_lens = [len(item[2]) for item in traj_dataset.dataset] #item[2]
        vmin = min(path_lens + ip_path_lens)
        vmax = max(path_lens + ip_path_lens)
        vmax = min(vmax, 119)
        vmax = int(100)
        vmin = int(60)
        # # vmax = 51

        # Make a user-defined colormap.
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('YlOrRd')
        sm = cm.ScalarMappable(norm=cNorm, cmap=cmap)
        
        # Works for static obstacle only
        obs_token = traj_dataset.dataset[0][1][0] #sample, obstacle key, timestep
        xyw=(-10,-10,5)

        ax = axs[0]
        self.setup_ax(ax)       
        im = self.plot_vel_field(ax,t=vmax,r=199)
        self.plot_obstacle(ax, xyw=xyw)
        # traj_dataset=random.shuffle(traj_dataset)

        for idx,traj in enumerate(ip_states_list):
            states = ip_states_list[idx]
            t_done = ip_path_lens[idx]
        #     pr_t_dones.append(t_done)
        #     # Plot sstates
        #     # shape: (eval_batch_size, max_test_ep_len, state_dim)
        #     if t_done < 68:
            # ax.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=sm.to_rgba(t_done))
            ax.plot(states[:t_done,1], states[:t_done,2], color=sm.to_rgba(t_done))

        #         # ax.scatter(states[0,:t_done+1,1], states[0,:t_done+1,2], color=sm.to_rgba(t_done),s=1)
        pr_t_dones = []
        ax = axs[1]
        self.setup_ax(ax, show_ylabel=False)
        im = self.plot_vel_field(ax,t=vmax,r=199)
        self.plot_obstacle(ax, xyw=xyw)

        for idx, traj in enumerate(preds_list[:1500]):
            states = preds_list[idx]
            t_done = path_lens[idx] 
            if success_list[idx]:
                # ax.scatter(states[:t_done,1], states[:t_done,2], color=sm.to_rgba(t_done), alpha=1, s=1 )
                ax.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=sm.to_rgba(t_done), alpha=1 )
                # ax.scatter(states[-1,1], states[-1,2], alpha=0.5, zorder=10000, s=5)

        summary = {}
        summary["mean Tarr logged dataset"] = np.mean(ip_path_lens)
        summary["std Tarr logged dataset"] = np.std(ip_path_lens)
        summary["mean Tarr prediction" ] = np.mean(path_lens)
        summary["std Tarr prediction" ] = np.std(path_lens)
        summary["success rate"] = np.sum([int(item) for item in success_list])/len(success_list)
        summary["prediction count"] = len(success_list)
        print("------ SUMMARY-------\n", summary)
        # cbar_fontsize = 12
        # cbar = fig.colorbar(sm, ax=ax, ticks=[i for i in range(vmin, vmax+1)])
        # cbar.set_label("Arrival Time (non-dim units)", fontsize=cbar_fontsize)
        
        # cbarv = fig.colorbar(im, ax=ax)
        # cbarv.set_label("Velocity Magnitude", fontsize=cbar_fontsize)
        
        plt.subplots_adjust( left= 0.1, right=0.9, top=0.9, bottom=0.2, wspace=-0.05)

        cax_arr = ax.inset_axes([1.05, 0, 0.05, 1])
        cax_vel = ax.inset_axes([1.30, 0, 0.05, 1])
        cbar_fontsize = 15
        cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), cax=cax_arr)
        cbar.set_label("Arrival Time (non-dim)", fontsize=cbar_fontsize)
     
        cbarv = fig.colorbar(im, ax=axs.ravel().tolist(), cax=cax_vel)
        cbarv.set_label("Velocity Magnitude (non-dim)", fontsize=cbar_fontsize)


        fname = info["fname"] 
        save_name = join(self.save_dir,fname)
        plt.savefig(save_name, bbox_inches = 'tight', dpi=600)

    def plot_train_val_ip_op(self, tr_traj_dataset, val_traj_dataset,
                            at_time=None):
        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(15,5))

        info = self.paper_plot_info["plot_train_val_ip_op"]
        t_dones = []
        for traj in val_traj_dataset:
            _,_,_,_, traj_mask,_ = traj
            t_done = int(np.sum(traj_mask.numpy()))   # no. of points to plot. No need to plot masked data.
            t_dones.append(t_done)
        vmin = min(t_dones)
        vmax = max(t_dones,119)

        # vmax = 51

        # Make a user-defined colormap.
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('YlOrRd')
        sm = cm.ScalarMappable(norm=cNorm, cmap=cmap)

        ax = axs[0]
        self.setup_ax(ax)       
        im = self.plot_vel_field(ax,t=vmax,r=199)
        # traj_dataset=random.shuffle(traj_dataset)
        for idx, traj in enumerate(tr_traj_dataset):
            timesteps, states, actions, returns_to_go, traj_mask,_ = traj
            t_done = int(np.sum(traj_mask.numpy()))   # no. of points to plot. No need to plot masked data
            mean, std = self.stats
            states = (states*std) + mean
            states = states*(traj_mask.reshape(-1,1))

            ax.plot(states[:t_done,1], states[:t_done,2], color=sm.to_rgba(t_done), alpha=1 )
            ax.scatter(states[-1,1], states[-1,2], alpha=0.5, zorder=10000, s=5)

        ax = axs[1]
        self.setup_ax(ax)       
        im = self.plot_vel_field(ax,t=vmax,r=199)
        # traj_dataset=random.shuffle(traj_dataset)
        for idx, traj in enumerate(val_traj_dataset):
            timesteps, states, actions, returns_to_go, traj_mask,_ = traj
            t_done = int(np.sum(traj_mask.numpy()))   # no. of points to plot. No need to plot masked data
            mean, std = self.stats
            states = (states*std) + mean
            states = states*(traj_mask.reshape(-1,1))

            ax.plot(states[:t_done,1], states[:t_done,2], color=sm.to_rgba(t_done), alpha=1 )
            ax.scatter(states[-1,1], states[-1,2], alpha=0.5, zorder=10000, s=5)

        pr_t_dones = []
        ax = axs[2]
        self.setup_ax(ax, show_ylabel=False)
        im = self.plot_vel_field(ax,t=vmax,r=199)
        print(f"{len(self.op_traj_dict_list)}")
        # sys.exit()
        for idx,traj in enumerate(self.op_traj_dict_list):
            states = traj['states']
            t_done =  traj['t_done']
            pr_t_dones.append(t_done)
            mean, std = self.stats
            states = (states*std) + mean
        
            # Plot sstates
            # shape: (eval_batch_size, max_test_ep_len, state_dim)
            ax.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=sm.to_rgba(t_done))
                
        wandb.run.summary["mean Tarr logged dataset"] = np.mean(t_dones)
        wandb.run.summary["std Tarr logged dataset"] = np.std(t_dones)
        wandb.run.summary["mean Tarr prediction" ] = np.mean(pr_t_dones)
        wandb.run.summary["std Tarr prediction" ] = np.std(pr_t_dones)
     
        # cbar_fontsize = 12
        # cbar = fig.colorbar(sm, ax=ax, ticks=[i for i in range(vmin, vmax+1)])
        # cbar.set_label("Arrival Time (non-dim units)", fontsize=cbar_fontsize)
        
        # cbarv = fig.colorbar(im, ax=ax)
        # cbarv.set_label("Velocity Magnitude", fontsize=cbar_fontsize)
        plt.subplots_adjust( left= 0.1, right=0.9, top=0.9, bottom=0.2, wspace=-0.05)

        cax_arr = ax.inset_axes([1.05, 0, 0.05, 1])
        cax_vel = ax.inset_axes([1.30, 0, 0.05, 1])
        cbar_fontsize = 14
        cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), cax=cax_arr)
        cbar.set_label("Arrival Time (non-dim)", fontsize=cbar_fontsize)
     
        cbarv = fig.colorbar(im, ax=axs.ravel().tolist(), cax=cax_vel)
        cbarv.set_label("Velocity Magnitude (non-dim)", fontsize=cbar_fontsize)


        fname = info["fname"] 
        save_name = join(self.save_dir,fname)
        plt.savefig(save_name, bbox_inches = 'tight', dpi=600)





    def plot_att_trajs_at_t(self,ax,at_time,mode,stats):
        for idx,traj in enumerate(self.op_traj_dict_list):
            states = traj['states']
            t_done =  traj['t_done']
            # print(f"-------- verify : states= {states}")
            if at_time != None:
                assert(at_time >= 1), f"Can only plot at_time >= 1 only"
                # if at_time > t_done, just plot for t_done
                t_done = min(at_time, t_done)
            else:
                at_time = t_done
           
            # Rescale
            mean, std = stats
            states = (states*std) + mean

            at_weights = traj['attention_weights'][0,0,:,:].cpu().detach().numpy()
            a_s_wts_scaled = scale_attention_rows(at_weights[2::3,1::3])
            a_a_wts_scaled = scale_attention_rows(at_weights[2::3,2::3])

            alpha = 0.7
            ax.scatter(states[0,at_time,1], states[0,at_time,2], c='k', marker='p')
            if mode == 'a_a_attention':
                for t in range(t_done):
                    im =ax.plot(states[0,t:t+2,1], states[0,t:t+2,2], 
                                c=cm.Reds(a_a_wts_scaled[t_done-1,t]), zorder=10, alpha=alpha)
            elif mode == 'a_s_attention':
                for t in range(t_done):
                    im = ax.plot(states[0,t:t+2,1], states[0,t:t+2,2], 
                                c=cm.Greens(a_s_wts_scaled[t_done-1,t]), zorder=10, alpha=alpha)
            else:
                raise Exception("invalid argument for mode")
        return im

    def plot_traj_by_att(self, mode):
        if mode == 'a_a_attention':
            cmap = cm.Reds
        elif mode == 'a_s_attention':
            cmap = cm.Greens
        else:
            raise Exception("invalid argument for mode")
        info = self.paper_plot_info["trajs_by_att"]
        nplots = len(info["ts"])
       
        fig, axs = plt.subplots(1,nplots, sharey=True, figsize=(15,5))
        # fig.suptitle('')    #title of overall plot
        show_ylabel = True
        for i in range(nplots):
            if i>=1:
                show_ylabel=False
            ax = axs[i]
            at_time = info["ts"][i]
            self.setup_ax(ax,show_ylabel=show_ylabel)
            self.plot_att_trajs_at_t(ax, at_time, mode, self.stats)
            im = self.plot_vel_field(ax,at_time,g_strmplot_lw=0.5, g_strmplot_arrowsize=0.5)
            ax.set_title(f"t={at_time}")
        
        # colorbars
        # cbar1_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar_fontsize = 12
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), shrink=0.65)
        cbar.set_label("Attention Weights (scaled)", fontsize=cbar_fontsize)
     
        cbarv = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.65)
        cbarv.set_label("Velocity Magnitude", fontsize=cbar_fontsize)
        fname = info["fname"] + "_" + mode
        save_name = join(self.save_dir,fname)
        plt.savefig(save_name, bbox_inches = 'tight', dpi=600)
        return


    def setup_ax(self, ax, show_xlabel= True, 
                            show_ylabel=True, 
                            show_states=True,
                            show_xticks=True,
                            show_yticks=True,
                            lab_fs = 15,
                            tick_fs = 14,
                            ):
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([0,self.env.xlim])
        ax.set_ylim([0,self.env.ylim])
        xticks = np.arange(0,self.env.xlim +25,25)
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
        if self.non_dim_plots == True:
            xlabel += "(Non-Dim)"
            ylabel += "(Non-Dim)"
        if show_xlabel:
            ax.set_xlabel(xlabel, fontsize=lab_fs)
        if show_ylabel:
            ax.set_ylabel(ylabel,fontsize=lab_fs)
        if show_states:
            ax.scatter(self.env.start_pos[0], self.env.start_pos[1], color='k', marker='o')
        
            if self.env.target_pos.ndim == 1:
                ax.scatter(self.env.target_pos[0], self.env.target_pos[1], color='k', marker='*')
                target_circle = plt.Circle(self.env.target_pos, self.env.target_rad, color='r', alpha=0.3)
                ax.add_patch(target_circle)
            elif self.env.target_pos.ndim > 1:
                for target_pos in self.env.target_pos:
                    ax.scatter(target_pos[0], target_pos[1], color='k', marker='*')
                    target_circle = plt.Circle(target_pos, self.env.target_rad, color='r', alpha=0.3)
                    ax.add_patch(target_circle)


    # TODO: Change as per causal_mask
    def plot_att_heatmap(self, set_idx=0, sample_idx=0):
        """
        attention_weights: weight matrix expected shape = 1(or 64),1,210,210 or B,N,T,T where T is 3*context_len
        idx: sample index of batch
        scale_each_row: scales each row INDEPENDENTLY to lie between 0 and 1 for visualization
        """
        info = self.paper_plot_info["att_heatmap"]
        op_traj_dict = self.op_traj_dict_list[set_idx]
        # attention weigghts in the last block
        attention_weights = op_traj_dict['attention_weights']
        # normalized_weights = F.softmax(attention_weights, dim=-1)

        weights = attention_weights.cpu().detach().numpy()
        shape = weights.shape
        
        # Plot attenetion scores for the ith trajectory/sample among the batch (for training batch)
        weights = weights[sample_idx,0,:,:]
        
        # scale each row visualization
        weights = scale_attention_rows(weights)

        fig, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_ticks(np.arange(0,180,25))
        ax.yaxis.set_ticks(np.arange(0,180, 25))
        shw = ax.imshow(weights, cmap=cm.Reds)
        cbar = plt.colorbar(shw)
        cbar_fontsize = 12
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        # cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), shrink=0.65)
        cbar.set_label("Attention Weights (scaled)", fontsize=cbar_fontsize)

        # title =  scale_str + info_string + "setIdx-" +  str(set_idx)
        # plt.title(title)
        # plt.figure(figsize=(12,10))
        # ax = sns.heatmap(weights, linewidth=0.05)
        fname = info["fname"]
        save_name = join(self.save_dir,fname)
        plt.savefig(save_name, bbox_inches="tight", dpi=600)

        return



    def load_velocity(self, vel_fname):
        U = np.load(join(vel_fname,"all_u_mat.npy"))
        V = np.load(join(vel_fname,"all_v_mat.npy"))
        Ui = np.load(join(vel_fname,"all_ui_mat.npy"))
        Vi = np.load(join(vel_fname,"all_vi_mat.npy"))
        Yi = np.load(join(vel_fname,"all_Yi.npy"))

        # replace nans with 0s and scale velocity as per vmax_by_F factor
        U[np.isnan(U)] = 0
        V[np.isnan(V)] = 0
        Ui[np.isnan(Ui)] = 0
        Vi[np.isnan(Vi)] = 0
        Yi[np.isnan(Yi)] = 0

        return (U, V, Ui, Vi, Yi)
    
