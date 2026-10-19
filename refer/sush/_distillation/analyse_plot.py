import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import gvar as gv
import lsqfit
from typing import List, Literal


class plot_init:
    def __init__(
            self,
            save_path:str
        ):

        self.marker_array = np.array(['s','*','+','x','p','h','v','X','D','P','H','o'])
        self.color = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
        
        self.save_path = save_path
    
    def plot_meff_2pt(self, data_info:dict, y_range:list, x_range:list = [0, 30], Re_or_Im:str = 'Re', name:str=None, save_name:str=None, data_label:list=[]):
        if not name : name = 'meff'
        if not save_name : save_name = 'meff'
        
        plt.rcParams.update({'font.size':25})
        fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        # ax.set_title('%s_meff_%dx%d_%s_ENV'%(self.hadron, self.Nx, self.Nt, self.meff_type),fontdict={'fontsize':30,'fontweight':'light'})
        ax.set_ylim(y_range)
        ax.set_xlim(x_range)

        ax.set_xlabel('%s'%('t/a'))
        ax.set_ylabel('%s'%('$E_{\mathrm{2pt}}$/Gev'))
        ax.set_title(label='%s'%(name), fontdict={'fontsize':30,'fontweight':'light'})
        
        i = 0
        
        for file_name in data_info.keys():
            if Re_or_Im == 'Re':
                pos = 0

            elif Re_or_Im == 'Im':
                pos = 1
            
            meff_2pt_mean = data_info[file_name]['meff_2pt_mean']
            meff_2pt_err = data_info[file_name]['meff_2pt_err']
            Mom = data_info[file_name]['Mom']
        

            for Nev_indx in range(meff_2pt_mean.shape[1]):
                Re_2pt_mean_ENV = meff_2pt_mean[0, 0, Nev_indx, 0, 0, :, pos]
                Re_2pt_err_ENV = meff_2pt_err[0, 0, Nev_indx, 0, 0, :, pos]
                ax.errorbar(
                    np.arange(Re_2pt_mean_ENV.shape[0]) - (len(data_info.keys())//2 + i) * 0.1,
                    Re_2pt_mean_ENV, 
                    yerr=Re_2pt_err_ENV, 
                    alpha=0.5, 
                    marker = self.marker_array[0 + i], 
                    capsize=3.5, 
                    capthick=1.5, 
                    label=f"{data_label[i]}, P={Mom[0]}", 
                    linestyle='none',
                    elinewidth=2
                    ) # fmt = 'bs'
            i += 1
            
        plt.legend()
        
        fig.savefig("%s/%s"%(self.save_path, '%s.pdf'%(save_name)))

    def plot_meff_all(self, data_info:dict, y_range:list, x_range:list = [0, 30], Re_or_Im:str = 'Re', name:str=None, save_name:str=None, data_label:list=[]):
        if not name : name = 'meff'
        if not save_name : save_name = 'meff'
        
        plt.rcParams.update({'font.size':20})
        fig, ax = plt.subplots(1,1, figsize=(30, 30*0.5))
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        # ax.set_title('%s_meff_%dx%d_%s_ENV'%(self.hadron, self.Nx, self.Nt, self.meff_type),fontdict={'fontsize':30,'fontweight':'light'})
        ax.set_ylim(y_range)
        ax.set_xlim(x_range)

        ax.set_xlabel('%s'%('t/a'))
        ax.set_ylabel('%s'%('$E_{\mathrm{2pt}}$/Gev'))
        ax.set_title(label='%s'%(name), fontdict={'fontsize':30,'fontweight':'light'})
        
        i = 0
        
        for file_name in data_info.keys():
            if Re_or_Im == 'Re':
                pos = 0

            elif Re_or_Im == 'Im':
                pos = 1
            
            meff_2pt_mean = data_info[file_name]['meff_2pt_mean']
            meff_2pt_err = data_info[file_name]['meff_2pt_err']
            Mom = data_info[file_name]['Mom']
        
            for P_indx in range(len(Mom)):
                for Nev_indx in range(meff_2pt_mean.shape[2]):
                    Re_2pt_mean_ENV = meff_2pt_mean[P_indx, 0, Nev_indx, 0, 0, :, pos]
                    Re_2pt_err_ENV = meff_2pt_err[P_indx, 0, Nev_indx, 0, 0, :, pos]
                    ax.errorbar(
                        np.arange(Re_2pt_mean_ENV.shape[0]) - (len(data_info.keys())//2 + i) * 0.1,
                        Re_2pt_mean_ENV, 
                        yerr=Re_2pt_err_ENV, 
                        alpha=0.5, 
                        marker = self.marker_array[P_indx + i], 
                        capsize=3.5, 
                        capthick=1.5, 
                        label=f"{data_label[i]}", 
                        linestyle='none',
                        elinewidth=2
                        ) # fmt = 'bs'
            i += 1
            
        plt.legend()
        
        fig.savefig("%s/%s"%(self.save_path, '%s.pdf'%(save_name)))

    def plot_3pt(self, data_info:dict, y_range, Re_or_Im:str = 'Re', num:int = 1, name:str = None, save_name:str = None, data_label:list = ['data']):
        if not name : name = '3pt'
        if not save_name : save_name = '3pt'
        
        N_data_ratio = len(data_info.keys())

        if len(data_label) != N_data_ratio:
            raise print(f"data_label size:{len(data_label)}, data size:{N_data_ratio}")
        
        if Re_or_Im == 'Re':
            pos = 0

        elif Re_or_Im == 'Im':
            pos = 1

        plt.rcParams.update({'font.size':25})
        fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        ax.set_title(label='%s'%(name), fontdict={'fontsize':30,'fontweight':'light'})
        ax.set_ylim(y_range)
        ax.set_xlabel(f't/a')
        ax.set_ylabel('$C_{\mathrm{3pt}}$') # $E_{\mathrm{2pt}}$/Gev # (N_data-1, N_P, N_ENV, N_tsep, N_link, inp.Nt) (N_data-1, N_P, N_ENV, N_tsep, N_link)
        gap = 0

        for file_name in data_info.keys():

            data_link_mean_change = data_info[file_name]['data_3pt_mean']
            data_link_err_change = data_info[file_name]['data_3pt_err']
            tsep_list = data_info[file_name][f'tsep']
            
            label = data_label[gap]
            if data_info[file_name]['file_type'] == '.iog':
                N_ENV_change = range(1); rephase = 0; gap_symble = 1
                
            else:
                N_ENV_change = range(-num,0,1); rephase = gap - int(N_data_ratio/2); gap_symble = -1
                    
            if data_link_mean_change.shape[-3] != len(data_info[file_name]['link_list']): normal = int(len(data_info[file_name]['link_list'])//2)

            else : normal = 0

            link_max = len(data_info[file_name]['link_list'])//2

            for ENV_indx in N_ENV_change:
                for tsep_indx in range(len(tsep_list)):
                    ax.errorbar(
                        np.arange(data_link_mean_change[0, 0, ENV_indx, tsep_indx, link_max, :tsep_list[tsep_indx] + 1, pos].shape[-1]) + (rephase) * 0.1, 
                        data_link_mean_change[0, 0, ENV_indx, tsep_indx, link_max, :tsep_list[tsep_indx] + 1, pos], 
                        yerr=data_link_err_change[0, 0, ENV_indx, tsep_indx, link_max, :tsep_list[tsep_indx] + 1, pos], 
                        alpha=0.5, 
                        marker = self.marker_array[ENV_indx +  gap_symble * gap], 
                        capsize=3.5, 
                        capthick=1.5, 
                        label=f"{label}",
                        linestyle='none',
                        elinewidth=2
                        ) # fmt = 'bs' ,ENV_array_change[ENV_indx]
                    
            plt.legend()

            gap += 1

        fig.savefig("%s/%s"%(self.save_path, '%s.pdf'%(save_name)))

    def plot_link_ratio(self, data_info:dict, y_range, Re_or_Im:str = 'Re', num:int = 1, name:str = None, save_name:str = None, data_label:list = ['data']):

        if not name : name = 'link_ratio'
        if not save_name : save_name = 'link_ratio'
        
        N_data_ratio = len(data_info.keys())

        if len(data_label) != N_data_ratio:
            raise print(f"data_label size:{len(data_label)}, data size:{N_data_ratio}")
        
        if Re_or_Im == 'Re':
            pos = 0

        elif Re_or_Im == 'Im':
            pos = 1

        plt.rcParams.update({'font.size':25})
        fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        ax.set_title(label='%s'%(name), fontdict={'fontsize':30,'fontweight':'light'})
        ax.set_ylim(y_range)
        ax.set_xlabel('%s'%('Z'))
        ax.set_ylabel('%s'%('R = $C_{\mathrm{3pt}}$ / $C_{\mathrm{2pt}}$')) # $E_{\mathrm{2pt}}$/Gev # (N_data-1, N_P, N_ENV, N_tsep, N_link, inp.Nt) (N_data-1, N_P, N_ENV, N_tsep, N_link)
        gap = 0
        for file_name in data_info.keys():
            data_link_mean_change = data_info[file_name]['data_link_mean']
            data_link_err_change = data_info[file_name]['data_link_err']
            label = data_label[gap]
            if data_info[file_name]['file_type'] == '.iog':
                N_ENV_change = range(1); rephase = 0; gap_symble = 1
                
            else:
                N_ENV_change = range(-num,0,1); rephase = gap - int(N_data_ratio/2); gap_symble = -1
                    
            if data_link_mean_change.shape[-3] != len(data_info[file_name]['link_list']): normal = int(len(data_info[file_name]['link_list'])//2)

            else : normal = 0
            for ENV_indx in N_ENV_change:
                ax.errorbar(
                    np.arange(data_link_mean_change[0, 0, ENV_indx, ..., pos].shape[-1]) + (rephase) * 0.1, 
                    data_link_mean_change[0, 0, ENV_indx, 0, :, pos], 
                    yerr=data_link_err_change[0, 0, ENV_indx, 0, :, pos], 
                    alpha=0.5, 
                    marker = self.marker_array[ENV_indx +  gap_symble * gap], 
                    capsize=3.5, 
                    capthick=1.5, 
                    label=f"{label}",
                    linestyle='none',
                    elinewidth=2
                    ) # fmt = 'bs' ,ENV_array_change[ENV_indx]
                
                plt.legend()

            gap += 1

        fig.savefig("%s/%s"%(self.save_path, '%s.pdf'%(save_name)))

    def plot_link_ratio_AT(self, data_info:dict, y_range, link_indx:int = 0, data_label:list = ['data'], Re_or_Im:str = 'Re', num:int = 1, name:str = None, save_name:str = None):

        if not name : name = 'link_ratio'
        if not save_name : save_name = 'link_ratio'
        
        N_data_ratio = len(data_info.keys())

        if len(data_label) != N_data_ratio:
            raise print(f"data_label size:{len(data_label)}, data size:{N_data_ratio}")
        
        if Re_or_Im == 'Re':
            pos = 0

        elif Re_or_Im == 'Im':
            pos = 1

        plt.rcParams.update({'font.size':25})
        fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        ax.set_title(label='%s'%(name), fontdict={'fontsize':30,'fontweight':'light'})
        ax.set_ylim(y_range)
        ax.set_xlabel(f'tcurr/a')
        ax.set_ylabel('R = $C_{\mathrm{3pt}}$ / $C_{\mathrm{2pt}}$') # $E_{\mathrm{2pt}}$/Gev # (N_data-1, N_P, N_ENV, N_tsep, N_link, inp.Nt) (N_data-1, N_P, N_ENV, N_tsep, N_link)
        gap = 0

        for file_name in data_info.keys():

            data_link_mean_change = data_info[file_name]['data_PDF_mean']
            data_link_err_change = data_info[file_name]['data_PDF_err']
            tsep_list = data_info[file_name][f'tsep']
            
            label = data_label[gap]
            if data_info[file_name]['file_type'] == '.iog':
                N_ENV_change = range(1); rephase = 0; gap_symble = 1
                
            else:
                N_ENV_change = range(-num,0,1); rephase = gap - int(N_data_ratio/2); gap_symble = -1
                    
            if data_link_mean_change.shape[-3] != len(data_info[file_name]['link_list']): normal = int(len(data_info[file_name]['link_list'])//2)

            else : normal = 0

            link_max = len(data_info[file_name]['link_list'])//2 - normal + link_indx

            for ENV_indx in N_ENV_change:
                for tsep_indx in range(len(tsep_list)):
                    ax.errorbar(
                        np.arange(data_link_mean_change[0, 0, ENV_indx, tsep_indx, link_max, :tsep_list[tsep_indx] + 1, pos].shape[-1]) + (rephase) * 0.1, 
                        data_link_mean_change[0, 0, ENV_indx, tsep_indx, link_max, :tsep_list[tsep_indx] + 1, pos], 
                        yerr=data_link_err_change[0, 0, ENV_indx, tsep_indx, link_max, :tsep_list[tsep_indx] + 1, pos], 
                        alpha=0.5, 
                        marker = self.marker_array[ENV_indx +  gap_symble * gap], 
                        capsize=3.5, 
                        capthick=1.5, 
                        label=f"{label}",
                        linestyle='none',
                        elinewidth=2
                        ) # fmt = 'bs' ,ENV_array_change[ENV_indx]
                    
            plt.legend()

            gap += 1

        fig.savefig("%s/%s"%(self.save_path, '%s.pdf'%(save_name)))

    def plot_link_ratio_Atsep(self, data_info:dict, link_indx:int, y_range:list, Re_or_Im:str = 'Re', num:int=1, name:str=None, save_name:str=None, data_label:list=['data']):
        if not name : name = f'link_ratio_indx{link_indx}_Atsep'
        if not save_name : save_name = f'link_ratio_indx{link_indx}_Atsep'
        
        N_data_ratio = len(data_info.keys())
        gap = 0

        if Re_or_Im == 'Re':
            pos = 0

        elif Re_or_Im == 'Im':
            pos = 1

        if len(data_label) != N_data_ratio:
            raise print(f"data_label size:{len(data_label)}, data size:{N_data_ratio}")
        
        plt.rcParams.update({'font.size':25})
        fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        ax.set_title(label='%s'%(name), fontdict={'fontsize':30,'fontweight':'light'})
        ax.set_ylim(y_range)
        # ax.set_xlim([-(N_data_ratio*0.1 + 0.2) + link_indx, (N_data_ratio*0.1 + 0.2) + link_indx])
        ax.set_xlabel('%s'%('t'))
        ax.set_ylabel('%s'%('R = $C_{\mathrm{3pt}}$ / $C_{\mathrm{2pt}}$')) # $E_{\mathrm{2pt}}$/Gev # (N_data-1, N_P, N_ENV, N_tsep, N_link, inp.Nt) (N_data-1, N_P, N_ENV, N_tsep, N_link)
        
        for file_name in data_info.keys():
            data_PDF_mean_change = data_info[file_name]['data_PDF_mean']
            data_PDF_err_change = data_info[file_name]['data_PDF_err']
            tsep_list = data_info[file_name]['tsep']

            label = data_label[gap]
            if data_info[file_name]['file_type'] == '.iog':
                N_ENV_change = range(1); rephase = 0; gap_symble = 1
                
            else:
                N_ENV_change = range(-num,0,1); rephase = 1 + gap - int(N_data_ratio/2); gap_symble = -1
                    
            if data_PDF_mean_change.shape[-3] != len(data_info[file_name]['link_list']): normal = 0

            else : normal = int(len(data_info[file_name]['link_list'])//2)

            for ENV_indx in N_ENV_change:
                data = np.zeros(len(tsep_list), dtype = float)
                data_err = np.zeros(len(tsep_list), dtype = float)
                for tsep_indx in range(len(tsep_list)):
                    data[tsep_indx] = data_PDF_mean_change[0, 0, ENV_indx, tsep_indx, link_indx + normal, tsep_list[tsep_indx]//2, pos]
                    data_err[tsep_indx] = data_PDF_err_change[0, 0, ENV_indx, tsep_indx, link_indx + normal, tsep_list[tsep_indx]//2, pos]

                ax.errorbar(
                    np.arange(len(tsep_list)) + (rephase) * 0.1 + tsep_list[0], 
                    data, 
                    yerr=data_err, 
                    alpha=0.5, 
                    marker = self.marker_array[ENV_indx + gap * gap_symble], 
                    capsize=3.5, capthick=1.5, 
                    label=f"{label}_Nev{data_info[file_name]['Nev'][ENV_indx]}",
                    linestyle='none',
                    elinewidth=2
                    ) # fmt = 'bs' ,ENV_array_change[ENV_indx]
                
                plt.legend()
            
            gap += 1

        fig.savefig("%s/%s"%(self.save_path, '%s.pdf'%(save_name)))

    def plot_conserved_tsrc(self, data_info:dict, y_range,  Re_or_Im:str = 'Re', name:str=None, save_name:str=None, data_label:list=[]):
        if not name : name = 'link_ratio_conserved'
        if not save_name : save_name = 'link_ratio_conserved'
        
        N_data_ratio = len(data_info.keys())
        gap = 0

        if Re_or_Im == 'Re':
            pos = 0

        elif Re_or_Im == 'Im':
            pos = 1

        if len(data_label) != N_data_ratio:
            raise print(f"data_label size:{len(data_label)}, data size:{N_data_ratio}")
        
        plt.rcParams.update({'font.size':25})
        fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        ax.set_title(label='%s'%(name), fontdict={'fontsize':30,'fontweight':'light'})
        ax.set_ylim(y_range)
        ax.set_xlabel('%s'%('t_curr'))
        ax.set_ylabel('%s'%('R = $C_{\mathrm{3pt}}$ / $C_{\mathrm{2pt}}$')) # $E_{\mathrm{2pt}}$/Gev # (N_data-1, N_P, N_ENV, N_tsep, N_link, inp.Nt) (N_data-1, N_P, N_ENV, N_tsep, N_link)
        
        for file_name in data_info.keys():
            data_link_mean_change = data_info[file_name]['data_PDF_mean']
            data_link_err_change = data_info[file_name]['data_PDF_err']
            tsep_list = data_info[file_name]['tsep']
            
            label = data_label[gap]
            if data_info[file_name]['file_type'] == '.iog':
                rephase = 0; gap_symble = 1
                
            else:
                rephase = 1 + gap - int(N_data_ratio/2); gap_symble = -1

            for ENV_indx in [-1]:
                PDF_analyse = np.mean(
                    gv.gvar(
                        data_link_mean_change[0, 0, ENV_indx, 0, 0, 1:(tsep_list[0] - 1), pos].reshape(-1), 
                        data_link_err_change[0, 0, ENV_indx, 0, 0, 1:(tsep_list[0] - 1), pos].reshape(-1))
                            )
                
                print(
                    label + ' ' * len('_mean'), 
                    gv.gvar(
                        data_link_mean_change[0, 0, ENV_indx, 0, 0, 0:(tsep_list[0]), pos].reshape(-1), 
                        data_link_err_change[0, 0, ENV_indx, 0, 0, 0:(tsep_list[0]), pos].reshape(-1))
                            )
                
                mean = np.ones(tsep_list[0]) * gv.mean(PDF_analyse)
                err = np.ones(tsep_list[0]) * gv.sdev(PDF_analyse)
                print(label + '_mean', gv.gvar(mean, err))
                # print(err)
                print('\n')
                # print(PDF_y_change[:, ENV_indx, :, :, 0:(tsep_list[0])].shape)
                ax.errorbar(
                    np.arange((data_link_mean_change[0, 0, ENV_indx, :, :, 0:(tsep_list[0]), pos].reshape(-1)).shape[0]) + (rephase) * 0.1 + 1, 
                    data_link_mean_change[0, 0, ENV_indx, :, :, 0:(tsep_list[0]), pos].reshape(-1), 
                    yerr = data_link_err_change[0, 0, ENV_indx, :, :, 0:(tsep_list[0]), pos].reshape(-1), 
                    alpha = 0.5, marker = self.marker_array[ENV_indx + gap_symble * gap], capsize=3.5, capthick=1.5, label='%s'%(label), linestyle='none', elinewidth=2) # fmt = 'bs' ,ENV_array_change[ENV_indx]
                plt.plot(np.arange(tsep_list[0]) + 1, mean, label = f"{label}_'t_mean'")
                plt.fill_between(np.arange(tsep_list[0]) + 1, mean - err, mean + err, color='violet', alpha=0.2)
                plt.legend()

        fig.savefig("%s/%s"%(self.save_path, '%s.pdf'%(save_name)))

    def plot_fit(self, y_range, flavour, X, Y, name:str=None, save_name:str=None, normal:int=0, data_label:list=[], iog_label:list=[]):
        if not name : name = '%s_PDF_%dx%d_ENV_%s_fit'%(self.hadron, self.Nx, self.Nt, flavour)
        if not save_name : save_name = '%s_PDF_%dx%d_ENV_%s_fit'%(self.hadron, self.Nx, self.Nt, flavour)
        plt.rcParams.update({'font.size':25})
        fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        # ax.set_title(name,fontdict={'fontsize':30,'fontweight':'light'})
        ax.set_ylim(y_range)
        ax.set_xlabel('%s'%('N_ENV'))
        ax.set_ylabel('%s'%('R = $C_{\mathrm{3pt}}$/$C_{\mathrm{2pt}}$')) # $E_{\mathrm{2pt}}$/Gev
        for i in range(1):
            data_name = 'link_data_U'; N_ENV_change = self.ENV; label = 'distillation'
            for tsep_indx in range(self.N_tsep):
                ax.errorbar(N_ENV_change, self.link_y[data_name][0,:,tsep_indx,self.link_max_indx], yerr=self.link_err[data_name][0,:,tsep_indx,self.link_max_indx], alpha=0.5, marker = self.marker_array[tsep_indx], capsize=3.5, capthick=1.5, label='%s_tsep%d'%(label,self.tsep[tsep_indx]), linestyle='none',elinewidth=2) # fmt = 'bs'
                plt.legend(loc=2)
        for i in range(2):
            ax.errorbar(X, Y, yerr=np.zeros_like(X), alpha=0.5, marker = self.marker_array[tsep_indx], capsize=3.5, capthick=1.5, label='%s_tsep%d'%('fit',self.tsep[tsep_indx]), linestyle='none',elinewidth=2) # fmt = 'bs'
        
        fig.savefig("%s/%s"%(self.save_path, '%s.pdf'%(save_name)))

    # def plot_ENV_contribution(self, indx, y_range, N, name:str=None, save_name:str=None, normal:int=0, data_label:list=[], iog_label:list=[]):
    #     if not name : name = 'every_eigens_contribution'
    #     if not save_name : save_name = 'every_eigens_contribution'
    #     plt.rcParams.update({'font.size':25})
    #     fig, ax = plt.subplots(1,1, figsize=(20, 20*0.5))
    #     fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    #     ax.set_title(name, fontdict={'fontsize':30,'fontweight':'light'})
    #     ax.set_ylim(y_range)
    #     ax.set_xlabel('%s'%('ENV'))
    #     ax.set_ylabel('%s'%('contribution')) # $E_{\mathrm{2pt}}$/Gev # (N_data-1, N_P, N_ENV, N_tsep, N_link, inp.Nt) (N_data-1, N_P, N_ENV, N_tsep, N_link)
    #     for i in range(len(self.link_y.keys())):
    #         data_name = str(list(self.link_y.keys())[i])
    #         if data_name == 'link_iog_U': N_ENV_change = self.ENV[-1]; data_type = 'chroma';  N_ENV = 1; N=1
    #         elif data_name == 'link_data_D': N_ENV_change = self.ENV[:self.link_y[data_name].shape[1]]; data_type = 'dis_reduse'; N_ENV = self.N_ENV
    #         elif data_name == 'link_data_U': N_ENV_change = self.ENV[:self.link_y[data_name].shape[1]]; data_type = 'dis_none'; N_ENV = self.N_ENV
    #         if normal == 1:
    #             link_y_change = self.link_y[data_name] / self.link_y[data_name][...,self.link_max_indx].reshape(self.N_P,N_ENV,self.N_tsep,1)
    #             link_err_change = self.link_err[data_name] / self.link_y[data_name][...,self.link_max_indx].reshape(self.N_P,N_ENV,self.N_tsep,1)
    #         else:
    #             link_y_change = self.link_y[data_name]
    #             link_err_change = self.link_err[data_name]
    #         ax.errorbar(N_ENV_change, link_y_change[0,:,0, indx+self.link_max_indx]/N, yerr=link_err_change[0,:,0, indx+self.link_max_indx], alpha=0.5, marker = self.marker_array[i], capsize=3.5, capthick=1.5, label='%s_indx%d'%(data_type, indx),linestyle='none',elinewidth=2) # fmt = 'bs'
    #         plt.legend()
        
    #     fig.savefig("%s/%s"%(self.save_path, '%s.pdf'%(save_name)))
            
    # def min_fun_descent(function, X:np.ndarray, X0:np.ndarray, type:str):
    #     alpha = 1e-12
    #     delta = 1e-12
    #     max_cycle_index = 10000
    #     cycle_index = 0
        
    #     br = sp.symbols('br')
    #     N = X.shape[0]
    #     d_fun_value = np.zeros(N)
        
    #     d_fun = np.asarray([sp.diff(function, X[n]) for n in range(N)])
    #     if type == 'multinomial' :
    #         X0_dic = sp.solve([d_fun[n] for n in range(N)],[X[n]for n in range(N)])
    #         X0 = [X0_dic[indx] for indx in X]
    #     d_fun_value = np.asarray([(float)(d_fun[n].subs({X[i]: X0[i] for i in range(N)}).evalf())  for n in range(N)])
    #     while (cycle_index <= max_cycle_index and sum(abs(d_value) for d_value in d_fun_value ) >= delta):
    #         cycle_index += 1
    #         X0_mid = X0 - d_fun_value * br
    #         # X0_mid = [ X0[n] - d_fun_value[n] * br  for n in range(N)] # * np.exp(abs(d_fun_value_2)-abs(dd_fun_value_2)) # * u
    #         beta = float(abs(sp.solve((function.subs({X[i]: X0_mid[i] for i in range(N)}).evalf()))[0]))
    #         X0 = X0 - d_fun_value * beta * alpha 
    #         print('beta:',beta)
    #         d_fun_value = np.asarray([(float)(d_fun[n].subs({X[i]: X0[i] for i in range(N)}).evalf()) for n in range(N)])
    #         function_value = (float)(function.subs({X[i]: X0[i] for i in range(N)}).evalf())
    #         print('cycle_index:%d'%(cycle_index))
    #         print('X0:',X0)
    #         print('d_fun_value:',d_fun_value)
    #         print('function_value:%f'%(function_value))
    #         print('\n')
    #     function_value = (float)(function.subs({X[i]: X0[i] for i in range(N)}).evalf())
    #     print(function_value)
    #     return X0, function_value


    # def min_fun_newton(function, X_1, X_2, X0_1, X0_2):
    #     max_cycle_index = 200
    #     alpha = 0.01
    #     delta = 0.0001
    #     cycle_index = 0
        
    #     d_fun_1 = sp.diff(function, X_1)[0]
    #     d_fun_2 = sp.diff(function, X_2)[0]
        
    #     dd_fun_1  = sp.diff(function, X_1, 2)[0]
    #     dd_fun_12 = sp.diff(function, X_1, X_2)[0]
    #     dd_fun_2  = sp.diff(function, X_2, 2)[0]
        
    #     d_fun_value_1 = (float)(d_fun_1.subs({X_1: X0_1, X_2: X0_2}).evalf()) 
    #     d_fun_value_2 = (float)(d_fun_2.subs({X_1: X0_1, X_2: X0_2}).evalf())
        
    #     dd_fun_value_1  = (float)(dd_fun_1.subs({X_1: X0_1, X_2: X0_2}).evalf())
    #     dd_fun_value_2  = (float)(dd_fun_2.subs({X_1: X0_1, X_2: X0_2}).evalf())
    #     dd_fun_value_12 = (float)(dd_fun_12.subs({X_1: X0_1, X_2: X0_2}).evalf())
        
    #     while (cycle_index <= max_cycle_index and abs(d_fun_value_1)+abs(d_fun_value_2) >= delta):
    #         cycle_index += 1
            
    #         G = np.vstack((d_fun_value_1,d_fun_value_2))
    #         H = np.vstack(([dd_fun_value_1,dd_fun_value_12],[dd_fun_value_12,dd_fun_value_2]))
            
            
    #         z_value = np.vstack((X0_1, X0_2))-np.linalg.inv(H)@G
    #         X0_1 = (float)(z_value[0])
    #         X0_2 = (float)(z_value[1])
            
    #         d_fun_value_1 = (float)(d_fun_1.subs({X_1: X0_1, X_2: X0_2}).evalf()) 
    #         d_fun_value_2 = (float)(d_fun_2.subs({X_1: X0_1, X_2: X0_2}).evalf())
        
    #         dd_fun_value_1  = (float)(dd_fun_1.subs({X_1: X0_1, X_2: X0_2}).evalf())
    #         dd_fun_value_2  = (float)(dd_fun_2.subs({X_1: X0_1, X_2: X0_2}).evalf())
    #         dd_fun_value_12 = (float)(dd_fun_12.subs({X_1: X0_1, X_2: X0_2}).evalf())
            
    #         print('cycle_index:%d\n X0_1:%f X0_2:%f\n d_fun_value_1:%f  d_fun_value_2:%f\n dd_fun_value_1:%f  dd_fun_value_2:%f\n'%(cycle_index,X0_1,X0_2,d_fun_value_1,d_fun_value_2, dd_fun_value_1, dd_fun_value_2))
            
    #     function_value = (float)(function[0].subs({X_1: X0_1, X_2: X0_2}).evalf())
    #     parameter = np.array([X0_1, X0_2, function_value])
    #     return parameter


    # def c_square(data, function, X_1, X_2, X0_1, X0_2):
    #     n = data.shape[0]
    #     t = data.shape[1]
    #     c_square_ni = 0
    #     c_square_ni = np.zeros((n,3))
        
    #     cov_m_A = np.cov(data.T)
    #     cov_m_inv = np.linalg.inv(cov_m_A)
        
    #     c_square_fun = ((data[0,:].reshape(t,1)-function).T)@cov_m_inv@(data[0,:].reshape(t,1)-function)    
    #     c_square_ni[0] = min_fun_newton(c_square_fun, X_1, X_2, X0_1, X0_2)
        
    #     for ni in range(1,n):
    #         c_square_fun = ((data[ni,:].reshape(t,1)-function).T)@cov_m_inv@(data[ni,:].reshape(t,1)-function)    
    #         c_square_ni[ni] = min_fun_newton(c_square_fun, X_1, X_2, c_square_ni[ni-1, 0], c_square_ni[ni-1, 1])
        
    #     c_square_mean = np.sum(c_square_ni, axis = 0)/n
    #     return c_square_mean




    # Para = min_fun(fit_function, )

    # def dispersion_fit()
    # def plt():
        # figure = plt.figure()
        # axes = plt.axes(projection = '3d')
        # x = np.arange(0,2,0.01)
        # y = np.arange(0,1.5,0.01)
        # z=np.zeros((x.size,y.size))
        # for i in range(x.size):
        #     for j in range(y.size):
                
        #         z[i,j] = (float)(c_square_fun[0].subs({X_1: x[i], X_2: y[j]}).evalf())
        # x,y = np.meshgrid(x,y)
        # axes.plot_surface(x,y,z.T,cmap='rainbow')
        # plt.savefig("./c_square.pdf")