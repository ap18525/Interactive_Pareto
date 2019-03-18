# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:37:11 2019

@author: ap18525
"""

import numpy as np
from bqplot import pyplot as plt
from bqplot import *
from bqplot.traits import *
import ipywidgets as widgets

def Interactive_Pareto(T,members_num,population_size,I_for,E_for,d_for,
                       S0,Smax,env_min,
                       results1_optim_relea,results2_optim_relea,solutions_optim_relea,sdpen):

    def update_operation(i):
        S,env,w,r    = syst_sim(T,I_for+solutions_optim_relea[i],E_for,d_for,S0,Smax,env_min)
        fig_wd.title = 'Supply deficit - Probability = {:.0f}'.format(np.max(np.count_nonzero(d_for-r,axis =0)))+' / '+str(members_num)
        fig_pi.title = 'Pumped inflow - Energy cost = £{:.0f}'.format(results2_optim_relea[i])
        return       S,solutions_optim_relea[i],r,results1_optim_relea[i],results2_optim_relea[i],i
    
    def solution_selected(change):
        if pareto_front.selected == None:
            pareto_front.selected = [0]
        storage.y = update_operation(pareto_front.selected[0])[0]
        deficit.y = np.maximum(d_for-update_operation(pareto_front.selected[0])[2],np.zeros(np.shape(d_for)))
        pinflows.y = update_operation(pareto_front.selected[0])[1]
        pareto_front_ensemble.x = np.reshape([results2_optim_relea for i in range(0, members_num)],(members_num,population_size))[:,pareto_front.selected[0]]
        pareto_front_ensemble.y = sdpen[:,pareto_front.selected[0]]
        pareto_front_ensemble.unselected_style={'opacity': 0.1}
        pareto_front_ensemble.selected_style={'opacity': 0.1}
        pareto_front_ensemble.opacity = [0.1]*10
        
    x_sc_pf = LinearScale()
    y_sc_pf = LinearScale(min = 0,max = 50)
    
    x_ax_pf = Axis(label='Pumping energy cost [£]', scale=x_sc_pf)
    y_ax_pf = Axis(label='Supply deficit [ML]', scale=y_sc_pf, orientation='vertical')
    
    pareto_front = plt.scatter(results2_optim_relea[:],results1_optim_relea[:],scales={'x': x_sc_pf, 'y': y_sc_pf},colors=['deepskyblue'], interactions={'hover':'tooltip','click': 'select'})
    pareto_front.unselected_style={'opacity': 0.8}
    pareto_front.selected_style={'fill': 'red', 'stroke': 'yellow', 'width': '1125px', 'height': '125px'}
    
    if pareto_front.selected == []:
        pareto_front.selected = [0]
        
    pareto_front_ensemble = plt.Scatter(x=np.reshape([results2_optim_relea for i in range(0, members_num)],(members_num,population_size))[:,pareto_front.selected[0]],
                                        y=sdpen[:,pareto_front.selected[0]],scales={'x': x_sc_pf, 'y': y_sc_pf},
                                        colors=['red'], interactions={'hover':'tooltip','click': 'select'})
    pareto_front_ensemble.unselected_style={'opacity': 0.1}
    pareto_front_ensemble.selected_style={'opacity': 0.1}
    pareto_front_ensemble.opacity = [0.1]*10
    fig_pf = plt.Figure(marks=[pareto_front,pareto_front_ensemble],title = 'Pareto front', axes=[x_ax_pf, y_ax_pf],layout={'width': '500px', 'height': '500px'}, 
                        animation_duration=500)
    
    pareto_front.observe(solution_selected,'selected')    
    
    S,env,w,r    = syst_sim(T,I_for+solutions_optim_relea[pareto_front.selected[0]],E_for,d_for,S0,Smax,env_min)
    
    x_sc_pi    = OrdinalScale(min=1,max=T);y_sc_pi = LinearScale(min=0,max=40); x_ax_pi = Axis(label='week', scale=x_sc_pi);                              y_ax_pi = Axis(label='ML/week', scale=y_sc_pi, orientation='vertical')
    x_sc_st    = LinearScale(min=0,max=T); y_sc_st = LinearScale(min=10,max=160);x_ax_st = Axis(label='week', scale=x_sc_st,tick_values=[0.5,1.5,2.5,3.5]);y_ax_st = Axis(label='ML', scale=y_sc_st, orientation='vertical')
    x_sc_wd    = OrdinalScale(min=1,max=T);y_sc_wd = LinearScale(min=0,max=60); x_ax_wd = Axis(label='week', scale=x_sc_wd);                              y_ax_wd = Axis(label='ML/week', scale=y_sc_wd, orientation='vertical')
    
    pinflows = plt.bar(np.arange(1,T+1),solutions_optim_relea[pareto_front.selected[0]],scales={'x': x_sc_pi, 'y': y_sc_pi},
                                   colors=['orange'],opacities = [1],stroke = 'lightgray',
                                   labels = ['pumped inflow'], display_legend = False)
    
    fig_pi   = plt.Figure(marks = [pinflows],axes=[x_ax_pi, y_ax_pi],layout={'max_width': '480px', 'max_height': '250px'},
                        scales={'x': x_sc_pi, 'y': y_sc_pi}, animation_duration=1000,legend_location = 'bottom-right')
    
    storage           = plt.plot(x=np.arange(0,T+1),y=S,scales={'x': x_sc_st, 'y': y_sc_st},
                                  colors=['blue'], stroke_width = 0.1,
                                  fill = 'bottom', fill_opacities = [0.1]*members_num)
    max_storage       = plt.plot(x=np.arange(0,T+1),y=[Smax]*(T+1),colors=['red'],scales={'x': x_sc_st, 'y': y_sc_st})
    max_storage_label = plt.label(text = ['Max storage'], x=[0],y=[Smax+10],colors=['red'])
    fig_st            = plt.Figure(marks = [storage,max_storage,max_storage_label], title = 'Reservoir storage volume', 
                         axes=[x_ax_st, y_ax_st],layout={'width': '1000px', 'height': '350px'}, animation_duration=1000,scales={'x': x_sc_st, 'y': y_sc_st})
    
    vertical_lines_wd = plt.vline([1,2,3],colors = ['black'])
    
    deficit = plt.bar(np.arange(1,T+1),np.maximum(d_for-r,np.zeros(np.shape(r))),scales={'x': x_sc_wd, 'y': y_sc_wd},
                                    colors=['red'],opacities = [0.7]*members_num*T,stroke = 'lightgray',
                                    labels = ['release'], display_legend = False,type = 'grouped',base=0,align='center')
    fig_wd = plt.Figure(marks = [deficit,vertical_lines_wd],axes=[x_ax_wd, y_ax_wd],layout={'max_width': '480px', 'max_height': '250px'},
                        scales={'x': x_sc_wd, 'y': y_sc_wd}, animation_duration=1000,legend_location = 'bottom-right')
    
    storage.y  = update_operation(pareto_front.selected[0])[0]
    deficit.y  = np.maximum(d_for-update_operation(pareto_front.selected[0])[2],np.zeros(np.shape(d_for)))
    pinflows.y = update_operation(pareto_front.selected[0])[1]
    
    storage.observe(solution_selected, ['x', 'y'])
    deficit.observe(solution_selected, ['x', 'y'])
    pinflows.observe(solution_selected, ['x', 'y'])
    
    return fig_pf, fig_pi, fig_wd, fig_st