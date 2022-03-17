#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 09:15:49 2020

@author: ckittel
"""


import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from matplotlib import ticker as mticker



def fig_ts(reg,what="SMB"):
    '''
    reg=name of the reg for the plot => ice, grd, shelf
    '''
    
    AC=result_dic[reg]['AC3']
    AC_ano=result_dic_ano[reg]['AC3']
    
    NOR=result_dic[reg]['NOR']
    NOR_ano=result_dic_ano[reg]['NOR']
    
    CNRM=result_dic[reg]['CN6']
    CNRM_ano=result_dic_ano[reg]['CN6']

    CESM=result_dic[reg]['CSM']
    CESM_ano=result_dic_ano[reg]['CSM']
   
    pal = sns.color_palette("PuBu",10)
    pal_or = sns.color_palette("OrRd",10)

    lw2=1.5
    lws=1.8
    
    nrow=3
    ncol=2    
    fig, ax2 = plt.subplots(nrows=nrow,ncols=ncol,figsize=(14,16))
        
    idi=0
    idj=0
    
    ymin=-1000
    ymax=2500
    if what=='flux':
        ymin=-30
        ymax=30
#    if reg == 'shelf':
#        ymin=-1000
#        ymax=1500
#    if reg == 'grd':
#        ymin=-200
#        ymax=1200
    
    for var in range(0,len(vars_SMB)):
        AC_ano[vars_SMB[var]].plot(ax=ax2[idi,idj],color=pal[7], label="MAR(AC3)",lw=lws)
        NOR_ano[vars_SMB[var]].plot(ax=ax2[idi,idj],color=pal[4], label="MAR(NOR)",lw=lws)
        CNRM_ano[vars_SMB[var]].plot(ax=ax2[idi,idj],color=pal_or[7], label="MAR(CN6)",lw=lws)
        CESM_ano[vars_SMB[var]].plot(ax=ax2[idi,idj],color=pal_or[4], label="MAR(CSM)",lw=lws)  
        
        ax2[idi,idj].set_xlabel('Year', fontsize=12)
        
        ax2[idi,idj].set_ylabel(vars_SMB[var]+' anomaly (Gt yr$^{-1}$)', fontsize=14)
        ax2[idi,idj].tick_params(axis="x", labelsize=12) 
        ax2[idi,idj].tick_params(axis="y", labelsize=12)
        ax2[idi,idj].yaxis.set_minor_locator(mticker.AutoMinorLocator(n=2))
        ax2[idi,idj].tick_params(axis='y',which='major',direction='inout',left=True,right=False)
        ax2[idi,idj].tick_params(axis='y',which='minor',direction='in',left=True,right=False)
        ax2[idi,idj].tick_params(axis='x',which='major',direction='inout',top=False,bottom=True)
        ax2[idi,idj].set_ylim(ymin,ymax)

        ax2[idi,idj].axhline(color='black', alpha=1,lw=1)
        ax2[idi,idj].set_title("", fontsize=0)
#        ax2[idi,idj].set_ylabel('', fontsize=0)
        
        idj=idj+1
        if idj== ncol:
            idj=0
            idi=idi+1
    
    ax2[0,0].legend(frameon=False, prop={'size': 14})
          
#   
    plt.tight_layout() 
#    sns.despine()
    plt.show()
    
    fig.savefig('./Proj_TS_all'+reg+what+'.png',format='PNG', dpi=300)
    
def fig_ts_maj(reg,what="SMB"):
    '''
    reg=name of the reg for the plot => ice, grd, shelf
    '''
    
    AC=result_dic[reg]['AC3']
    AC_ano=result_dic_ano[reg]['AC3']
    
    NOR=result_dic[reg]['NOR']
    NOR_ano=result_dic_ano[reg]['NOR']
    
    CNRM=result_dic[reg]['CNRM']
    CNRM_ano=result_dic_ano[reg]['CNRM']

    CESM=result_dic[reg]['CESM']
    CESM_ano=result_dic_ano[reg]['CESM']
   
    pal = sns.color_palette("PuBu",10)
    pal_or = sns.color_palette("OrRd",10)

    lw2=1.5
    lws=1.8
    
    nrow=2
    ncol=2    
    fig, ax2 = plt.subplots(nrows=nrow,ncols=ncol,figsize=(14,10))
        
    idi=0
    idj=0
    
    ymin=-1000
    ymax=2500
#    if reg == 'shelf':
#        ymin=-1000
#        ymax=1500
#    if reg == 'grd':
#        ymin=-200
#        ymax=1200

    for var in range(0,len(vars_SMB)-2):
        AC_ano[vars_SMB[var]].plot(ax=ax2[idi,idj],color=pal[7], label="MAR(AC3)",lw=lws)
        NOR_ano[vars_SMB[var]].plot(ax=ax2[idi,idj],color=pal[4], label="MAR(NOR)",lw=lws)
        CNRM_ano[vars_SMB[var]].plot(ax=ax2[idi,idj],color=pal_or[7], label="MAR(CN6)",lw=lws)
        CESM_ano[vars_SMB[var]].plot(ax=ax2[idi,idj],color=pal_or[4], label="MAR(CSM)",lw=lws)  
        
        ax2[idi,idj].set_xlabel('Year', fontsize=14)
        
        ax2[idi,idj].set_ylabel(vars_SMB[var]+' anomaly (Gt yr$^{-1}$)', fontsize=16)
        ax2[idi,idj].tick_params(axis="x", labelsize=14) 
        ax2[idi,idj].tick_params(axis="y", labelsize=14)
        ax2[idi,idj].yaxis.set_minor_locator(mticker.AutoMinorLocator(n=2))
        ax2[idi,idj].tick_params(axis='y',which='major',direction='inout',left=True,right=False)
        ax2[idi,idj].tick_params(axis='y',which='minor',direction='in',left=True,right=False)
        ax2[idi,idj].tick_params(axis='x',which='major',direction='inout',top=False,bottom=True)
        ax2[idi,idj].set_ylim(ymin,ymax)

        ax2[idi,idj].axhline(color='black', alpha=1,lw=1)
        ax2[idi,idj].set_title("", fontsize=0)
#        ax2[idi,idj].set_ylabel('', fontsize=0)
        
        idj=idj+1
        if idj== ncol:
            idj=0
            idi=idi+1
    
    ax2[0,0].legend(frameon=False, prop={'size': 16})
    
    plt.tight_layout() 
#    sns.despine()
    plt.show()
    fig.savefig('./Proj_TS_maj'+reg+what+'.png',format='PNG', dpi=300)
    
def fig_ts_sf_pe_ice(reg,what="SMB"):   
    pal = sns.color_palette("PuBu",10)
    pal_or = sns.color_palette("OrRd",10)
    lw2=1.5
    lws=1.8
    
    AC=result_dic[reg]['AC3']
    AC_ano=result_dic_ano[reg]['AC3']
    
    NOR=result_dic[reg]['NOR']
    NOR_ano=result_dic_ano[reg]['NOR']
    
    CNRM=result_dic[reg]['CNRM']
    CNRM_ano=result_dic_ano[reg]['CNRM']

    CESM=result_dic[reg]['CESM']
    CESM_ano=result_dic_ano[reg]['CESM']
    
    nrow=2
    ncol=2    
    fig, ax2 = plt.subplots(nrows=nrow,ncols=ncol,figsize=(14,10))
        
    idi=0
    idj=0
    
    AC['SMB'].rolling(TIME=5, center=True).mean().plot(ax=ax2[0,0],color=pal[7], label="MAR(AC3)",lw=lws)
    NOR['SMB'].rolling(TIME=5, center=True).mean().plot(ax=ax2[0,0],color=pal[4], label="MAR(NOR)",lw=lws)
    CNRM['SMB'].rolling(TIME=5, center=True).mean().plot(ax=ax2[0,0],color=pal_or[7], label="MAR(CN6)",lw=lws)
    CESM['SMB'].rolling(TIME=5, center=True).mean().plot(ax=ax2[0,0],color=pal_or[4], label="MAR(CSM)",lw=lws) 
    

    ruAC=AC['SMB']+AC['RU']
    ruNOR=NOR['SMB']+NOR['RU']
    ruCN=CNRM['SMB']+CNRM['RU']
    ruCE=CESM['SMB']+CESM['RU']
    
    ruAC.rolling(TIME=5, center=True).mean().plot(ax=ax2[0,1],color=pal[7], label="MAR(AC3)",lw=lws)
    ruNOR.rolling(TIME=5, center=True).mean().plot(ax=ax2[0,1],color=pal[4], label="MAR(NOR)",lw=lws)
    ruCN.rolling(TIME=5, center=True).mean().plot(ax=ax2[0,1],color=pal_or[7], label="MAR(CN6)",lw=lws)
    ruCE.rolling(TIME=5, center=True).mean().plot(ax=ax2[0,1],color=pal_or[4], label="MAR(CSM)",lw=lws)

    
    ruAC=AC_ano['SMB']+AC_ano['RU']-AC_ano['SMB']
    ruNOR=NOR_ano['SMB']+NOR_ano['RU']-NOR_ano['SMB']
    ruCN=CNRM_ano['SMB']+CNRM_ano['RU']- CNRM_ano['SMB']
    ruCE=CESM_ano['SMB']+CESM_ano['RU']-CESM_ano['SMB']
    
    ruAC.plot(ax=ax2[1,0],color=pal[7], label="MAR(AC3)",lw=lws)
    ruNOR.plot(ax=ax2[1,0],color=pal[4], label="MAR(NOR)",lw=lws)
    ruCN.plot(ax=ax2[1,0],color=pal_or[7], label="MAR(CN6)",lw=lws)
    ruCE.plot(ax=ax2[1,0],color=pal_or[4], label="MAR(CSM)",lw=lws)
    
    
    AC_ano['RU'].cumsum().plot(ax=ax2[1,1],color=pal[7], label="MAR(AC3)",lw=lws)
    NOR_ano['RU'].cumsum().plot(ax=ax2[1,1],color=pal[4], label="MAR(NOR)",lw=lws)
    CNRM_ano['RU'].cumsum().plot(ax=ax2[1,1],color=pal_or[7], label="MAR(CN6)",lw=lws)
    CESM_ano['RU'].cumsum().plot(ax=ax2[1,1],color=pal_or[4], label="MAR(CSM)",lw=lws)    

    for var in range(0,2):     
        ax2[idi,idj].set_xlabel('Year', fontsize=12)
        
        ax2[idi,idj].tick_params(axis="x", labelsize=12) 
        ax2[idi,idj].tick_params(axis="y", labelsize=12)
        ax2[idi,idj].yaxis.set_minor_locator(mticker.AutoMinorLocator(n=2))
        ax2[idi,idj].tick_params(axis='y',which='major',direction='inout',left=True,right=True)
        ax2[idi,idj].tick_params(axis='y',which='minor',direction='in',left=True,right=True)
        ax2[idi,idj].tick_params(axis='x',which='major',direction='inout',top=True,bottom=True)
        ax2[idi,idj].set_ylim(1500,4600)
        if reg == 'shelf':
            ax2[idi,idj].set_ylim(-1000,1000)

        ax2[idi,idj].axhline(color='black', alpha=1,lw=1)
        ax2[idi,idj].set_title("", fontsize=0)
#        ax2[idi,idj].set_ylabel('', fontsize=0)
        
        idj=idj+1
        if idj== ncol:
            idj=0
            idi=idi+1
            
            
    for var in range(2,3):     
        ax2[idi,idj].set_xlabel('Year', fontsize=12)
        ax2[idi,idj].tick_params(axis="x", labelsize=12) 
        ax2[idi,idj].tick_params(axis="y", labelsize=12)
        ax2[idi,idj].yaxis.set_minor_locator(mticker.AutoMinorLocator(n=2))
        ax2[idi,idj].tick_params(axis='y',which='major',direction='inout',left=True,right=True)
        ax2[idi,idj].tick_params(axis='y',which='minor',direction='in',left=True,right=True)
        ax2[idi,idj].tick_params(axis='x',which='major',direction='inout',top=True,bottom=True)
        ax2[idi,idj].set_ylim(-250,2500)


        ax2[idi,idj].axhline(color='black', alpha=1,lw=1)
        ax2[idi,idj].set_title("", fontsize=0)
#        ax2[idi,idj].set_ylabel('', fontsize=0)
        
        idj=idj+1
        if idj== ncol:
            idj=0
            idi=idi+1            
    
    ax2[0,0].legend(frameon=False, prop={'size': 14})
    ax2[0,0].set_ylabel('SMB (Gt yr$^{-1}$)', fontsize=14)
    ax2[0,1].set_ylabel('P-E (Gt yr$^{-1}$)', fontsize=14)
    
    ax2[1,0].set_ylabel('Difference (P-E - SMB) (Gt yr$^{-1}$)', fontsize=14)
    ax2[1,1].set_ylabel('Cumulated difference (P-E - SMB) (Gt)', fontsize=14)
    ax2[1,1].set_title("", fontsize=0)
    ax2[1,1].axhline(color='black', alpha=1,lw=1)
    ax2[1,1].set_ylim(-250,30000)
#        ax2[idi,idj].set_ylabel('', fontsize=0)    
#   
    plt.tight_layout() 
    
    plt.show()
    fig.savefig('./Proj_TS_p-e_'+reg+what+'.png',format='PNG', dpi=300)
  
    
def fig_comb_maj(what="SMB"):
    '''
    reg=name of the reg for the plot => ice, grd, shelf
    '''
    list_letter   = ['A','C','E','G','B','D','F','H'] 
    pal = sns.color_palette("PuBu",10)
    pal_or = sns.color_palette("OrRd",10)

    lw2=1.5
    lws=1.8
    
    nrow=4
    ncol=2    
    fig, ax2 = plt.subplots(nrows=nrow,ncols=ncol,figsize=(14,16))
        

    idj=0 
    ii=0
    reg_list=["grd","shelf"]
    
    for reg in reg_list:
        
        idi=0
        
        AC=result_dic[reg]['AC3']
        AC_ano=result_dic_ano[reg]['AC3']
        
        NOR=result_dic[reg]['NOR']
        NOR_ano=result_dic_ano[reg]['NOR']
        
        CNRM=result_dic[reg]['CN6']
        CNRM_ano=result_dic_ano[reg]['CN6']
    
        CESM=result_dic[reg]['CSM']
        CESM_ano=result_dic_ano[reg]['CSM']
       
        ymin=-1000
        ymax=1500
        if what=='flux':
            ymin=-50
            ymax=50
        
        
        if reg == 'shelf':
            idj=1
#        ymin=-1000
#        ymax=1500
        if reg == 'grd':
            idj=0
#        ymin=-200
#        ymax=1200

        for var in range(0,len(vars_SMB)-2):
            print (var)
            AC_ano[vars_SMB[var]].plot(ax=ax2[idi,idj],color=pal[7], label="MAR(ACCESS1.3)",lw=lws)
            NOR_ano[vars_SMB[var]].plot(ax=ax2[idi,idj],color=pal[4], label="MAR(NorESM1-M)",lw=lws)
            CNRM_ano[vars_SMB[var]].plot(ax=ax2[idi,idj],color=pal_or[7], label="MAR(CNRM-CM6-1)",lw=lws)
            CESM_ano[vars_SMB[var]].plot(ax=ax2[idi,idj],color=pal_or[4], label="MAR(CESM2)",lw=lws)  
            
            ax2[idi,idj].set_xlabel('', fontsize=0)
            
            if vars_SMB[var] == "AL2":
                ymin=-0.1
                ymax=0.1
            
            print(ymin,ymax)
            
            varl=vars_SMB[var]
            if vars_SMB[var] == 'SF':
                varl='Snowfall'
            if vars_SMB[var] == 'RF':
                varl='Rainfall'
            if vars_SMB[var] == 'SU':
                varl='Sublimation'
            if vars_SMB[var] == 'ME':
                varl='Melt'
            if vars_SMB[var] == 'RU':
                varl='Runoff' 
        
            
            ax2[idi,idj].set_ylabel(varl+' anomaly (Gt yr$^{-1}$)', fontsize=16)
            ax2[idi,idj].tick_params(axis="x", labelsize=14) 
            ax2[idi,idj].tick_params(axis="y", labelsize=14)
            ax2[idi,idj].yaxis.set_minor_locator(mticker.AutoMinorLocator(n=2))
            ax2[idi,idj].tick_params(axis='y',which='major',direction='inout',left=True,right=False)
            ax2[idi,idj].tick_params(axis='y',which='minor',direction='in',left=True,right=False)
            ax2[idi,idj].tick_params(axis='x',which='major',direction='inout',top=False,bottom=True)
            

            ax2[idi,idj].set_ylim(ymin,ymax)
    
            ax2[idi,idj].axhline(color='black', alpha=1,lw=1)
            ax2[idi,idj].set_title("", fontsize=0)
    #        ax2[idi,idj].set_ylabel('', fontsize=0)
    
            ax2[idi,idj].text(0.1, 0.1, list_letter[ii],fontsize=18, transform=ax2[idi,idj].transAxes, fontdict={'weight': 'bold'})
            
            idi=idi+1
            ii+=1

    ax2[0,0].legend(frameon=False, prop={'size': 16})
    ax2[0,0].set_title("Grounded ice", fontsize=18)
    ax2[0,1].set_title("Ice shelves", fontsize=18)
    plt.tight_layout() 
#    sns.despine()
    plt.show()
    fig.savefig('./fig/Fig_2combined_maj'+what+'.png',format='PNG', dpi=600)    
    
    
    
def fig_comb_all(what="SMB"):
    '''
    reg=name of the reg for the plot => ice, grd, shelf
    '''
    list_letter   = ['A','C','E','G','I','K','B','D','F','H','J','L'] 
    
    pal = sns.color_palette("PuBu",10)
    pal_or = sns.color_palette("OrRd",10)

    lw2=1.5
    lws=1.8
    
    nrow=6
    ncol=2    
    fig, ax2 = plt.subplots(nrows=nrow,ncols=ncol,figsize=(14,20))
        
    ii=0
    idj=0    
    reg_list=["grd","shelf"]
    
    for reg in reg_list:
        
        idi=0
        
        AC=result_dic[reg]['AC3']
        AC_ano=result_dic_ano[reg]['AC3']
        
        NOR=result_dic[reg]['NOR']
        NOR_ano=result_dic_ano[reg]['NOR']
        
        CNRM=result_dic[reg]['CN6']
        CNRM_ano=result_dic_ano[reg]['CN6']
    
        CESM=result_dic[reg]['CSM']
        CESM_ano=result_dic_ano[reg]['CSM']
       
        ymin=-1000
        ymax=1500
        if what=='flux':
            ymin=-50
            ymax=50
        
        
        if reg == 'shelf':
            idj=1
#        ymin=-1000
#        ymax=1500
        if reg == 'grd':
            idj=0
#        ymin=-200
#        ymax=1200

        for var in range(0,len(vars_SMB)):
            print (var)
            AC_ano[vars_SMB[var]].plot(ax=ax2[idi,idj],color=pal[7], label="MAR(ACCESS1.3)",lw=lws)
            NOR_ano[vars_SMB[var]].plot(ax=ax2[idi,idj],color=pal[4], label="MAR(NorESM1-M)",lw=lws)
            CNRM_ano[vars_SMB[var]].plot(ax=ax2[idi,idj],color=pal_or[7], label="MAR(CNRM-CM6-1)",lw=lws)
            CESM_ano[vars_SMB[var]].plot(ax=ax2[idi,idj],color=pal_or[4], label="MAR(CESM2)",lw=lws)  
            
            ax2[idi,idj].set_xlabel('', fontsize=0)
            
            if vars_SMB[var] == "AL2":
                ymin=-0.1
                ymax=0.1
            
            print(ymin,ymax)
            
            ax2[idi,idj].set_ylabel(vars_SMB[var]+' anomaly (Gt yr$^{-1}$)', fontsize=16)
            ax2[idi,idj].tick_params(axis="x", labelsize=14) 
            ax2[idi,idj].tick_params(axis="y", labelsize=14)
            ax2[idi,idj].yaxis.set_minor_locator(mticker.AutoMinorLocator(n=2))
            ax2[idi,idj].tick_params(axis='y',which='major',direction='inout',left=True,right=False)
            ax2[idi,idj].tick_params(axis='y',which='minor',direction='in',left=True,right=False)
            ax2[idi,idj].tick_params(axis='x',which='major',direction='inout',top=False,bottom=True)
            

            ax2[idi,idj].set_ylim(ymin,ymax)
            

    
            ax2[idi,idj].axhline(color='black', alpha=1,lw=1)
            ax2[idi,idj].set_title("", fontsize=0)
    #        ax2[idi,idj].set_ylabel('', fontsize=0)
            
            ax2[idi,idj].text(0.1, 0.1, list_letter[ii],fontsize=18, transform=ax2[idi,idj].transAxes, fontdict={'weight': 'bold'})
    
            idi=idi+1
            ii=ii+1

    
    ax2[0,0].legend(frameon=False, prop={'size': 16})
    ax2[0,0].set_title("Grounded ice", fontsize=18)
    ax2[0,1].set_title("Ice shelves", fontsize=18)
    plt.tight_layout() 
#    sns.despine()
    plt.show()
    fig.savefig('./fig/Fig_sup_combined_all'+what+'.png',format='PNG', dpi=600)    