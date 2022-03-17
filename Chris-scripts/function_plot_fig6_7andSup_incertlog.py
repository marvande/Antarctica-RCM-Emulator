#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:55:02 2019

@author: ckittel
ANO SMB vs ANO warming
"""

import xarray as xr
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from netCDF4 import num2date


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  #WARNING WARNINGS ARE IGNORED

def polyfit(x, y, degree):
    """This function computes the determination coefficient between any
    curve fitted with.
    """
    results = {}

    coeffs,cp = np.polyfit(x, y, degree,cov=True)
    curve = np.polyval(coeffs, x)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssres = np.sum((y - curve)**2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = 1 - ssres / sstot
    return results,cp, curve

from sklearn.metrics import mean_squared_error
from math import sqrt




def plotvsall():
    fig, ax2 = plt.subplots(nrows=6,ncols=2,figsize=(12,22),sharex=False)
    runm=5
    reg='grd'
    results_CM5,cp=plotvst(ax2[0,0],'SF',runm,reg,'year',legend='True',vst='TAS')
    
    ax2[0,0].text(0.4, 0.16, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[0,0].transAxes)
    
    ax2[0,0].text(0.25, 0.05, r'f(x) = ${:.1f}\cdot x {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1]), fontsize=12,
             transform=ax2[0,0].transAxes)
    
    
    results_CM5,cp=plotvst(ax2[1,0],'RF',runm,reg,'year')
    #results_CM5,cp=plotvst(ax2[1,0],'RU',runm,reg,'year',vst='TT')
    
    ax2[1,0].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[1,0].transAxes)
    
    ax2[1,0].text(0.05, 0.7, r'f(x) = ${:.1f} + \cdot x^{{2}} {:.1f}\cdot x + {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1], results_CM5['polynomial'][2]), fontsize=12,
             transform=ax2[1,0].transAxes)
    
    results_CM5,cp=plotvst(ax2[2,0],'RU',runm,reg,'year')
    
    ax2[2,0].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[2,0].transAxes)
    
    ax2[2,0].text(0.05, 0.7, r'f(x) = ${:.1f} \cdot x^{{2}} {:.1f}\cdot x + {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1], results_CM5['polynomial'][2]), fontsize=12,
             transform=ax2[2,0].transAxes)
  
    
    results_CM5,cp=plotvst(ax2[3,0],'SU',runm,reg,'year')
    
    ax2[3,0].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[3,0].transAxes)
    
    ax2[3,0].text(0.05, 0.7, r'f(x) = ${:.1f} \cdot x^{{2}} + {:.1f}\cdot x {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1], results_CM5['polynomial'][2]), fontsize=12,
             transform=ax2[3,0].transAxes)
    
    results_CM5,cp=plotvst(ax2[4,0],'SMB',runm,reg,'year')
    
    ax2[4,0].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[4,0].transAxes)
    
    ax2[4,0].text(0.05, 0.7, r'f(x) = ${:.1f} \cdot x^{{2}} + {:.1f}\cdot x  {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1], results_CM5['polynomial'][2]), fontsize=12,
             transform=ax2[4,0].transAxes)
    
    results_CM5,cp=plotvst(ax2[5,0],'ME',runm,reg,'year')
    
    ax2[5,0].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[5,0].transAxes)
    
    ax2[5,0].text(0.05, 0.7, r'f(x) = ${:.1f} \cdot x^{{2}} + {:.1f}\cdot x {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1], results_CM5['polynomial'][2]), fontsize=12,
             transform=ax2[5,0].transAxes)
  
    
    
    reg='shelf'
    results_CM5,cp=plotvst(ax2[0,1],'SF',runm,reg,'year')
    
    ax2[0,1].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[0,1].transAxes)
    
    ax2[0,1].text(0.05, 0.7, r'f(x) = ${:.1f}\cdot x + {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1]), fontsize=12,
             transform=ax2[0,1].transAxes)
  
    results_CM5,cp=plotvst(ax2[1,1],'RF',runm,reg,'year')
    #results_CM5,cp=plotvst(ax2[1,0],'RU',runm,reg,'year',vst='TT')
    
    ax2[1,1].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[1,1].transAxes)
    
    ax2[1,1].text(0.05, 0.7, r'f(x) = ${:.1f} \cdot x^{{2}} + {:.1f}\cdot x + {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1], results_CM5['polynomial'][2]), fontsize=12,
             transform=ax2[1,1].transAxes)
    
    results_CM5,cp=plotvst(ax2[2,1],'RU',runm,reg,'year')
    
    ax2[2,1].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[2,1].transAxes)
    
    ax2[2,1].text(0.05, 0.7, r'f(x) = ${:.1f} \cdot x^{{2}} + {:.1f}\cdot x + {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1], results_CM5['polynomial'][2]), fontsize=12, 
             transform=ax2[2,1].transAxes)
  
    
    results_CM5,cp=plotvst(ax2[3,1],'SU',runm,reg,'year')
    
    ax2[3,1].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[3,1].transAxes)
    
    ax2[3,1].text(0.05, 0.7, r'f(x) = ${:.1f} \cdot x^{{2}} + {:.1f}\cdot x {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1], results_CM5['polynomial'][2]), fontsize=12, 
             transform=ax2[3,1].transAxes)
    
    results_CM5,cp=plotvst(ax2[4,1],'SMB',runm,reg,'year')
    
    ax2[4,1].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[4,1].transAxes)
    
    ax2[4,1].text(0.05, 0.7, r'f(x) = ${:.1f} \cdot x^{{2}} + {:.1f}\cdot x {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1], results_CM5['polynomial'][2]), fontsize=12, 
             transform=ax2[4,1].transAxes)  
    
    
    results_CM5,cp=plotvst(ax2[5,1],'ME',runm,reg,'year')
    
    ax2[5,1].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[5,1].transAxes)
    
    ax2[5,1].text(0.05, 0.7, r'f(x) = ${:.1f} \cdot x^{{2}} + {:.1f}\cdot x {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1], results_CM5['polynomial'][2]), fontsize=12, 
             transform=ax2[5,1].transAxes)
  
       
    ymin=200
    ymax=-1200

    ax2[4,1].set_ylim([ymin,ymax])
    
    fig.tight_layout(pad=2.5)

    fig.savefig('./plotALLMEvsTAS.png',
           format='PNG', dpi=500)    
def plotvsind():
    fig, ax2 = plt.subplots(nrows=3,ncols=2,figsize=(14,18),sharex=False)
    
    runm=5
    reg='grd'
    results_CM5,cp=plotvst(ax2[0,0],'PP',runm,reg,'year',legend='True',vst='TAS')
    #results_CM5,cp=plotvst(ax2[0,0],'SF',runm,reg,'year',legend='True',vst='TT')   
    
    
    ax2[0,0].text(0.4, 0.16, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[0,0].transAxes)

    
    ax2[0,0].text(0.25, 0.05, r'f(x) = ${:.1f}\cdot x {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1]), fontsize=12,
             transform=ax2[0,0].transAxes)
    

    ax2[0,0].text(0.05, 0.9, "A",fontsize=18, transform=ax2[0,0].transAxes,
       fontdict={'weight': 'bold'})
    
    ax2[0,0].legend(frameon=False,prop={'size': 12},loc=6)
    
  
    results_CM5,cp=plotvst(ax2[1,0],'RU',runm,reg,'year')
    #results_CM5,cp=plotvst(ax2[1,0],'RU',runm,reg,'year',vst='TT')
    
    ax2[1,0].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[1,0].transAxes)
    
    ax2[1,0].text(0.05, 0.7, r'f(x) = ${:.1f} \cdot x^{{2}} {:.1f}\cdot x + {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1], results_CM5['polynomial'][2]), fontsize=12,
             transform=ax2[1,0].transAxes)

    ax2[1,0].text(0.05, 0.9, "C",fontsize=18, transform=ax2[1,0].transAxes,
       fontdict={'weight': 'bold'})    
    
#    plotvst(ax2[1,0],'SMB',runm,reg,'year')
    
    plotanomar(ax2[2,0],'SMB',runm,reg,season='year',vst='TAS')
    
    ax2[2,0].text(0.05, 0.9, "E",fontsize=18, transform=ax2[2,0].transAxes,
       fontdict={'weight': 'bold'}) 

    reg='shelf'
    results_CM5,cp=plotvst(ax2[0,1],'PP',runm,reg,'year',vst='TAS')

    ax2[0,1].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[0,1].transAxes)
    
    ax2[0,1].text(0.05, 0.7, r'f(x) = ${:.1f}\cdot x {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1]), fontsize=12,
             transform=ax2[0,1].transAxes)    
    
    ax2[0,1].text(0.05, 0.9, "B",fontsize=18, transform=ax2[0,1].transAxes,
       fontdict={'weight': 'bold'})
    
    results_CM5,cp=plotvst(ax2[1,1],'RU',runm,reg,'year')
#    results_CM5,cp=plotvst(ax2[1,1],'RU',runm,reg,'year',vst='TT')
    
    ax2[1,1].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[1,1].transAxes)
    ax2[1,1].text(0.05, 0.7, r'f(x) = ${:.1f} \cdot x^{{2}} {:.1f}\cdot x + {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1], results_CM5['polynomial'][2]), fontsize=12,
             transform=ax2[1,1].transAxes)    
    
    
    ax2[1,1].text(0.05, 0.9, "D",fontsize=18, transform=ax2[1,1].transAxes,
       fontdict={'weight': 'bold'})
    
#    plotvst(ax2[1,1],'SMB',runm,reg,'year')
 
    plotanomar(ax2[2,1],'SMB',runm,reg,season='year',vst='TAS')
    
    ax2[2,1].text(0.05, 0.9, "F",fontsize=18, transform=ax2[2,1].transAxes,
       fontdict={'weight': 'bold'})
#    plotanomar(ax2[2,1],'SMB',runm,reg,season='year',vst='TT')    

#    fig.tight_layout()
    fig.tight_layout(pad=2.5)
    
    ax2[0,0].set_title("Grounded ice", fontsize=18)
    ax2[0,1].set_title("Ice shelves", fontsize=18)
    
    fig.savefig('./plotPPRUSMBvsTTTTeq.png',
           format='PNG', dpi=900)
    
#    fig.savefig('./plotvsTAS_SFRF.png',
#           format='PNG', dpi=500)
    
    
#    fig.savefig('./plotvsTAS_SMB.png',
#           format='PNG', dpi=500)
    
    plt.show()
    
    
    
    
def plotvsind2():
    fig, ax2 = plt.subplots(nrows=3,ncols=2,figsize=(14,18),sharex=False)
    
    runm=5
    reg='grd'
    results_CM5,cp=plotvst(ax2[0,0],'SF',runm,reg,'year',legend='True',vst='TAS',deg=1)
    #results_CM5,cp=plotvst(ax2[0,0],'SF',runm,reg,'year',legend='True',vst='TT')   
    
    
    ax2[0,0].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[0,0].transAxes)

    
    ax2[0,0].text(0.05, 0.70, r'$f(x) = {:.1f}\cdot x {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1]), fontsize=12,
             transform=ax2[0,0].transAxes)
    

    ax2[0,0].text(0.05, 0.9, "A",fontsize=18, transform=ax2[0,0].transAxes,
       fontdict={'weight': 'bold'})
    
    ax2[0,0].legend(frameon=False,prop={'size': 12},loc=6)
    
    
    results_CM5,cp=plotvst(ax2[1,0],'RF',runm,reg,'year')
    #results_CM5,cp=plotvst(ax2[1,0],'RU',runm,reg,'year',vst='TT')
    
    ax2[1,0].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[1,0].transAxes)
    
    ax2[1,0].text(0.05, 0.7, r'f(x) = ${:.1f} \cdot x^{{2}} + {:.1f}\cdot x + {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1], results_CM5['polynomial'][2]), fontsize=12,
             transform=ax2[1,0].transAxes)

    ax2[1,0].text(0.05, 0.9, "C",fontsize=18, transform=ax2[1,0].transAxes,
       fontdict={'weight': 'bold'})    
    
  
    results_CM5,cp=plotvst(ax2[2,0],'RU',runm,reg,'year')
    #results_CM5,cp=plotvst(ax2[1,0],'RU',runm,reg,'year',vst='TT')
    
    ax2[2,0].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[2,0].transAxes)
    
    ax2[2,0].text(0.05, 0.7, r'f(x) = ${:.1f} \cdot x^{{2}} {:.1f}\cdot x + {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1], results_CM5['polynomial'][2]), fontsize=12,
             transform=ax2[2,0].transAxes)

    ax2[2,0].text(0.05, 0.9, "E",fontsize=18, transform=ax2[2,0].transAxes,
       fontdict={'weight': 'bold'})    
    


    reg='shelf'
    results_CM5,cp=plotvst(ax2[0,1],'SF',runm,reg,'year',vst='TAS')

    ax2[0,1].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[0,1].transAxes)
    
    ax2[0,1].text(0.05, 0.7, r'f(x) = ${:.1f} \cdot x^{{2}} + {:.1f}\cdot x {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1], results_CM5['polynomial'][2]), fontsize=12,
             transform=ax2[0,1].transAxes)    
    
    ax2[0,1].text(0.05, 0.9, "B",fontsize=18, transform=ax2[0,1].transAxes,
       fontdict={'weight': 'bold'})
    
    results_CM5,cp=plotvst(ax2[1,1],'RF',runm,reg,'year')
#    results_CM5,cp=plotvst(ax2[1,1],'RU',runm,reg,'year',vst='TT')
    
        
    ax2[1,1].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[1,1].transAxes)
    ax2[1,1].text(0.05, 0.7, r'f(x) = ${:.1f} \cdot x^{{2}} + {:.1f}\cdot x + {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1], results_CM5['polynomial'][2]), fontsize=12,
             transform=ax2[1,1].transAxes)    
    
    
    ax2[1,1].text(0.05, 0.9, "D",fontsize=18, transform=ax2[1,1].transAxes,
       fontdict={'weight': 'bold'})
    
    results_CM5,cp=plotvst(ax2[2,1],'RU',runm,reg,'year')
    
    ax2[2,1].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[2,1].transAxes)
    ax2[2,1].text(0.05, 0.7, r'f(x) = ${:.1f} \cdot x^{{2}} {:.1f}\cdot x + {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1], results_CM5['polynomial'][2]), fontsize=12,
             transform=ax2[2,1].transAxes)    
    
    
    ax2[2,1].text(0.05, 0.9, "F",fontsize=18, transform=ax2[2,1].transAxes,
       fontdict={'weight': 'bold'})
  

#    fig.tight_layout()
    fig.tight_layout(pad=2.5)
    
    ax2[0,0].set_title("Grounded ice", fontsize=18)
    ax2[0,1].set_title("Ice shelves", fontsize=18)
    
#    fig.savefig('./plotPPRUSMBvsTTTTeq.png',
#           format='PNG', dpi=900)
    
#    fig.savefig('./plotvsTAS_SFRF.png',
#           format='PNG', dpi=500)
    
    
    fig.savefig('./plotCOMPvsTAS.png',
           format='PNG', dpi=500)
    
    plt.show()    
    
def plotvsSMB():
    fig, ax2 = plt.subplots(nrows=1,ncols=2,figsize=(14,6),sharex=False)
    
    runm=5
    reg='grd'

    results_CM5,cp=plotvst(ax2[0],'SMB',runm,reg,'year',legend='True',vst='TAS',deg=2)
    #results_CM5,cp=plotvst(ax2[0,0],'SF',runm,reg,'year',legend='True',vst='TT')   
    
    
    ax2[0].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[0].transAxes)

    
    ax2[0].text(0.05, 0.70,r'f(x) = ${:.1f} \cdot x^{{2}} + {:.1f}\cdot x  {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1], results_CM5['polynomial'][2]), fontsize=12,
             transform=ax2[0].transAxes)
    
    

    ax2[0].text(0.05, 0.9, "A",fontsize=18, transform=ax2[0].transAxes,
       fontdict={'weight': 'bold'})
    
    ax2[0].legend(frameon=False,prop={'size': 12},loc=6)
    
    print (cp)
    
    reg='shelf'
    results_CM5,cp=plotvst(ax2[1],'SMB',runm,reg,'year',vst='TAS')

    ymin=-1400
    ymax=200

    ax2[1].set_ylim([ymin,ymax])
    
    ax2[1].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[1].transAxes)
    ax2[1].text(0.05, 0.7, r'f(x) = ${:.1f} \cdot x^{{2}} + {:.1f}\cdot x  {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1], results_CM5['polynomial'][2]), fontsize=12,
             transform=ax2[1].transAxes)  
    
    ax2[1].text(0.05, 0.9, "B",fontsize=18, transform=ax2[1].transAxes,
       fontdict={'weight': 'bold'}) 

 
      
    ax2[0].set_title("Grounded ice", fontsize=18)
    ax2[1].set_title("Ice shelves", fontsize=18)
    
    fig.tight_layout(pad=2.5)
    plt.show()
    fig.savefig('./plotSMBvsTAS.png',
           format='PNG', dpi=500)
    
    print(cp)
    
    
#    plt.cla('all')   # Clear axis
#    plt.clf('all')   # Clear figure    
    plt.close('all')
    fig, ax2 = plt.subplots(nrows=1,ncols=2,figsize=(14,6),sharex=False)
    
    runm=5
    reg='grd'
    
    plotanomar(ax2[0],'SMB',runm,reg,season='year',vst='SMB')
    
    ax2[0].text(0.05, 0.9, "A",fontsize=18, transform=ax2[0].transAxes,
       fontdict={'weight': 'bold'}) 
 
    

    reg='shelf'
    plotanomar(ax2[1],'SMB',runm,reg,season='year',vst='SMB')
    
    ax2[1].text(0.05, 0.9, "B",fontsize=18, transform=ax2[1].transAxes,
       fontdict={'weight': 'bold'})

  

#    fig.tight_layout()
    fig.tight_layout(pad=2.5)
    
    ax2[0].set_title("Grounded ice", fontsize=18)
    ax2[1].set_title("Ice shelves", fontsize=18)
    
    
    fig.savefig('./eval_plotSMBvsTAS.png',
           format='PNG', dpi=500)
    
    plt.show()    

def plotvsmain(var):
    fig, ax2 = plt.subplots(nrows=1,ncols=2,figsize=(13.5,5),sharex=True)

    
    runm=5
    plotvst(ax2[0],var,runm,'grd','year')
    plotvst(ax2[1],var,runm,'shelf','year')
    
    fig.tight_layout()
    
    plt.show()
    
#    fig.savefig('./plotvsmain'+reg+'.png','
#            format='PNG', dpi=500)
    

def plotvst(ax2,var,runm,reg,season,legend='False',vst='TAS',deg=2,mrk="o",fit_reg='True',s=8,lw=1,aplha=0.7,lwr=1.5):
    '''
    var => SMB and co, deg=1 or 2 (1 for SF, 2 for other one), reg=ice, grd, shelf
    result_mar_dic => MAR
    result_tt_dic  => GCM
    
    '''
    print (ax2,var,runm,reg,season)
    
        
    if var == 'PP':
        deg=1    
    
    if season == 'DJF':
        ys=1981
    else:
        ys=1981
     
    pal = sns.color_palette("PuBu",10)
    pal_or = sns.color_palette("OrRd",10)    
    
    #TT from the 4 GCMs
    if vst == 'SF':
        AC3g=gcm_dic_year[reg]['AC3']['PRSN']
        NORg=gcm_dic_year[reg]['NOR']['PRSN']
        CNRg=gcm_dic_year[reg]['CNR']['PRSN']
        CSMg=gcm_dic_year[reg]['CSM']['PRSN']
        AC3_GCM=compute_ano(AC3g,1981,2010,runm)
        NOR_GCM=compute_ano(NORg,1981,2010,runm)
        CNR_GCM=compute_ano(CNRg,1981,2010,runm)
        CSM_GCM=compute_ano(CSMg,1981,2010,runm)
    elif vst== 'TT':
        AC3g=result_mar_dic[season]['ice']['AC3'][vst]
        NORg=result_mar_dic[season]['ice']['NOR'][vst]
        CNRg=result_mar_dic[season]['ice']['CNR'][vst]
        CSMg=result_mar_dic[season]['ice']['CSM'][vst]
        AC3_GCM=compute_ano(AC3g,1981,2010,runm)
        NOR_GCM=compute_ano(NORg,1981,2010,runm)
        CNR_GCM=compute_ano(CNRg,1981,2010,runm)
        CSM_GCM=compute_ano(CSMg,1981,2010,runm)
    else:
        AC3_GCM=result_tt_dic[season]['AC3'][vst].sel(TIME=slice(str(ys)+'-01-01','2100-01-01')).rolling(TIME=runm, center=True).mean()
        NOR_GCM=result_tt_dic[season]['NOR'][vst].sel(TIME=slice(str(ys)+'-01-01','2100-01-01')).rolling(TIME=runm, center=True).mean()
        CNR_GCM=result_tt_dic[season]['CNR'][vst].sel(TIME=slice(str(ys)+'-01-01','2100-01-01')).rolling(TIME=runm, center=True).mean()
        CSM_GCM=result_tt_dic[season]['CSM'][vst].sel(TIME=slice(str(ys)+'-01-01','2100-01-01')).rolling(TIME=runm, center=True).mean()


    #Récup output dico par saison reg, modèle, variable
    if var == 'PP':
        AC3=result_mar_dic[season][reg]['AC3']['SF']+result_mar_dic[season][reg]['AC3']['RF']
        NOR=result_mar_dic[season][reg]['NOR']['SF']+result_mar_dic[season][reg]['NOR']['RF']
        CNR=result_mar_dic[season][reg]['CNR']['SF']+result_mar_dic[season][reg]['CNR']['RF']
        CSM=result_mar_dic[season][reg]['CSM']['SF']+result_mar_dic[season][reg]['CSM']['RF'] 
    else:
        
        AC3=result_mar_dic[season][reg]['AC3'][var]
        NOR=result_mar_dic[season][reg]['NOR'][var]
        CNR=result_mar_dic[season][reg]['CNR'][var]
        CSM=result_mar_dic[season][reg]['CSM'][var]
   
    AC3_MAR= compute_ano(AC3,1981,2010,runm)
    NOR_MAR=compute_ano(NOR,1981,2010,runm)
    CNR_MAR=compute_ano(CNR,1981,2010,runm)
    CSM_MAR=compute_ano(CSM,1981,2010,runm)
    

    mar=np.append(AC3_MAR.values,[NOR_MAR.values,CNR_MAR.values,CSM_MAR.values])
    gcm=np.append(AC3_GCM.values,[NOR_GCM.values,CNR_GCM.values,CSM_GCM.values])


    mar2=AC3_MAR.values
    gcm2=AC3_GCM.values
    msk2=~np.isnan(gcm2)
    results,cp, curve = polyfit(gcm2[msk2],mar2[msk2], deg)

    print(r'f(x) = ${:.1f} \cdot x^{{2}} + {:.1f}\cdot x  {:.1f}$'.format(results['polynomial'][0],
             results['polynomial'][1], results['polynomial'][2]))
    
    mar2=NOR_MAR.values
    gcm2=NOR_GCM.values
    msk2=~np.isnan(gcm2)
    results,cp, curve = polyfit(gcm2[msk2],mar2[msk2], deg)

    print(r'f(x) = ${:.1f} \cdot x^{{2}} + {:.1f}\cdot x  {:.1f}$'.format(results['polynomial'][0],
             results['polynomial'][1], results['polynomial'][2]))
    
    mar2=CNR_MAR.values
    gcm2=CNR_GCM.values
    msk2=~np.isnan(gcm2)
    results,cp, curve = polyfit(gcm2[msk2],mar2[msk2], deg)

    print(r'f(x) = ${:.1f} \cdot x^{{2}} + {:.1f}\cdot x  {:.1f}$'.format(results['polynomial'][0],
             results['polynomial'][1], results['polynomial'][2]))
    
    mar2=CSM_MAR.values
    gcm2=CSM_GCM.values
    msk2=~np.isnan(gcm2)
    results,cp, curve = polyfit(gcm2[msk2],mar2[msk2], deg)

    print(r'f(x) = ${:.1f} \cdot x^{{2}} + {:.1f}\cdot x  {:.1f}$'.format(results['polynomial'][0],
             results['polynomial'][1], results['polynomial'][2]))

    
    if fit_reg=='False': #to remove legend for SF and RF with PP

        sns.regplot(x=AC3_GCM,y=AC3_MAR,ax=ax2,order=deg,
                color='grey',marker=mrk,scatter_kws={"s": s, "alpha": aplha},
                line_kws={"linewidth": lw}, ci=None,fit_reg=fit_reg,truncate=True)
        sns.regplot(x=NOR_GCM,y=NOR_MAR,ax=ax2,order=deg,
                color='grey',marker=mrk, scatter_kws={"s": s, "alpha": aplha},
                line_kws={"linewidth": lw}, ci=None,fit_reg=fit_reg,truncate=True)
        sns.regplot(x=CNR_GCM,y=CNR_MAR,ax=ax2,order=deg, 
                color='grey',marker=mrk, scatter_kws={"s": s, "alpha": aplha},
                line_kws={"linewidth": lw}, ci=None,fit_reg=fit_reg,truncate=True)
        sns.regplot(x=CSM_GCM,y=CSM_MAR,ax=ax2,order=deg, 
                color='grey',marker=mrk, scatter_kws={"s": s, "alpha": aplha},
                line_kws={"linewidth": lw}, ci=None,fit_reg=fit_reg,truncate=True)
    elif fit_reg=='True':   
        sns.regplot(x=AC3_GCM,y=AC3_MAR,ax=ax2,order=deg,label='ACCESS1.3', 
                color=pal[7],marker=mrk,scatter_kws={"s": s, "alpha": aplha,"zorder":10},
                line_kws={"linewidth": lw}, ci=None,fit_reg=fit_reg,truncate=True)
        sns.regplot(x=NOR_GCM,y=NOR_MAR,ax=ax2,order=deg,label='NorESM1-M', 
                color=pal[4],marker=mrk, scatter_kws={"s": s, "alpha": aplha,"zorder": 10},
                line_kws={"linewidth": lw}, ci=None,fit_reg=fit_reg,truncate=True)
        sns.regplot(x=CNR_GCM,y=CNR_MAR,ax=ax2,order=deg, label='CNRM-CM6-1', 
                color=pal_or[7],marker=mrk, scatter_kws={"s": s, "alpha": aplha,"zorder":10},
                line_kws={"linewidth": lw}, ci=None,fit_reg=fit_reg,truncate=True)
        sns.regplot(x=CSM_GCM,y=CSM_MAR,ax=ax2,order=deg, label='CESM2', 
                color=pal_or[4],marker=mrk, scatter_kws={"s": s, "alpha": aplha,"zorder":10},
                line_kws={"linewidth": lw}, ci=None,fit_reg=fit_reg,truncate=True)
    
    
    sns.regplot(x=gcm,y=mar,ax=ax2,order=deg,
                color='k', scatter_kws={"s": 0, "alpha": 0.7,"zorder": 10},
                line_kws={"linewidth": lwr}, ci=None,fit_reg=fit_reg,truncate=True)

    #sns.despine()
    
    #ax2.set_xlim(-1,9)
    
    if fit_reg=='True':
        msk=~np.isnan(gcm)
        results_CM5,cp, curve_CM5 = polyfit(gcm[msk],mar[msk], deg)
    else:
        results_CM5=0
        
        
        
        
    # Do the interpolation for plotting
    t=np.sort(gcm)
    p=results_CM5['polynomial'] #coef
    C_p=cp #covariance
    n=deg #deg de la régression
# Matrix with rows 1, t, t**2, ...: tout son blabla non modifié où je n'ai pas tout compris
    TT = np.vstack([t**(n-i) for i in range(n+1)]).T
    yi = np.dot(TT, p)  # matrix multiplication calculates the polynomial values
    C_yi = np.dot(TT, np.dot(C_p, TT.T)) # C_y = TT*C_z*TT.T
    sig_yi = np.sqrt(np.diag(C_yi))  # Standard deviations are sqrt of diagonal
    
    ax2.fill_between(t, yi + 30*sig_yi,
                        yi - 30*sig_yi,
                        color='b', alpha=0.2)




    if vst == 'TAS':
        ax2.set_xlabel('ESM 90S-60S near-surface temperature anomaly (°C)', fontsize=14)
    
    if vst == 'TA700':
        ax2.set_xlabel('ESM 90S-60S 700hPa temperature anomaly (°C)', fontsize=14)
    
    if vst == 'SF':
        ax2.set_xlabel('ESM Antarctic snowfall anomaly (Gt yr$^{-1}$)', fontsize=14)
        
        
    if var == 'SF':
        ax2.set_ylabel('MAR snowfall anomaly (Gt yr$^{-1}$)', fontsize=14)
    if var == 'RF':
        ax2.set_ylabel('MAR rainfall anomaly (Gt yr$^{-1}$)', fontsize=14)
    if var == 'PP':
        ax2.set_ylabel('MAR precipitation anomaly (Gt yr$^{-1}$)', fontsize=14)
    if var == 'RU':
        ax2.set_ylabel('MAR runoff anomaly (Gt yr$^{-1}$)', fontsize=14)
    if var == 'SU':
        ax2.set_ylabel('MAR sublimation anomaly (Gt yr$^{-1}$)', fontsize=14)        
    if var == 'SMB':
        ax2.set_ylabel('MAR SMB anomaly (Gt yr$^{-1}$)', fontsize=14)
    if var == 'ME':
        ax2.set_ylabel('MAR melt anomaly (Gt yr$^{-1}$)', fontsize=14)
    
#    print(legend)
#    if legend == 'True':
#        ax2.legend(frameon=False,prop={'size': 12})
    
    ax2.axhline(color='black', alpha=1,lw=1)
    
    ax2.tick_params(axis='both', labelsize=12)
    ymin=-200
    ymax=1400

    ax2.set_ylim([ymin,ymax])
    
    
#    ax2.xaxis.set_ticks(range(9))
#    sns.despine()

    print (results_CM5)
    
    return results_CM5,cp
    

def plotanomar(ax2,var,runm,reg,season,legend='False',vst='TAS'):

    deg=1
    ys=1981

     
    pal = sns.color_palette("PuBu",10)
    pal_or = sns.color_palette("OrRd",10)

    #Récup Anomalie MAR en y
    AC3=result_mar_dic[season][reg]['AC3']['SMB']
    NOR=result_mar_dic[season][reg]['NOR']['SMB']
    CNR=result_mar_dic[season][reg]['CNR']['SMB']
    CSM=result_mar_dic[season][reg]['CSM']['SMB']
   
    AC3_MAR= compute_ano(AC3,1981,2010,runm)
    NOR_MAR=compute_ano(NOR,1981,2010,runm)
    CNR_MAR=compute_ano(CNR,1981,2010,runm)
    CSM_MAR=compute_ano(CSM,1981,2010,runm)

    mar=np.append(AC3_MAR.values,[NOR_MAR.values,CNR_MAR.values,CSM_MAR.values])
    
    #Récup anom GCM
    AC3_GCM=result_tt_dic[season]['AC3']['TAS'].sel(TIME=slice(str(ys)+'-01-01','2100-01-01')).rolling(TIME=runm, center=True).mean()
    NOR_GCM=result_tt_dic[season]['NOR']['TAS'].sel(TIME=slice(str(ys)+'-01-01','2100-01-01')).rolling(TIME=runm, center=True).mean()
    CNR_GCM=result_tt_dic[season]['CNR']['TAS'].sel(TIME=slice(str(ys)+'-01-01','2100-01-01')).rolling(TIME=runm, center=True).mean()
    CSM_GCM=result_tt_dic[season]['CSM']['TAS'].sel(TIME=slice(str(ys)+'-01-01','2100-01-01')).rolling(TIME=runm, center=True).mean()
    if vst == 'SF':
        AC3g=gcm_dic_year[reg]['AC3']['PRSN']
        NORg=gcm_dic_year[reg]['NOR']['PRSN']
        CNRg=gcm_dic_year[reg]['CNR']['PRSN']
        CSMg=gcm_dic_year[reg]['CSM']['PRSN']
        AC3_GCMsf=compute_ano(AC3g,1981,2010,runm)
        NOR_GCMsf=compute_ano(NORg,1981,2010,runm)
        CNR_GCMsf=compute_ano(CNRg,1981,2010,runm)
        CSM_GCMsf=compute_ano(CSMg,1981,2010,runm)
    else:        
        AC3_GCMsf=AC3_GCM
        NOR_GCMsf=NOR_GCM
        CNR_GCMsf=CNR_GCM
        CSM_GCMsf=CSM_GCM
    
    
    def ru2(x, a, b,c):
        return (a*x*x)+ (b * x) + c
    def sf1(x, a, b):
        return (a * x) + b
    
    if reg=="grd":
        #ru= [6.495452019801952, -0.0467806093036059, 0.9313132086099879]
        #sf=0.9575614506456429, 6.152112806912676
        if vst=='SF':
            #only SF
            #asf=0.9575614506456429
            #bsf=6.152112806912676
            #SF+PP
            asf=1.0#452468687012129
            bsf=2.5#400869044159284
        else:
            #Only SF
#            asf=132.51738522787983
#            bsf=-22.81895457794965

            #P=SF+RF
            asf=145#144.9775984757646
            bsf=-29.8#-29.766894984760455
            
        aru=6.5#6.495452019801952
        bru=0.0#-0.0467806093036059
        cru=0.9#0.9313132086099879
            
        AC3_SMB= sf1(AC3_GCMsf,asf,bsf)-ru2(AC3_GCM,aru,bru, cru)
        NOR_SMB= sf1(NOR_GCMsf,asf,bsf)-ru2(NOR_GCM,aru,bru, cru)
        CNR_SMB= sf1(CNR_GCMsf,asf,bsf)-ru2(CNR_GCM,aru,bru, cru)
        CSM_SMB= sf1(CSM_GCMsf,asf,bsf)-ru2(CSM_GCM,aru,bru, cru)
        
        if vst=='SMB': #SMB with TAS directly
            asf=-1.3
            bsf=115.4
            csf=-11.1
            AC3_SMB= ru2(AC3_GCMsf,asf,bsf,csf)
            NOR_SMB= ru2(NOR_GCMsf,asf,bsf,csf)
            CNR_SMB= ru2(CNR_GCMsf,asf,bsf,csf)
            CSM_SMB= ru2(CSM_GCMsf,asf,bsf,csf)
            
            
            
#        #SMB with TAS directly
#        aru=-1.3219359584560246
#        bru=115.44973234398654
#        cru=-11.057213085516853
#        AC3_SMB= ru2(AC3_GCM,aru,bru, cru)
#        NOR_SMB=ru2(NOR_GCM,aru,bru, cru)
#        CNR_SMB=ru2(CNR_GCM,aru,bru, cru)
#        CSM_SMB= ru2(CSM_GCM,aru,bru, cru)
        
        xmin,ymin=-200,-200
        xmax,ymax=1400,1400
    
    if reg=="shelf":
        #ru14.055040085683629, -4.741013603926703, 4.342599756823334]
        #sf 0.7683096065237025, 9.981409812406175]
        
        if vst=='SF':
                        #Only SF
            #asf=0.7683096065237025
            #bsf=9.981409812406175
               #P=SF+¨
            asf=1.2 #1.1971841904292764
            bsf=12.3 #12.29406621198801
        
        else:
                        #Only SF
#            asf=19.97956570346236
#            bsf=5.77720837987533
         #P=SF+RF   
            asf=35.3 #35.289646069633164
            bsf=-2.9#-2.9479631284055436
            
        aru=14.1#14.055040085683629
        bru=-4.7#-4.741013603926703
        cru=4.3#4.342599756823334
        
        AC3_SMB= sf1(AC3_GCMsf,asf,bsf)-ru2(AC3_GCM,aru,bru, cru)
        NOR_SMB= sf1(NOR_GCMsf,asf,bsf)-ru2(NOR_GCM,aru,bru, cru)
        CNR_SMB= sf1(CNR_GCMsf,asf,bsf)-ru2(CNR_GCM,aru,bru, cru)
        CSM_SMB= sf1(CSM_GCMsf,asf,bsf)-ru2(CSM_GCM,aru,bru, cru)
        
        if vst=='SMB': #SMB with TAS directly
            aru=-12.7
            bru=32.1
            cru=-3.1
            
            AC3_SMB= ru2(AC3_GCM,aru,bru, cru)
            NOR_SMB= ru2(NOR_GCM,aru,bru, cru)
            CNR_SMB= ru2(CNR_GCM,aru,bru, cru)
            CSM_SMB= ru2(CSM_GCM,aru,bru, cru)

# =============================================================================
# #SMB with TAS directly
#         aru=-12.66309478902407
#         bru=32.14047139924037
#         cru=-3.1282045641708605
#         AC3_SMB= ru2(AC3_GCM,aru,bru, cru)
#         NOR_SMB=ru2(NOR_GCM,aru,bru, cru)
#         CNR_SMB=ru2(CNR_GCM,aru,bru, cru)
#         CSM_SMB= ru2(CSM_GCM,aru,bru, cru)
# =============================================================================
        xmin,ymin=-1400,-1400
        xmax,ymax=200,200
    
    gcm=np.append(AC3_SMB.values,[NOR_SMB.values,CNR_SMB.values,CSM_SMB.values])


    
    sns.regplot(x=gcm,y=mar,ax=ax2,order=deg,
                color='k', scatter_kws={"s": 0, "alpha": 0.7},
                line_kws={"linewidth": 1}, ci=None,fit_reg=True,truncate=True)
    
    sns.regplot(x=AC3_SMB,y=AC3_MAR,ax=ax2,order=deg, label='ACCES1-3',
                color=pal[7], scatter_kws={"s": 8, "alpha": 0.7},
                line_kws={"linewidth": 1}, ci=None,fit_reg=True,truncate=True)
    sns.regplot(x=NOR_SMB,y=NOR_MAR,ax=ax2,order=deg, label='NorESM1-M',
                color=pal[4], scatter_kws={"s": 8, "alpha": 0.7},
                line_kws={"linewidth": 1}, ci=None,fit_reg=True,truncate=True)
    sns.regplot(x=CNR_SMB,y=CNR_MAR,ax=ax2,order=deg, label='CNRM-CM6-1',
                color=pal_or[7], scatter_kws={"s": 8, "alpha": 0.7},
                line_kws={"linewidth": 1}, ci=None,fit_reg=True,truncate=True)
    sns.regplot(x=CSM_SMB,y=CSM_MAR,ax=ax2,order=deg, label='CESM2',
                color=pal_or[4], scatter_kws={"s": 8, "alpha": 0.7},
                line_kws={"linewidth": 1}, ci=None,fit_reg=True,truncate=True)
    
    
    
    msk=~np.isnan(gcm)
    results_CM5,cp, curve_CM5 = polyfit(gcm[msk],mar[msk], deg)
    rms = sqrt(mean_squared_error(mar[msk], gcm[msk]))
    #print (rms)
    print (np.std(mar[msk]-gcm[msk]))
    
    ax2.text(0.05, 0.8, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2.transAxes)
    ax2.text(0.05, 0.73, r'RMSE = {}'.format(round(rms)),fontsize=12, transform=ax2.transAxes)
#    ax2.text(0.34, 0.73, 'Gt yr$^{-1}$',fontsize=12, transform=ax2.transAxes)
    ax2.set_xlabel('ESM reconstructed SMB anomaly (Gt yr$^{-1}$)', fontsize=14)
    
    ax2.set_ylabel('MAR '+str(var)+' anomaly (Gt yr$^{-1}$)', fontsize=14)



    print(legend)
    if legend == 'True':
        ax2.legend(frameon=False,prop={'size': 12})
    
    ax2.axhline(color='black', alpha=1,lw=1)
    
    ax2.set_xlim([ymin,ymax])
    ax2.set_ylim([ymin,ymax])
    
    
#    ax2.xaxis.set_ticks(range(9))
#    sns.despine()

    ax2.tick_params(axis='both', labelsize=12)

    
    x = [xmin,xmax]
    y = [ymin,ymax]
    
    ax2.set_xticks(np.arange(xmin,xmax,200))
    ax2.plot(x,y,color='k',linestyle='--') 
    
    print (results_CM5)
        

 
    
def plotSFvsmerf(reg):
    '''
    var => SMB and co, deg=1 or 2 (1 for SF, 2 for other one), reg=ice, grd, shelf
    result_mar_dic => MAR
    result_tt_dic  => GCM
    
    '''
    
    fig, ax2 = plt.subplots(nrows=1,ncols=1,figsize=(5,5))

    
    runm=5
    season='year'
    
    
    ys=1981
    deg=2
        

     
    pal = sns.color_palette("PuBu",10)
    pal_or = sns.color_palette("OrRd",10)    
    
    #TT from the 4 GCMs

    AC3_GCM=result_mar_dic[season][reg]['AC3']['ME'].sel(SECTOR1_1="1")+result_mar_dic[season][reg]['AC3']['RF']
    NOR_GCM=result_mar_dic[season][reg]['NOR']['ME'].sel(SECTOR1_1="1")+result_mar_dic[season][reg]['NOR']['RF']
    CNR_GCM=result_mar_dic[season][reg]['CNR']['ME'].sel(SECTOR1_1="1")+result_mar_dic[season][reg]['CNR']['RF']
    CSM_GCM=result_mar_dic[season][reg]['CSM']['ME'].sel(SECTOR1_1="1")+result_mar_dic[season][reg]['CSM']['RF']
    
    
    AC3_MAR=compute_ano(AC3_GCM,ys,2010,runm)
    NOR_MAR=compute_ano(NOR_GCM,ys,2010,runm)
    CNR_MAR=compute_ano(CNR_GCM,ys,2010,runm)
    CSM_MAR=compute_ano(CSM_GCM,ys,2010,runm)
    

    var='SF'
    #Récup output dico par saison reg, modèle, variable
    AC3=result_mar_dic[season][reg]['AC3'][var]
    NOR=result_mar_dic[season][reg]['NOR'][var]
    CNR=result_mar_dic[season][reg]['CNR'][var]
    CSM=result_mar_dic[season][reg]['CSM'][var]
   
    AC3_GCM= compute_ano(AC3,ys,2010,runm)
    NOR_GCM=compute_ano(NOR,ys,2010,runm)
    CNR_GCM=compute_ano(CNR,ys,2010,runm)
    CSM_GCM=compute_ano(CSM,ys,2010,runm)
    

    mar=np.append(AC3_MAR.values,[NOR_MAR.values,CNR_MAR.values,CSM_MAR.values])
    gcm=np.append(AC3_GCM.values,[NOR_GCM.values,CNR_GCM.values,CSM_GCM.values])


    
    sns.regplot(x=gcm,y=mar,ax=ax2,order=deg,
                color='k', scatter_kws={"s": 0, "alpha": 0.7},
                line_kws={"linewidth": 1}, ci=None,fit_reg=True,truncate=True)
    
    sns.regplot(x=AC3_GCM,y=AC3_MAR,ax=ax2,order=deg, label='ACCES1-3',
                color=pal[7], scatter_kws={"s": 8, "alpha": 0.7},
                line_kws={"linewidth": 1}, ci=None,fit_reg=True,truncate=True)
    sns.regplot(x=NOR_GCM,y=NOR_MAR,ax=ax2,order=deg, label='NorESM1-M',
                color=pal[4], scatter_kws={"s": 8, "alpha": 0.7},
                line_kws={"linewidth": 1}, ci=None,fit_reg=True,truncate=True)
    sns.regplot(x=CNR_GCM,y=CNR_MAR,ax=ax2,order=deg, label='CNRM-CM6-1',
                color=pal_or[7], scatter_kws={"s": 8, "alpha": 0.7},
                line_kws={"linewidth": 1}, ci=None,fit_reg=True,truncate=True)
    sns.regplot(x=CSM_GCM,y=CSM_MAR,ax=ax2,order=deg, label='CESM2',
                color=pal_or[4], scatter_kws={"s": 8, "alpha": 0.7},
                line_kws={"linewidth": 1}, ci=None,fit_reg=True,truncate=True)
    
    
    ax2.set_xlabel('Warming ($^{\circ}$C)', fontsize=11)
    ax2.set_ylabel(str(var)+' Anomaly (Gt yr$^{-1}$)', fontsize=11)
    ax2.legend()
    sns.despine()
    
    
    msk=~np.isnan(gcm)
    results_CM5,cp, curve_CM5 = polyfit(gcm[msk],mar[msk], deg)
    
    ax2.text(0.05, 0.63, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2.transAxes)
    

    ax2.set_xlabel('SF anomaly (Gt/yr)', fontsize=11)
    ax2.set_ylabel('ME+RF anomaly (Gt/yr)', fontsize=11)



    ax2.legend()
    
    ax2.axhline(color='black', alpha=1,lw=1)
    

#    sns.despine()

    print (results_CM5)
    if reg == "grd":
        ax2.set_title("Grounded ice", fontsize=14)
        amin=-200
        amax=1400
        ax2.set_xlim([amin,amax])
        ax2.set_ylim([amin,amax])
    if reg == "shelf":
        ax2.set_title("Ice shelves", fontsize=14)
        amin=-50
        amax=1600
        
    

    
    fig.tight_layout()
    
    plt.show()
    
    fig.savefig('fig/MERFvsSF'+reg+'.png',
           format='PNG', dpi=300)  
    


def plotvsind3():
    '''
    Third way to do it...
    '''
    fig, ax2 = plt.subplots(nrows=1,ncols=2,figsize=(14,6),sharex=False)
    
    runm=5
    reg='grd'
    
    results_CM5,cp=plotvst(ax2[0],'PP',runm,reg,'year',vst='TAS')
    ax2[0].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
           fontsize=12, transform=ax2[0].transAxes)

    
    ax2[0].text(0.05, 0.70, r'f(x) = ${:.1f}\cdot x {:.1f}$'.format(results_CM5['polynomial'][0],
            results_CM5['polynomial'][1]), fontsize=12,
             transform=ax2[0].transAxes)
    
    ax2[0].legend(frameon=False,prop={'size': 12},loc=6)
 
    
    ax2[0].text(0.05, 0.9, "A",fontsize=18, transform=ax2[0].transAxes,
       fontdict={'weight': 'bold'})

    reg='shelf'

    
    results_CM5,cp=plotvst(ax2[1],'PP',runm,reg,'year',vst='TAS')
    ax2[1].text(0.05, 0.77, r'All $R^{{2}} = {}$'.format(round(results_CM5['determination'], 2)),
            fontsize=12, transform=ax2[1].transAxes)
    
    ax2[1].text(0.05, 0.7, r'f(x) =  ${:.1f}\cdot x {:.1f}$'.format(results_CM5['polynomial'][0],
             results_CM5['polynomial'][1]), fontsize=12,
             transform=ax2[1].transAxes)    
    
    ax2[1].text(0.05, 0.9, "B",fontsize=18, transform=ax2[1].transAxes,
       fontdict={'weight': 'bold'})

#    fig.tight_layout()
    fig.tight_layout(pad=2.5)
    
    ax2[0].set_title("Grounded ice", fontsize=18)
    ax2[1].set_title("Ice shelves", fontsize=18)
    
#    fig.savefig('./plotPPRUSMBvsTTTTeq.png',
#           format='PNG', dpi=900)
    
#    fig.savefig('./plotvsTAS_SFRF.png',
#           format='PNG', dpi=500)
    
    
    fig.savefig('./plotPPvsTAS.png',
           format='PNG', dpi=500)
    
    plt.show()    
        
#plotvsind3()

#plotSFvsRU('grd')
#plotvsmain('ice')
#plotvsmain('grd')
#plotvsmain('shelf')
    
    
#plotvsind()
