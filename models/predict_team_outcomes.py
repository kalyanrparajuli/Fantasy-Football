import os
import sys
import csv
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import poisson
from scipy.stats import beta
from scipy.stats import gamma
from scipy.stats import dirichlet
import itertools
from utils import *
from priors import *
sys.path.append('../')
from get_api import *

def predictOutcomes(gw, option="win", niter=1000):
    
    # outcome is either 'win' or 'topfour'
	
    # fixture list this season to predict
    fixture_list_this_season = getFixtures(np.linspace(gw, 38, 38 - gw + 1).astype(int), team_id_file="../data/team_id_20192020.csv",
                                           filelead="../")
										   
    # download team parameters
    all_team_params = pd.read_csv("../parameters/all_teams_params.csv", header=None).values
    teams = pd.read_csv("../parameters/all_teams.csv", header=None).values[:, 0]
	
    # find mean and stds
    means = all_team_params[:, 0]
    sds = all_team_params[:, 1]
	
    # preallocate
    wins = np.zeros(20)
    topfour = np.zeros(20)
    tms = np.unique(getFixtures(np.linspace(1, 38, 38).astype(int), team_id_file="../data/team_id_20192020.csv",
                                filelead="../")[:, 0])
	
    # get current points
    current_pnts = np.zeros(20)
    if gw > 1:
        for i in range(1, gw):
            results = pd.read_csv("../data/prem_results_20192020_gw" + str(i) + ".csv", header=None).values
            for j in range(np.shape(results)[0]):
                goals_ht = results[j, 2]
                goals_at = results[j, 3]
                current_pnts[np.where(tms == results[j, 0])[0][0].astype(int)] += ((goals_at < goals_ht) * 3) + (goals_at == goals_ht)
                current_pnts[np.where(tms == results[j, 1])[0][0].astype(int)] += ((goals_at > goals_ht) * 3) + (goals_at == goals_ht)

    # predict lambdas - average form
    interceptest = np.hstack((means[0], sds[0]))
    muest = np.hstack((means[1], sds[1]))
    aest = np.hstack((np.reshape(means[2:(2+len(teams))], ((len(teams), 1))),
                      np.reshape(sds[2:(2+len(teams))], ((len(teams), 1)))))
    dest = np.hstack((np.reshape(means[(2+len(teams)):(2+(2*len(teams)))], ((len(teams), 1))),
                      np.reshape(sds[(2+len(teams)):(2+(2*len(teams)))], ((len(teams), 1)))))

    for j in range(niter):
	
        # work out total goals predicted
        total_goals = np.zeros(20)
        total_goals_conceded = np.zeros(20)
		
        goals_this_season = predict_fixtures(fixture_list_this_season, teams, interceptest, muest, aest, dest, uncertainty=True)
        pnts = np.zeros(20)
        pnts += current_pnts
    
        for i in range(np.shape(fixture_list_this_season)[0]):
            goals_ht = np.random.poisson(goals_this_season[0][i])
            goals_at = np.random.poisson(goals_this_season[1][i])
            total_goals[np.where(tms == fixture_list_this_season[i, 0])[0][0].astype(int)] += goals_ht
            total_goals[np.where(tms == fixture_list_this_season[i, 1])[0][0].astype(int)] += goals_at
            pnts[np.where(tms == fixture_list_this_season[i, 0])[0][0].astype(int)] += ((goals_at < goals_ht) * 3) + (goals_at == goals_ht)
            pnts[np.where(tms == fixture_list_this_season[i, 1])[0][0].astype(int)] += ((goals_at > goals_ht) * 3) + (goals_at == goals_ht)
            total_goals_conceded[np.where(tms == fixture_list_this_season[i, 0])[0][0].astype(int)] += goals_at
            total_goals_conceded[np.where(tms == fixture_list_this_season[i, 1])[0][0].astype(int)] += goals_ht
		
        # update title wins
        wins[np.argmax(pnts)] += 1
        
        # update top four finishes
        topfour[np.argsort(pnts)[16:]] += 1
	
    if option == "win":
        dat = pd.DataFrame({'Team': tms, 'Chance (%)': (wins / niter) * 100})
        dat.to_csv("../data/win_chance_from_gw" + str(gw) + ".csv", index=False)
    if option == "topfour":
        dat = pd.DataFrame({'Team': tms, 'Chance (%)': (topfour / niter) * 100})
        dat.to_csv("../data/topfour_chance_from_gw" + str(gw) + ".csv", index=False)
	
    print(dat)


' Run '

if __name__ == "__main__":

    gw = int(sys.argv[1])
    option = sys.argv[2]
    niter = int(sys.argv[3])
    
    predictOutcomes(gw, option, niter)