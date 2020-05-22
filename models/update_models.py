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

' Update team model '

def update_team_model(gw, N, save_to_csv):
	
	# read fixture list
    fixture_list_this_week = getResults(gw, team_id_file="../data/team_id_20192020.csv", filelead="../")

    # read params and teams
    all_teams_params = pd.read_csv("../parameters/all_teams_params.csv", header=None)
    teams = pd.read_csv("../parameters/all_teams.csv", header=None).values[:, 0]

    # particles
    tp = all_teams_params.values
    params = np.zeros((np.shape(tp)[0], N))
    for j in range(np.shape(tp)[0]):
        params[j, :] = np.random.normal(tp[j, 0], tp[j, 1], N)
    
    # adjust for zero constraint
    params[2:(len(teams) + 2), :] = params[2:(len(teams) + 2), :] - np.mean(params[2:(len(teams) + 2), :], axis=0)
    params[(len(teams) + 2):(2 + (2 * len(teams))), :] = params[(len(teams) + 2):(2 + (2 * len(teams))), :] - np.mean(params[(len(teams) + 2):(2 + (2 * len(teams))), :], axis=0)
	
    # particle filter likelihood
    xi = np.zeros(N)
    for i in range(N):
        for j in range(np.shape(fixture_list_this_week)[0]):
            a_ht = params[2 + np.where(teams == fixture_list_this_week[j, 0])[0], i]
            a_at = params[2 + np.where(teams == fixture_list_this_week[j, 1])[0], i]
            d_ht = params[2 + len(teams) + np.where(teams == fixture_list_this_week[j, 0])[0], i]
            d_at = params[2 + len(teams) + np.where(teams == fixture_list_this_week[j, 1])[0], i]
            xi[i] += np.log(likelihood_one_game(int(fixture_list_this_week[j, 2]),
                                                int(fixture_list_this_week[j, 3]),
                                                params[0, i], params[1, i], a_ht, d_ht, a_at, d_at))
    
    resampled = np.random.choice(np.linspace(0, N - 1, N), N, p=np.exp(xi) / np.sum(np.exp(xi)))
    resampled_params = params[:, resampled.astype(int)]
	
    new_means = np.mean(resampled_params, axis=1)
    new_sds = np.std(resampled_params, axis=1)
	
    if save_to_csv:
        print('saving new team parameters to csv...')
        with open('../parameters/all_teams_params.csv', mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            for i in range(((2 * len(teams)) + 2)):
                csv_writer.writerow([new_means[i], new_sds[i]])
        csv_file.close()


' Update player model'

def update_player_model(gw, ffgoals, ffmins, ffgames, save_to_csv=True, raw_player_data="../data/draft_data/draft_player_raw.csv"):
    
    current_season = 4
    all_players_parameters = pd.read_csv("../parameters/all_players_params.csv")
	
    filelead = "../data/draft_data/gws/gw"
	
    # read raw gw data
    dataraw = pd.read_csv(filelead + str(gw) + ".csv")

    # find id of players in gw data
    ids = dataraw['_id']

    # match these ids to player raw data (with correct format names, corresponding to players parameters data frame)
    # this raw data needs to be same as this season
    name_data = pd.read_csv(raw_player_data)
    name_data['team_name'] = team_code(name_data['team_id'], season="2019/2020")
	
    p = []
    for i in range(len(name_data.index)):
        p.append(np.array(['GKP', 'DEF', 'MID', 'FWD'])[int(name_data.loc[name_data.index[i], "element_type_id"] - 1)])
    name_data['position'] = p

    corr_names = []
    corr_positions = []
    corr_teams = []
    for i in range(len(ids)):
        corr_names.append(name_data.loc[name_data.index[np.where(name_data.loc[:, '_id'] == ids[i])[0][0]], 'first_name'] + ' ' + name_data.loc[name_data.index[np.where(name_data.loc[:, '_id'] == ids[i])[0][0]], 'second_name'])
        corr_positions.append(name_data.loc[name_data.index[np.where(name_data.loc[:, '_id'] == ids[i])[0][0]], 'position'])
        corr_teams.append(name_data.loc[name_data.index[np.where(name_data.loc[:, '_id'] == ids[i])[0][0]], 'team_name'])

    # finally revise dataraw
    data = pd.read_csv(filelead + str(gw) + ".csv")
    data['name'] = corr_names
    data['position'] = corr_positions
    data['team'] = corr_teams
	
    for i in range(len(data.index)):
    
        # in data, search for matches in all_players params, if not found, append row to all players paramas and set with priors
        ind = np.where(all_players_parameters['player'] == data.loc[data.index[i], 'name'])[0]
    
        # then, after all players are included, update with gameweeks worth of data (if player didn't exist before, this will
        # be first addition to inference after prior setting)
        if len(ind) == 0:

            # set new ID
            idnew = data.loc[data.index[i], "_id"]
            teamnew = data.loc[data.index[i], 'team']
            positionnew = data.loc[data.index[i], 'position']
            namenew = data.loc[data.index[i], 'name']
            if positionnew == 'GKP':
                all_players_parameters = all_players_parameters.append({'ID': idnew, 'a_goals': ga_prior_a_g, 'a_mins': m_prior_a_g, 'b_goals': ga_prior_b_g,
                                               'b_mins': m_prior_b_g, 'a_games': p_prior_a_g, 'b_games': p_prior_b_g, 'c_goals': ga_prior_c_g, 'last_season': current_season,
                                               'player': namenew, 'position': positionnew, 'team': teamnew,
                                               'a_tir': t_prior_a_g, 'b_tir': t_prior_b_g}, ignore_index=True)
            if positionnew == 'DEF':
                all_players_parameters = all_players_parameters.append({'ID': idnew, 'a_goals': ga_prior_a_d, 'a_mins': m_prior_a_d, 'b_goals': ga_prior_b_d,
                                               'b_mins': m_prior_b_d, 'a_games': p_prior_a_d, 'b_games': p_prior_b_d, 'c_goals': ga_prior_c_d, 'last_season': current_season,
                                               'player': namenew, 'position': positionnew, 'team': teamnew,
                                               'a_tir': t_prior_a_d, 'b_tir': t_prior_b_d}, ignore_index=True)
            if positionnew == 'MID':
                all_players_parameters = all_players_parameters.append({'ID': idnew, 'a_goals': ga_prior_a_m, 'a_mins': m_prior_a_m, 'b_goals': ga_prior_b_m,
                                               'b_mins': m_prior_b_m, 'a_games': p_prior_a_m, 'b_games': p_prior_b_m, 'c_goals': ga_prior_c_m, 'last_season': current_season,
                                               'player': namenew, 'position': positionnew, 'team': teamnew,
                                               'a_tir': t_prior_a_m, 'b_tir': t_prior_b_m}, ignore_index=True)
            if positionnew == 'FWD':
                all_players_parameters = all_players_parameters.append({'ID': idnew, 'a_goals': ga_prior_a_f, 'a_mins': m_prior_a_f, 'b_goals': ga_prior_b_f,
                                               'b_mins': m_prior_b_f, 'a_games': p_prior_a_f, 'b_games': p_prior_b_f, 'c_goals': ga_prior_c_f, 'last_season': current_season,
                                               'player': namenew, 'position': positionnew, 'team': teamnew,
                                               'a_tir': t_prior_a_f, 'b_tir': t_prior_b_f}, ignore_index=True)
    
        # update all_players params
        new_ind = np.where(all_players_parameters['player'] == data.loc[data.index[i], 'name'])[0][0]  # index now all_players_params is appended
        goa = data.loc[data.index[i], 'gs']
        assi = data.loc[data.index[i], 'a']
        mns = data.loc[data.index[i], 'mp']
        if goa >= np.ceil(data.loc[data.index[i], 'goals_for'] * (data.loc[data.index[i], 'mp'] / 90.)):
            tgoa = data.loc[data.index[i], 'goals_for']
        else:
            tgoa = np.ceil(data.loc[data.index[i], 'goals_for'] * (data.loc[data.index[i], 'mp'] / 90.))
        gms = data.loc[data.index[i], 'mp'] > 0
        if gms == 1: 
            tir = (data.loc[data.index[i], 'i'] + data.loc[data.index[i], 't']) / data.loc[data.index[i], 'mp']
        else:
            tir = 0
    
        post_a_goals, post_b_goals, post_c_goals = update_goals_and_assists_simplex(all_players_parameters.loc[all_players_parameters.index[new_ind], 'a_goals'],
                                                                                    all_players_parameters.loc[all_players_parameters.index[new_ind], 'b_goals'],
                                                                                    all_players_parameters.loc[all_players_parameters.index[new_ind], 'c_goals'], goa, assi, tgoa, ffgoals)
        post_a_tir, post_b_tir = update_tir_simplex(all_players_parameters.loc[all_players_parameters.index[new_ind], 'a_tir'],
                                                    all_players_parameters.loc[all_players_parameters.index[new_ind], 'b_tir'],
                                                    tir, gms, ffgoals)
        post_a_mins, post_b_mins = update_mins_simplex(all_players_parameters.loc[all_players_parameters.index[new_ind], 'a_mins'],
                                                       all_players_parameters.loc[all_players_parameters.index[new_ind], 'b_mins'], mns, gms, ffmins)
        post_a_played, post_b_played = update_games_played_simplex(all_players_parameters.loc[all_players_parameters.index[new_ind], 'a_games'],
                                                       all_players_parameters.loc[all_players_parameters.index[new_ind], 'b_games'], gms, 1 - gms, ffgames)

        all_players_parameters.loc[all_players_parameters.index[new_ind], 'a_goals'] = post_a_goals
        all_players_parameters.loc[all_players_parameters.index[new_ind], 'b_goals'] = post_b_goals
        all_players_parameters.loc[all_players_parameters.index[new_ind], 'c_goals'] = post_c_goals
        all_players_parameters.loc[all_players_parameters.index[new_ind], 'a_mins'] = post_a_mins
        all_players_parameters.loc[all_players_parameters.index[new_ind], 'b_mins'] = post_b_mins
        all_players_parameters.loc[all_players_parameters.index[new_ind], 'a_games'] = post_a_played
        all_players_parameters.loc[all_players_parameters.index[new_ind], 'b_games'] = post_b_played
        all_players_parameters.loc[all_players_parameters.index[new_ind], 'a_tir'] = post_a_tir
        all_players_parameters.loc[all_players_parameters.index[new_ind], 'b_tir'] = post_b_tir
		
    if save_to_csv:
        print('saving new player parameters to csv....')
        all_players_parameters.to_csv("../parameters/all_players_params.csv", index=False)

' Run '
if __name__ == "__main__":

    gw = int(sys.argv[1])
    N = int(sys.argv[2])
    save_to_csv = sys.argv[3]
    ffgoals = float(sys.argv[4])
    ffmins = float(sys.argv[5])
    ffgames = float(sys.argv[6])
    
    update_team_model(gw, N, save_to_csv)
    update_player_model(gw, ffgoals, ffmins, ffgames)