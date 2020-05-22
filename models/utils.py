import numpy as np
import os
import csv
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import beta
from scipy.stats import dirichlet
from scipy.stats import skellam
import pandas as pd
import itertools
import matplotlib.pyplot as plot
import sys
sys.path.append('../')
from get_team_api import *

def EstimateParameters(fixture_lists,
                       teams, beta, thetapriormeans, thetapriorsds,
                       niter=1000, log=False, temp=0, zerooutinds=np.array([])):
    
    # xdata and ydata are coordinates and y values of data
    # xmodel are coordinates of model evaluations
    # thetaprior are prior guesses for parameters
    
    # draw initial
    if log:
        if hasattr(thetapriormeans, '__len__'):
            theta = np.zeros(len(thetapriormeans))
            for i in range(len(thetapriormeans)):
                theta[i] = np.exp(np.random.normal(thetapriormeans[i], thetapriorsds[i], 1))
        else:
            theta = np.exp(np.random.normal(thetapriormeans, thetapriorsds, 1))
    else:
        if hasattr(thetapriormeans, '__len__'):
            theta = np.zeros(len(thetapriormeans))
            for i in range(len(thetapriormeans)):
                theta[i] = np.random.normal(thetapriormeans[i], thetapriorsds[i], 1)
            # normalize
            #theta[(len(teams) + 1 - 1)] = -np.sum(theta[1:(len(teams) + 1 - 1)])
            #theta[((2 * len(teams)) + 1 - 1)] = -np.sum(theta[(len(teams) + 1):((2 * len(teams)) + 1 - 1)])
        else:
            theta = np.random.normal(thetapriormeans, thetapriorsds, 1)
    
    if hasattr(thetapriormeans, '__len__'):
        thetaarray = np.zeros((niter, len(thetapriormeans)))
    else:
        thetaarray = np.zeros(niter)
        
    accept_count = 0
    
    for j in range(niter):
        
        # temperature
        T = np.exp(-temp * ((i + 1) / niter))
        
        if log:
            if hasattr(thetapriormeans, '__len__'):
                thetastar = np.exp(np.log(theta) + np.random.normal(0, np.sqrt(beta), len(theta)))
            else:
                thetastar = np.exp(np.log(theta) + np.random.normal(0, np.sqrt(beta), 1))
        else:
            if hasattr(thetapriormeans, '__len__'):
                ind = np.random.normal(0, np.sqrt(beta), len(theta))
                # normalize
                #ind[(len(teams) + 1 - 1)] = -np.sum(ind[1:(len(teams) + 1 - 1)])
                #ind[((2 * len(teams)) + 1 - 1)] = -np.sum(ind[(len(teams) + 1):((2 * len(teams)) + 1 - 1)])
                thetastar = theta + ind
            else:
                ind = np.random.normal(0, np.sqrt(beta), 1)
                thetastar = theta + ind
        
        # get likelihood for each
        intercept = theta[0]
        mu = theta[1]
        a = theta[2:(len(teams) + 2)]
        d = theta[(len(teams) + 2):((2 * len(teams)) + 2)]
        if len(zerooutinds) > 0:  # promoted team zero out
            a[zerooutinds] = 0
            d[zerooutinds] = 0
        a[0] = - np.sum(a[1:])  # normalize
        d[0] = - np.sum(d[1:])  # normalize
        prioreval = np.zeros(len(theta))
        for k in range(len(theta)):
            prioreval[k] = norm.pdf(theta[k], thetapriormeans[k], thetapriorsds[k])
        Htheta = likelihood_all_seasons(fixture_lists,
                                          teams, intercept, mu, a, d) + np.sum(np.log(prioreval))
        
        intercept = thetastar[0]
        mu = thetastar[1]
        a = thetastar[2:(len(teams) + 2)]
        d = thetastar[(len(teams) + 2):((2 * len(teams)) + 2)]
        if len(zerooutinds) > 0:  # promoted team zero out
            a[zerooutinds] = 0
            d[zerooutinds] = 0
        a[0] = - np.sum(a[1:])  # normalize
        d[0] = - np.sum(d[1:])  # normalize
        prioreval = np.zeros(len(thetastar))
        for k in range(len(thetastar)):
            prioreval[k] = norm.pdf(thetastar[k], thetapriormeans[k], thetapriorsds[k])
        Hthetastar = likelihood_all_seasons(fixture_lists,
                                              teams, intercept, mu, a, d) + np.sum(np.log(prioreval))
        
        alpha = np.min([0, (1 / T) * (Hthetastar - Htheta)])
        
        # sample uniformly
        u = np.random.uniform(0, 1)
        
        # accept or not
        accept = np.log(u) <= alpha
        
        if accept:
            theta = thetastar
            accept_count += 1
            
        if hasattr(thetapriormeans, '__len__'):
            thetaarray[j, :] = theta
            if (j%10) == 0:
                print('------')
                print('Iteration: ', str(j))
                print('Home coefficient: '+str(thetaarray[j, 1]))
                print('Arsenal attack coefficient: '+str(thetaarray[j, 2]))
                print('acceptance ratio: ', accept_count / (j + 1))
        else:
            thetaarray[j] = theta
    
    # convert back and normalize
    if hasattr(thetapriormeans, '__len__'):
        if len(zerooutinds) > 0:  # zero out promoted teams
            thetaarray[:, (2 + zerooutinds).astype(int)] = 0.
            thetaarray[:, (2 + zerooutinds + len(teams)).astype(int)] = 0.
        thetaarray[:, 2] = - np.sum(thetaarray[:, 3:(len(teams) + 2)], axis=1)
        thetaarray[:, (len(teams) + 2)] = - np.sum(thetaarray[:, (len(teams) + 3):((2 * len(teams)) + 2)], axis=1)
    
    return thetaarray

# create likelihood eval for one game
def likelihood_one_game(goals_ht, goals_at, intercept, mu, a_ht, d_ht, a_at, d_at):
    lambda_ht = np.exp(intercept + mu + a_ht + d_at)
    lambda_at = np.exp(intercept + a_at + d_ht)
    p = skellam.pmf(goals_ht - goals_at, lambda_ht, lambda_at)
    return(p)

# create likelihood eval for single season
def likelihood_season(fixtures_list, teams, intercept, mu, a, d):
    N = np.shape(fixtures_list)[0]
    goals_ht = fixtures_list[:, 2]
    goals_at = fixtures_list[:, 3]
    teams_ht = fixtures_list[:, 0]
    teams_at = fixtures_list[:, 1]
    
    teams_for_season = np.unique(teams_ht)
    team_count = np.zeros(20)
    likelihood = np.zeros(N)
    for i in range(N):
        ind_ht = np.where(teams == teams_ht[i])[0][0].astype(int)
        ind_at = np.where(teams == teams_at[i])[0][0].astype(int)
        ind_for_season_ht = np.where(teams_for_season == teams_ht[i])[0][0].astype(int)
        ind_for_season_at = np.where(teams_for_season == teams_at[i])[0][0].astype(int)
        l = likelihood_one_game(goals_ht[i], goals_at[i],
                                intercept, mu, a[ind_ht], d[ind_ht], a[ind_at], d[ind_at])
        team_count[np.where(teams_for_season == teams_ht[i])[0][0].astype(int)] += 1
        team_count[np.where(teams_for_season == teams_at[i])[0][0].astype(int)] += 1
        likelihood[i] = l
    
    return(np.sum(np.log(likelihood)))

# likelihood over three seasons - weighted
def likelihood_all_seasons(fixture_lists, teams, intercept, mu, a, d):
    likelihood = 0
    for fixture_list in fixture_lists:
        likelihood = likelihood + ((1.0 / len(fixture_lists)) * likelihood_season(fixture_list, teams, intercept, mu, a, d))
    return(likelihood)

# function to predict probabilities of fixtures
def predict_fixtures(new_fixtures, teams, intercept, mu, a, d, uncertainty=False):
    if uncertainty:
        N = np.shape(new_fixtures)[0]
        teams_ht = new_fixtures[:, 0]
        teams_at = new_fixtures[:, 1]
        lambda_1 = np.zeros(N)
        lambda_2 = np.zeros(N)
        for i in range(N):
            interceptest = np.random.normal(intercept[0], intercept[1])
            muest = np.random.normal(mu[0], mu[1])
            aest = np.zeros(len(teams))
            dest = np.zeros(len(teams))
            for u in range(len(teams)):
                aest[u] = np.random.normal(a[u, 0], a[u, 1])
                dest[u] = np.random.normal(d[u, 0], d[u, 1])
            ind_ht = np.where(teams == teams_ht[i])[0][0].astype(int)
            ind_at = np.where(teams == teams_at[i])[0][0].astype(int)
            lambda_1[i] = np.exp(interceptest + muest + aest[ind_ht] + dest[ind_at])
            lambda_2[i] = np.exp(interceptest + aest[ind_at] + dest[ind_ht])
    else:
        N = np.shape(new_fixtures)[0]
        teams_ht = new_fixtures[:, 0]
        teams_at = new_fixtures[:, 1]
        lambda_1 = np.zeros(N)
        lambda_2 = np.zeros(N)
        for i in range(N):
            ind_ht = np.where(teams == teams_ht[i])[0][0].astype(int)
            ind_at = np.where(teams == teams_at[i])[0][0].astype(int)
            lambda_1[i] = np.exp(intercept + mu + a[ind_ht] + d[ind_at])
            lambda_2[i] = np.exp(intercept + a[ind_at] + d[ind_ht])
    return(lambda_1, lambda_2)

def import_fixture_lists(filename_1, filename_2, filename_3):
    fixture_list_1 = pd.read_csv(filename_1, header=None)
    fixture_list_2 = pd.read_csv(filename_2, header=None)
    fixture_list_3 = pd.read_csv(filename_3, header=None)
    return(fixture_list_1, fixture_list_2, fixture_list_3)

def import_fixture_list(filename):
    fixture_list = pd.read_csv(filename, header=None)
    return(fixture_list)

def read_historical_data(file_y1, file_y2, file_y3):
    y1 = pd.read_csv(file_y1)
    y2 = pd.read_csv(file_y2)
    y3 = pd.read_csv(file_y3)
    return(y1, y2, y3)

def read_current_data(file):
    ycurrent = pd.read_csv(file)
    return(ycurrent)

def convert_team_marker(teams):
    new_teams = []
    for tm in teams:
        if tm == 'ARS':
            new_teams.append('Arsenal')
        if tm == 'CHE':
            new_teams.append('Chelsea')
        if tm == 'BOU':
            new_teams.append('Bournemouth')
        if tm == 'WHU':
            new_teams.append('West Ham')
        if tm == 'MCI':
            new_teams.append('Man City')
        if tm == 'MUN':
            new_teams.append('Man United')
        if tm == 'LEI':
            new_teams.append('Leicester')
        if tm == 'TOT':
            new_teams.append('Tottenham')
        if tm == 'LIV':
            new_teams.append('Liverpool')
        if tm == 'NEW':
            new_teams.append('Newcastle')
        if tm == 'HUD':
            new_teams.append('Huddersfield')
        if tm == 'FUL':
            new_teams.append('Fulham')
        if tm == 'SOU':
            new_teams.append('Southampton')
        if tm == 'CRY':
            new_teams.append('Crystal Palace')
        if tm == 'CAR':
            new_teams.append('Cardiff')
        if tm == 'EVE':
            new_teams.append('Everton')
        if tm == 'BHA':
            new_teams.append('Brighton')
        if tm == 'BUR':
            new_teams.append('Burnley')
        if tm == 'WOL':
            new_teams.append('Wolves')
        if tm == 'WAT':
            new_teams.append('Watford')
        if tm == 'AVL':
            new_teams.append('Aston Villa')
        if tm == 'MID':
            new_teams.append('Middlesbrough')
        if tm == 'HUL':
            new_teams.append('Hull')
        if tm == 'SWA':
            new_teams.append('Swansea')
        if tm == 'WBA':
            new_teams.append('West Brom')
        if tm == 'STK':
            new_teams.append('Stoke')
        if tm == 'SUN':
            new_teams.append('Sunderland')
        if tm == 'NOR':
            new_teams.append('Norwich')
        if tm == 'SHU':
            new_teams.append('Sheffield United')
    return(new_teams)

def team_code(team_code, season="2016/2017"):
    teams = []
    if season == "2016/2017":
        codes = pd.read_csv("../data/team_id_20162017.csv")
        for i in range(len(team_code)):
            teams.append(codes.loc[codes.index[np.where(codes['id'] == team_code[i])[0][0]], 'Team'])
    if season == "2017/2018":
        codes = pd.read_csv("../data/team_id_20172018.csv")
        for i in range(len(team_code)):
            teams.append(codes.loc[codes.index[np.where(codes['id'] == team_code[i])[0][0]], 'Team'])
    if season == "2018/2019":
        codes = pd.read_csv("../data/team_id_20182019.csv")
        for i in range(len(team_code)):
            teams.append(codes.loc[codes.index[np.where(codes['id'] == team_code[i])[0][0]], 'Team'])
    if season == "2019/2020":
        codes = pd.read_csv("../data/team_id_20192020.csv")
        for i in range(len(team_code)):
            teams.append(codes.loc[codes.index[np.where(codes['id'] == team_code[i])[0][0]], 'Team'])
    return(np.array(teams))

def update_goals_simplex(prior_a, prior_b, goals_perc, games, ff=1):
    return((prior_a * ff) + (goals_perc * games), (prior_b * ff) + games - (goals_perc * games))

def update_assists_simplex(prior_a, prior_b, assists_perc, games, ff=1):
    return((prior_a * ff) + (assists_perc * games), (prior_b * ff) + games - (assists_perc * games))

def update_goals_and_assists_simplex(prior_a, prior_b, prior_c, goals, assists, total_goals, ff=1):
    return((prior_a * ff) + goals, (prior_b * ff) + assists, (prior_c * ff) + (total_goals - (assists + goals)))

def update_mins_simplex(prior_a, prior_b, mins_per_season, games_played, ff=1):
    return((prior_a * ff) + mins_per_season, (prior_b * ff) + games_played)

def update_tir_simplex(prior_a, prior_b, tir_per_season, games_played, ff=1):
    return((prior_a * ff) + tir_per_season, (prior_b * ff) + games_played)

def update_games_played_simplex(prior_a, prior_b, games_played, games_not_played, ff=1):
    return((prior_a * ff) + games_played, (prior_b * ff) + games_not_played)

def name_replace(names):
    replacements = []
    for i in range(names):
        names[i].replace(" ", "_")
        replacements.append(names[i])
    return(np.array(replacements))

def sample_clean_sheet_for_team(lambda_2):
    return(np.random.poisson(lambda_2) == 0)

def sample_mins_played(a, b, a_games, b_games, starting=None):
    if starting == None:
        return(np.random.choice([1, 0], 1, p=np.random.dirichlet([a_games, b_games]))[0] * np.min([np.random.poisson(np.random.gamma(a, 1 / b)), 90]))
    elif starting == True:
        return(np.min([np.random.poisson(np.random.gamma(a, 1 / b)), 90]))
    elif starting == False:
        return(0)

def sample_tir_points(a, b, mins_played):
    return(np.floor(np.random.poisson(mins_played * np.random.gamma(a, 1 / b)) / 2.0))

def sample_goals_and_assists(a, b, c, n, mins_played):
    t_goals = np.random.uniform(0, 90, n)  # times of goals
    n_goals = np.sum(t_goals <= mins_played)
    d = np.random.dirichlet(np.array([a, b, c]))  # sample hyperparameters
    if (n_goals > 0):
        samples = np.random.choice(np.array([0, 1, 2]), n_goals, p=d)
        return(sum(samples == 0), sum(samples == 1))
    else:
        return(0, 0)

def sample_clean_sheet_points(goals_conceded, mins_played):
    return(int(mins_played > 60) * int(goals_conceded == 0))

def sample_mins_points(mins_played):
    return(int(mins_played > 60) + int(mins_played >= 1))

def get_players(all_players_params, team_sel):
    ind = all_players_params.index[all_players_params['team'] == team_sel]
    return(ind)

def ComputeExpectedPoints(fixtures_list, teams, all_players_params, all_teams_params,
                          zerooutbottom=0, Niter=250, additionalstats=False):
    
    # param data sets are pd Data Frames
    
    # Without bonus for now and goals conceded points
    
    # preallocate points for each iteration
    points = np.zeros((Niter, len(all_players_params.index)))
    clean_sheets = np.zeros((Niter, len(all_players_params.index)))
    goals_scored = np.zeros((Niter, len(all_players_params.index)))
    mplayed = np.zeros((Niter, len(all_players_params.index)))
    kb = np.zeros((Niter, len(all_players_params.index)))

    # mean and std of team hyperparameters
    intercept = (all_teams_params.as_matrix())[0, :]
    mu = (all_teams_params.as_matrix())[1, :]
    a = (all_teams_params.as_matrix())[2:(len(teams) + 2), :]
    d = (all_teams_params.as_matrix())[(len(teams) + 2):((2 * len(teams)) + 2), :]

    for l in range(Niter):
        
        for i in range(np.shape(fixtures_list)[0]):
            
            # find teams for each fixture
            h_team = fixtures_list[i, 0]
            a_team = fixtures_list[i, 1]
            
            # sample lambdas for team performance - sample from team hyperparameters
            lambdas = predict_fixtures(np.reshape(fixtures_list[i, :2], ((1, 2))),
                                       teams, intercept, mu, a, d, uncertainty=True)
            
            # sample score
            h_n = np.random.poisson(lambdas[0][0])
            a_n = np.random.poisson(lambdas[1][0])
            
            # find all players in these teams
            h_players = get_players(all_players_params, h_team)
            a_players = get_players(all_players_params, a_team)
			
            # player indicies for bonus points
            b_points_scalings = np.array([7, 8, 10])  # based on previous seasons thresholds for bonus points
            
            # loop over players
            for j in range(len(h_players)):
                
                # predefine scalings for points
                if all_players_params.loc[all_players_params.index[h_players[j]], 'position'] == "GKP":
                    scaling = np.array([6., 3., 4., 0.])
                if all_players_params.loc[all_players_params.index[h_players[j]], 'position'] == "DEF":
                    scaling = np.array([6., 3., 4., 0.])
                if all_players_params.loc[all_players_params.index[h_players[j]], 'position'] == "MID":
                    scaling = np.array([5., 3., 1., 1.])
                if all_players_params.loc[all_players_params.index[h_players[j]], 'position'] == "FWD":
                    scaling = np.array([4., 3., 0., 1.])
                
                # sample mins played
                mins_played = sample_mins_played(all_players_params.loc[all_players_params.index[h_players[j]], 'a_mins'],
                                                 all_players_params.loc[all_players_params.index[h_players[j]], 'b_mins'],
                                                 all_players_params.loc[all_players_params.index[h_players[j]], 'a_games'],
                                                 all_players_params.loc[all_players_params.index[h_players[j]], 'b_games'])
                
                goa, assi = sample_goals_and_assists(all_players_params.loc[all_players_params.index[h_players[j]], 'a_goals'],
                                                         all_players_params.loc[all_players_params.index[h_players[j]], 'b_goals'],
                                                         all_players_params.loc[all_players_params.index[h_players[j]], 'c_goals'],
                                                         h_n, mins_played)
                
                tir_points = sample_tir_points(all_players_params.loc[all_players_params.index[h_players[j]], 'a_tir'],
                                               all_players_params.loc[all_players_params.index[h_players[j]], 'b_tir'], mins_played) * scaling[3]
                gw_points = 0
                gw_points += ((goa * scaling[0]) + (assi * scaling[1]))  # goals and assists
                csp = sample_clean_sheet_points(a_n, mins_played)
                gw_points += ((scaling[2] * csp) + (sample_mins_points(mins_played)))
                points[l, h_players[j]] += gw_points
                points[l, h_players[j]] += np.max(np.concatenate((np.array([0]),
                                                                  1 + np.where((b_points_scalings <= gw_points) == 1)[0]))) + tir_points # kante bonus after bonus
                clean_sheets[l, h_players[j]] += csp
                goals_scored[l, h_players[j]] += goa
                mplayed[l, h_players[j]] += mins_played
                kb[l, h_players[j]] += tir_points
            
            for j in range(len(a_players)):
                
                # predefine scalings for points
                if all_players_params.loc[all_players_params.index[a_players[j]], 'position'] == "GKP":
                    scaling = np.array([6., 3., 4., 0.])
                if all_players_params.loc[all_players_params.index[a_players[j]], 'position'] == "DEF":
                    scaling = np.array([6., 3., 4., 0.])
                if all_players_params.loc[all_players_params.index[a_players[j]], 'position'] == "MID":
                    scaling = np.array([5., 3., 1., 1.])
                if all_players_params.loc[all_players_params.index[a_players[j]], 'position'] == "FWD":
                    scaling = np.array([4., 3., 0., 1.])
                
                # sample mins played
                mins_played = sample_mins_played(all_players_params.loc[all_players_params.index[a_players[j]], 'a_mins'],
                                                 all_players_params.loc[all_players_params.index[a_players[j]], 'b_mins'],
                                                 all_players_params.loc[all_players_params.index[a_players[j]], 'a_games'],
                                                 all_players_params.loc[all_players_params.index[a_players[j]], 'b_games'])
                
                goa, assi = sample_goals_and_assists(all_players_params.loc[all_players_params.index[a_players[j]], 'a_goals'],
                                                         all_players_params.loc[all_players_params.index[a_players[j]], 'b_goals'],
                                                         all_players_params.loc[all_players_params.index[a_players[j]], 'c_goals'],
                                                         a_n, mins_played)
														 
                tir_points = sample_tir_points(all_players_params.loc[all_players_params.index[a_players[j]], 'a_tir'],
                                               all_players_params.loc[all_players_params.index[a_players[j]], 'b_tir'], mins_played) * scaling[3]
                gw_points = 0
                gw_points += ((goa * scaling[0]) + (assi * scaling[1]))  # goals and assists
                csp = sample_clean_sheet_points(h_n, mins_played)
                gw_points += ((scaling[2] * csp) + (sample_mins_points(mins_played)))
                points[l, a_players[j]] += gw_points
                points[l, a_players[j]] += np.max(np.concatenate((np.array([0]),
                                                                  1 + np.where((b_points_scalings <= gw_points) == 1)[0]))) + tir_points  # kante bonus after bonus
                clean_sheets[l, a_players[j]] += csp
                goals_scored[l, a_players[j]] += goa
                mplayed[l, a_players[j]] += mins_played
                kb[l, a_players[j]] += tir_points

        #print('---')
        #print('Realisation ', l)
        #print('Top Points Scorers: ', all_players_params.loc[all_players_params.index[np.argsort(points[l, :])[-5:].astype(int)], 'player'],
        #      ' with ', np.sort(points[l, :])[-5:], ' points')

    if zerooutbottom > 0:
        for l in range(Niter):
            points[l, np.argsort(points[l, :])[:zerooutbottom]] = 0  # to account for bench players when simulating team performance

    expected_points = np.mean(points, axis=0)
    sd_points = np.std(points, axis=0)
    cs = np.mean(clean_sheets, axis=0)
    gs = np.mean(goals_scored, axis=0)
    mp = np.mean(mplayed, axis=0)
    kbm = np.mean(kb, axis=0)

    if additionalstats:
        return(expected_points, sd_points, cs, gs, mp, kbm)
    else:
        return(expected_points, sd_points)

# calculates match probability
def match_prob(exp_p1, sd_p1, exp_p2, sd_p2):
    exp_diff = exp_p1 - exp_p2
    sd_diff = np.sqrt((sd_p1 ** 2) + (sd_p2 ** 2))
    return(1 - norm.cdf(0, exp_diff, sd_diff))

# finds all players in leagues teams and exclude them from overall players parameters
def availablePlayers(leagueId, gw):
    
    users, team_ids, teams, matches = getTeam(leagueId, gw)
	
    all_players_params = pd.read_csv("../parameters/all_players_params.csv")
	
    ids = all_players_params.index
	
    all_unavailableplayers = []
    for i in range(len(teams)):
        all_unavailableplayers.extend(teams[i])
	
    for i in range(len(all_unavailableplayers)):
        ids = ids.drop(all_players_params.index[all_unavailableplayers[i] == all_players_params['ID']][0])
	
    return(ids)