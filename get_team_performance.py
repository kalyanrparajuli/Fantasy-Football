import os
import sys
from get_team_api import *
from get_api import *
import pandas as pd
sys.path.append('./models')
from utils import *

def simulate_team_points(leagueId, gw, niter=1000):
    
    # parameters
    all_players_params = pd.read_csv("./parameters/all_players_params.csv")
    all_teams_params = pd.read_csv("./parameters/all_teams_params.csv", header=None)
    teams = pd.read_csv("./parameters/all_teams.csv", header=None).values[:, 0]
    
    # fixtures
    fixture_list_this_gw = getFixtures(gw, team_id_file="./data/team_id_20192020.csv")
	
    # find  team ids
    users, team_ids, match_teams, matches = getTeam(leagueId, gw)
	
    # find expected points
    tm_exp = np.zeros(len(team_ids))
    tm_std = np.zeros(len(team_ids))
    for j in range(len(team_ids)):

        tm_players_id = getTeam(leagueId, gw, team_ids[j])[2]
        tm_players = []
        for i in range(len(tm_players_id)):
            tm_players.append(all_players_params.loc[all_players_params.index[all_players_params['ID'] == tm_players_id[i]], 'player'].values[0])

        new_players_frame = all_players_params[all_players_params['player'].isin(tm_players)]
        new_players_frame = new_players_frame.reset_index()
		
        C, S = ComputeExpectedPoints(fixture_list_this_gw, teams, new_players_frame, all_teams_params, Niter=niter, zerooutbottom=(np.max([len(new_players_frame.index) - 11, 0])))

        tm_exp[j] = np.sum(C)
        tm_std[j] = np.sqrt(np.sum(S ** 2))
	
    # display (and team ids)
    match = np.array(matches)
    win_perc = np.zeros(len(team_ids))
    for i in range(len(team_ids)):
        for j in range(len(matches)):
            if team_ids[i] in matches[j]:
                other_ind = np.delete(np.array(matches[j]), np.where(team_ids[i] == np.array(matches[j]))[0])
                if len(other_ind) > 0:
                    win_perc[i] = match_prob(tm_exp[i], tm_std[i], tm_exp[np.where(other_ind[0] == np.array(team_ids))[0].astype(int)], tm_std[np.where(other_ind[0] == np.array(team_ids))[0].astype(int)])
                else:
                    win_perc[i] = match_prob(tm_exp[i], tm_std[i], np.mean(np.delete(tm_exp, i)), np.mean(np.delete(tm_std, i)))
    df = pd.DataFrame({ "Id": team_ids, "User": users, "Expected Score": tm_exp, "Standard Deviation Score": tm_std, "Win Percentage": win_perc * 100})

    print('===========================================================================================')
    print('===========================================================================================')
    print(df)

' Run '
if __name__ == "__main__":

    leagueId = sys.argv[1]
    gw = int(sys.argv[2])
    niter = int(sys.argv[3])
    
    simulate_team_points(leagueId, gw, niter)