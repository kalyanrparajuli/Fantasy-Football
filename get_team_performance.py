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
    users = getTeam(leagueId, gw)[0]
    team_ids = getTeam(leagueId, gw)[1]
	
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
		
        C, S = ComputeExpectedPoints(fixture_list_this_gw, teams, new_players_frame, all_teams_params, zerooutbottom=3, Niter=niter)

        tm_exp[j] = np.sum(C)
        tm_std[j] = np.sqrt(np.sum(S ** 2))
	
    # display (and team ids)
    df = pd.DataFrame({ "Id": team_ids, "User": users, "Expected Score": tm_exp, "Standard Deviation Score": tm_std})

    print('===========================================================================================')
    print('===========================================================================================')
    print(df)

' Run '
if __name__ == "__main__":

    leagueId = sys.argv[1]
    gw = int(sys.argv[2])
    niter = int(sys.argv[3])
    
    simulate_team_points(leagueId, gw, niter)