import os
import sys
from get_team_api import *
from get_api import *
import pandas as pd
sys.path.append('./models')
from utils import *
from scipy.stats import mode
import seaborn as sns

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
    tm_median = np.zeros(len(team_ids))
    tm_75 = np.zeros(len(team_ids))
    for j in range(len(team_ids)):

        tm_players_id = getTeam(leagueId, gw, team_ids[j])[2]
        tm_players = []
        for i in range(len(tm_players_id)):
            tm_players.append(all_players_params.loc[all_players_params.index[all_players_params['ID'] == tm_players_id[i]], 'player'].values[0])

        new_players_frame = all_players_params[all_players_params['player'].isin(tm_players)]
        new_players_frame = new_players_frame.reset_index()
		
        tm_s = 0
        tm_sq = 0
        scores = np.zeros(niter)
        CS = 0
        SS = 0

        for k in range(niter):
		
            C, S = ComputeExpectedPoints(fixture_list_this_gw, teams, new_players_frame, all_teams_params, Niter=1, zerooutbottom=(np.max([len(new_players_frame.index) - 11, 0])))

            tm_s += np.sum(C)
            scores[k] = np.sum(C)
            tm_sq += np.sum(C) ** 2
			
            CS += C
            SS += C ** 2
		
        print('Team Name: ', users[j])
        print(pd.DataFrame({"Player": new_players_frame['player'], "Expected Points": CS / niter, "Std Points": np.sqrt((SS / niter) - ((CS / niter) ** 2))}).head(20))
		
        tm_exp[j] = tm_s / niter
        tm_std[j] = np.sqrt((tm_sq / niter) - (tm_exp[j] ** 2))
        tm_median[j] = np.median(scores)
        tm_75[j] = np.quantile(scores, 0.75)
	
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
    df = pd.DataFrame({ "Id": team_ids, "User": users, "Median Score": tm_median, "75% Q": tm_75, "Exp Score": tm_exp, "Std Score": tm_std, "Win Percentage": win_perc * 100})

    print('===========================================================================================')
    print('===========================================================================================')
    print(df)
	
	# plotting match outcomes
    users = list(users)
    users.append('Average Team')
    tm_exp = list(tm_exp)
    tm_std = list(tm_std)
    tm_exp.append(np.mean(np.array(tm_exp)))
    tm_std.append(np.std(np.array(tm_exp)) / np.sqrt(len(tm_exp)))
    for i in range(len(matches)):
        indss = []
        if len(matches[i]) == 1:
            indss.append(np.where(matches[i][0] == np.array(team_ids))[0][0])
            indss.append(len(users) - 1)
            tm1 = users[indss[0]]
            tm2 = users[indss[1]]
        else:
            indss.append(np.where(matches[i][0] == np.array(team_ids))[0][0])
            indss.append(np.where(matches[i][1] == np.array(team_ids))[0][0])
            tm1 = users[indss[0]]
            tm2 = users[indss[1]]
        p1=sns.kdeplot(np.random.normal(tm_exp[indss[0]], tm_std[indss[0]], 100000), shade=True, color="r", bw=0.5)
        p1=sns.kdeplot(np.random.normal(tm_exp[indss[1]], tm_std[indss[1]], 100000), shade=True, color="b", bw=0.5)
        plot.xlabel("points")
        plot.title(users[indss[0]] + " (red) vs " + users[indss[1]] + " (blue)")
        plot.ylabel('frequency density')
        plot.show()


' Run '
if __name__ == "__main__":

    leagueId = sys.argv[1]
    gw = int(sys.argv[2])
    niter = int(sys.argv[3])
    
    simulate_team_points(leagueId, gw, niter)