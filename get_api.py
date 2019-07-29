import requests
import json
import os
import csv
import numpy as np

# set up query
query_full = """query getPlayersFull($scoring: ScoringInput, $leagueId: String) {
	players {
		_id
		web_name
		first_name
		second_name
		element_type_id
		team_id
		type_name
		team_name
		team_name_short
		gameweek_totals
		rating
		total_mins
		total_goals
		total_assists
		total_clean_sheets
		total_goals_conceded
		total_own_goals
		total_penalties_saved
		total_penalties_missed
		total_yellow_cards
		total_red_cards
		total_saves
		total_penalties_earned
		total_penalties_conceded
		total_crosses
		total_key_passes
		total_big_chances_created
		total_clearances
		total_blocks
		total_interceptions
		total_tackles
		total_recoveries
		total_errors_leading_to_goal
		total_own_goal_earned
		total_pass_completion
		total_shots
		total_was_fouled
		total_accurate_pass_percentage
		total_shots_on_target
		total_aerial_won
		total_touches
		total_dribbles
		total_dispossessed
		total_fouls
		total_bps
		total_tir_points
		total_bonus
		total_points: totalCustomPoints(scoring: $scoring, leagueId: $leagueId)
		draft_points_per_90min
		draft_points_per_game
		averageDraftPosition
		rotowireInjuryUpdate {
			injuryType
			status
			returnDate
			__typename
			}
		__typename
		}
	clubs {
	_id
	id
	name
	__typename
	}
}"""


query_player = """query getPlayer($playerId: String!, $leagueId: String) {
	player(_id: $playerId) {
		_id
		first_name
		second_name
		web_name
		team_name
		type_name
		rating
		team_name_short
		team_id
		total_points: totalCustomPoints(leagueId: $leagueId)
		total_tir_points
		total_bps
		total_bonus
		draft_points_per_game
		form5
		form10
		total_mins
		total_goals
		total_assists
		total_clean_sheets
		total_goals_conceded
		total_own_goals
		total_penalties_saved
		total_penalties_missed
		total_yellow_cards
		total_red_cards
		total_saves
		total_interceptions
		total_tackles
		total_recoveries
		rotowireInjuryUpdate {
			injuryType
			status
			returnDate
			__typename
		}
		rotowireNews {
			_id
			rotowireId
			date
			priority
			headline
			notes
			__typename
		}
		seasonHistories {
		season
		mp
		gs
		a
		cs
		gc
		og
		ps
		pm
		yc
		rc
		s
		p
		tir
		pe
		pc
		cr
		kp
		bcc
		cl
		b
		i
		t
		r
		eltg
		oge
		pac
		sh
		wf
		bps
		bonus
		__typename
	}
	fixtures_data {
		fixtureId
		optaFixtureId
		date
		gw
		vs
		started
		squad
		mp
		gs
		a
		cs
		gc
		og
		ps
		pm
		yc
		rc
		s
		i
		r
		t
		tir
		bps
		bonus
		p: customPoints(leagueId: $leagueId) __typename
		}
		club {
			_id
			name
			fixtures(onlyUnplayed: true) {
				homeTeam
				awayTeam
				date
				homeScore
				awayScore
				gameweek
				__typename
			}
			__typename
		}
		__typename
	}
}
"""

# set up request for all players 
request = {"query": query_full}
response = requests.post("https://draftfantasyfootball.co.uk/graphql", json=request)

# get data for all players
data = response.json()

# retrieve player data
keys_players = []
for key in data['data']['players'][0]:
    keys_players.append(key)

# extract all player data
data_raw = []
data_raw.append(keys_players)
player_ids = []
for i in range(len(data['data']['players'])):
    data_raw_i = []
    for j in range(len(keys_players)):
        data_raw_i.append(data['data']['players'][i][keys_players[j]])
    player_ids.append(data['data']['players'][i]['_id'])
    data_raw.append(data_raw_i)

data_raw = np.array(data_raw)

if not os.path.exists("data/draft_data"):
    os.mkdir("data/draft_data")

# write to csv
with open('data/draft_data/draft_player_raw.csv', mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    for i in range(np.shape(data_raw)[0]):
        csv_writer.writerow(data_raw[i, :])
csv_file.close()

# set up request for individual players
def getPlayerData(id):
    variables = {"playerId": id}
    request = {"query": query_player, "variables": variables}
    response = requests.post("https://draftfantasyfootball.co.uk/graphql", json=request)
    data = response.json()
    return(data)

# save data for gw-by-gw
keys_fix = []
for key in getPlayerData(player_ids[0])['data']['player']['fixtures_data'][0]:
    keys_fix.append(key)


if not os.path.exists("data/draft_data/players"):
    os.mkdir("data/draft_data/players")

for i in range(len(player_ids)):
    
    player_data = getPlayerData(player_ids[i])['data']['player']['fixtures_data']
	
    print(player_ids[i])
    
    data_raw = []
    data_raw.append(keys_fix)
    for j in range(len(player_data)):
        data_raw_i = []
        for k in range(len(keys_fix)):
            data_raw_i.append(player_data[j][keys_fix[k]])
        data_raw.append(data_raw_i)
    data_raw = np.array(data_raw)

    with open('data/draft_data/players/' + player_ids[i] + '.csv', mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for l in range(np.shape(data_raw)[0]):
            csv_writer.writerow(data_raw[l, :])
    csv_file.close()