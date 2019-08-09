import requests
import json
import os
import csv
import sys
import numpy as np

# set up query
query_matches = """query getHeadToHeadMatches($leagueId: String!, $gameweek: Int!) {
	headToHeadMatches(leagueId: $leagueId, gameweek: $gameweek) {
		_id
		gameweek
		leagueId
		team1 {
			__typename
		}
		team2 {
			__typename
		}
		__typename
	}
	league(_id: $leagueId) {
		_id
		gameweekAverage(gameweek: $gameweek)
		__typename
	}
}
"""

# set up query
query_teams = """query getHeadToHeadMatchById($matchId: String!, $leagueId: String!, $gameweek: Int!) {
	match: headToHeadMatchById(matchId: $matchId) {
	_id
	gameweek
	team1Score
	team2Score
	team1 {
		...TeamFragment
		__typename
	}
	team2 {
		...TeamFragment
		__typename
	}
	__typename
}
league(_id: $leagueId) {
	_id
	name
	gameweekAverage(gameweek: $gameweek)
	 __typename
}
}
fragment TeamFragment on Team {
	_id
	userId
	name
	user {
		_id
		name
		__typename
	}
	gameweekHistory(gameweek: $gameweek) {
		lineup
		subs
		score
		finalLineup
		finalSubs
		autoSubs {
			in
			out
			__typename
		}
		__typename
	}
	gameweekPlayers(gameweek: $gameweek) {
		...PlayerFragment
		__typename
	}
	__typename
}
fragment PlayerFragment on Player {
	_id
	web_name
	element_type_id
	type_name
	team_name
	clubId
	team_name_short
	gameweekPoints: customGameweekPoints(gameweek: $gameweek, leagueId: $leagueId)
	total_points: totalCustomPoints(leagueId: $leagueId)
	rotowireInjuryUpdate {
		injuryType
		status
		returnDate
		__typename
	}
	club {
		_id
		gameweekFixtures: fixtures(gameweek: $gameweek) {
			homeTeam
			awayTeam
			homeTeamShort
			awayTeamShort
			status
			date
			__typename
		}
		__typename
	}
	__typename
}
"""

def getMatchIds(leagueId, gw):
    variables = {"leagueId": leagueId, "gameweek": gw}
    request = {"query": query_matches, "variables": variables}
    response = requests.post("https://draftfantasyfootball.co.uk/graphql", json=request)
    data = response.json()
    ids = []
    for idx in data['data']['headToHeadMatches']:
        ids.append(idx['_id'])
    return(ids)

def getTeam(leagueId, gw, teamId=None):
    
    matchIds = getMatchIds(leagueId, gw)
    all_team_ids = []
    all_teams = []
    all_users = []
    all_matches = []
	
    for i in range(len(matchIds)):
	    
        # get data
        variables = {"matchId": matchIds[i], "leagueId": leagueId, "gameweek": gw}
        request = {"query": query_teams, "variables": variables}
        response = requests.post("https://draftfantasyfootball.co.uk/graphql", json=request)
        data = response.json()
		
        # team1
        all_team_ids.append(data['data']['match']['team1']['_id'])
        all_users.append(data['data']['match']['team1']['user']['name'])
        team = []
        for j in range(len(data['data']['match']['team1']['gameweekPlayers'])):
            team.append(data['data']['match']['team1']['gameweekPlayers'][j]['_id'])
        all_teams.append(team)
		
        # team2
        try:
            data['data']['match']['team2']['_id']
        except TypeError:
            all_matches.append([data['data']['match']['team1']['_id']])
        else:
            all_team_ids.append(data['data']['match']['team2']['_id'])
            all_users.append(data['data']['match']['team2']['user']['name'])
            team = []
            for j in range(len(data['data']['match']['team2']['gameweekPlayers'])):
                team.append(data['data']['match']['team2']['gameweekPlayers'][j]['_id'])
            all_teams.append(team)
            all_matches.append([data['data']['match']['team1']['_id'], data['data']['match']['team2']['_id']])
	
    ind = np.where(teamId == np.array(all_team_ids))[0]
	
    if teamId != None:
        if teamId in all_team_ids:
            if __name__ == "__main__":
                print('Player Ids: ')
                print(all_teams[int(ind[0])])
            return(all_users[int(ind[0])], all_team_ids[int(ind[0])], all_teams[int(ind[0])])
        else:
            print('teamId could not be found')
    else:
        return(all_users, all_team_ids, all_teams, all_matches)

' Run '
if __name__ == "__main__":

    leagueId = sys.argv[1]
    gw = int(sys.argv[2])
    try:
        teamId = sys.argv[3]
    except IndexError:
        getTeam(leagueId, gw)
    else:
        teamId = sys.argv[3]
        getTeam(leagueId, gw, teamId)