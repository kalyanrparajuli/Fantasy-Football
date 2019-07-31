import os
import csv
import numpy as np
import sys
import pandas as pd
from get_api import *

# create csv gameweek data for all players - if they don't appear on a gw - just dont append
def produce_gw_data(gw):

    # loop over all player ids
    filelist = os.listdir("./data/draft_data/players/")
    for i in range(len(filelist)):
    
        # find player id
        player_id = filelist[i].split(".csv")[0]
	
        # read each players data
        player_data = pd.read_csv("./data/draft_data/players/" + filelist[i])
		
        if i == 0:
            colnames = ["_id"]
            colnames.extend(list(player_data))
            gw_data = np.reshape(np.array(colnames), ((1, 1 + len(list(player_data)))))
            gw_data_len = 1 + len(list(player_data))
	
        # if exists - extract gw column
        available_gws = player_data.loc[:, 'gw'].values
        if gw in available_gws:
            player_gw_data = [player_id]
            player_gw_data.extend(player_data.loc[np.where(gw == available_gws)[0].astype(int), :].values[0])
            gw_data = np.concatenate((np.reshape(gw_data, ((np.shape(gw_data)[0], gw_data_len))),
			                          np.reshape(player_gw_data, ((1, gw_data_len)))))
		
    # write csv
	
    if not os.path.exists("data/draft_data/gws"):
        os.mkdir("data/draft_data/gws")
	
    with open('data/draft_data/gws/gw' + str(gw) + '.csv', mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for l in range(np.shape(gw_data)[0]):
            csv_writer.writerow(gw_data[l, :])
    csv_file.close()


' Run '
if __name__ == "__main__":

    get_api_data()
	
    gw = int(sys.argv[1])
    
    produce_gw_data(gw)