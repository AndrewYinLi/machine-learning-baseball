import pandas as pd
import sqlite3
from tqdm import tqdm # For some sick loading bars 8D
import sys

# scikitlearn goes here lol

statcastdb = sqlite3.connect('raw2017.db') # Defining statcast database for MLB 2017 season

# Defining all 30 MLB team abbreviations
teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CIN', 
	    'CLE', 'COL', 'CWS', 'DET', 'HOU', 'KC', 
	    'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 
	    'NYY', 'OAK', 'PHI', 'PIT', 'SD', 'SEA', 
	    'SF', 'STL', 'TB', 'TEX', 'TOR','WSH']

if len(sys.argv) > 1:
	if sys.argv[1] == '-r': 
		print("Scraping database...")

		for year in tqdm(range(2017, 2018), desc = 'Years scraped'): # Currently capping year at 2017
		    for team in tqdm(teams, desc = 'Teams scraped', leave = False): # Currently iterating through all 30 MLB teams
		        # Link is concatenated with team and year in ID locations
		        link = 'https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&hfGT=&hfC=&hfSea=' + str(year) + '%7C&hfSit=&player_type=pitcher&hfOuts=&opponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt=&game_date_lt=&team=' + team + '&position=&hfRO=&home_road=&hfFlag=&metric_1=&hfInn=&min_pitches=0&min_results=0&group_by=name-event&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc&min_abs=0&type=details&'
		        data = pd.read_csv(link, low_memory=False) # Error if no low_memory    
		        data.rename(columns={'player_name': 'pitcher_name'}, inplace=True) # Rename player_name to denote that it is the pitcher
		        pd.io.sql.to_sql(data, name='statcast', con=statcastdb, if_exists='append') # Insert to table

		print("Processing database...")

		statcastdb.commit()
		statcastdb.execute("DELETE FROM statcast WHERE rowid NOT IN (SELECT MIN(rowid) FROM statcast GROUP BY sv_id, batter, pitcher)")
		statcastdb.execute("DELETE FROM statcast WHERE description = foul")
		statcastdb.execute("ALTER TABLE statcast ADD batter_name TEXT")
		statcastdb.execute("UPDATE statcast SET description='hit_into_play' WHERE description='hit_into_play_score'")
		statcastdb.execute("DELETE FROM statcast WHERE description!='hit_into_play'")
		statcastdb.commit()
		#statcastdb.close()
		print("Database done processing!")

allBatterEntries = statcastdb.execute("SELECT batter FROM statcast").fetchall()
uniqueBatterEntriesSet = set(allBatterEntries)
uniqueBatterEntriesList = list(uniqueBatterEntriesSet)
playerIDLink = 'http://crunchtimebaseball.com/master.csv'
playerID = pd.read_csv(playerIDLink, low_memory=False, encoding='latin-1') # Error if no low_memory
playerIDNumber = list(playerID["mlb_id"])
playerIDName = list(playerID["mlb_name"])

for batterEntry in tqdm(uniqueBatterEntriesList, desc = 'Player IDs updated', leave = False):
	row = playerIDNumber.index(batterEntry[0])
	#print(playerIDName[row] + " | " + str(playerIDNumber[row]))
	statcastdb.execute("UPDATE statcast SET batter_name=? WHERE batter=?", (playerIDName[row],playerIDNumber[row]))
	statcastdb.execute("UPDATE statcast SET pos1_person_id=? WHERE pos1_person_id=?", (playerIDName[row],playerIDNumber[row]))
	statcastdb.execute("UPDATE statcast SET pos2_person_id=? WHERE pos2_person_id=?", (playerIDName[row],playerIDNumber[row]))
	statcastdb.execute("UPDATE statcast SET pos3_person_id=? WHERE pos3_person_id=?", (playerIDName[row],playerIDNumber[row]))
	statcastdb.execute("UPDATE statcast SET pos4_person_id=? WHERE pos4_person_id=?", (playerIDName[row],playerIDNumber[row]))
	statcastdb.execute("UPDATE statcast SET pos5_person_id=? WHERE pos5_person_id=?", (playerIDName[row],playerIDNumber[row]))
	statcastdb.execute("UPDATE statcast SET pos6_person_id=? WHERE pos6_person_id=?", (playerIDName[row],playerIDNumber[row]))
	statcastdb.execute("UPDATE statcast SET pos7_person_id=? WHERE pos7_person_id=?", (playerIDName[row],playerIDNumber[row]))
	statcastdb.execute("UPDATE statcast SET pos8_person_id=? WHERE pos8_person_id=?", (playerIDName[row],playerIDNumber[row]))
	statcastdb.execute("UPDATE statcast SET pos9_person_id=? WHERE pos9_person_id=?", (playerIDName[row],playerIDNumber[row]))
	statcastdb.commit()
	
	
# NOW WE DO THE MACHINE LEARNING STUFF!!!!!!
