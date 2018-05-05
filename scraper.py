import pandas as pd
import sqlite3
from tqdm import tqdm # For some sick loading bars 8D
import sys
import statistics
import math

statcastdb = sqlite3.connect('raw2017.db') # Defining statcast database for MLB 2017 season

# Defining all 30 MLB team abbreviations
teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CIN', 
		'CLE', 'COL', 'CWS', 'DET', 'HOU', 'KC', 
		'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 
		'NYY', 'OAK', 'PHI', 'PIT', 'SD', 'SEA', 
		'SF', 'STL', 'TB', 'TEX', 'TOR','WSH']

refresh = False
normalize = False

for arg in sys.argv:
	if arg == '-r':
		refresh = True
	elif arg == '-n':
		normalize = True

if refresh: # Checking for flag
	print("Scraping database...") # Begin scraping from mlb.com
	for year in tqdm(range(2017, 2018), desc = 'Years scraped'): # Currently capping year at 2017
		for team in tqdm(teams, desc = 'Teams scraped', leave = False): # Currently iterating through all 30 MLB teams
			# Link is concatenated with team and year in ID locations
			link = 'https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&hfGT=&hfC=&hfSea=' + str(year) + '%7C&hfSit=&player_type=pitcher&hfOuts=&opponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt=&game_date_lt=&team=' + team + '&position=&hfRO=&home_road=&hfFlag=&metric_1=&hfInn=&min_pitches=0&min_results=0&group_by=name-event&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc&min_abs=0&type=details&'
			data = pd.read_csv(link, low_memory=False) # Error if no low_memory    
			data.rename(columns={'player_name': 'pitcher_name'}, inplace=True) # Rename player_name to denote that it is the pitcher
			pd.io.sql.to_sql(data, name='statcast', con=statcastdb, if_exists='append') # Insert to table
	statcastdb.commit()
	print("Processing database...") # Make adjustments to database
	statcastdb.execute("DELETE FROM statcast WHERE rowid NOT IN (SELECT MIN(rowid) FROM statcast GROUP BY sv_id, batter, pitcher)") # I forgot what this does
	statcastdb.execute("DELETE FROM statcast WHERE description = 'foul'") # Get rid of fouls
	statcastdb.execute("ALTER TABLE statcast ADD batter_name TEXT") # Add column for batter names
	statcastdb.execute("UPDATE statcast SET description='hit_into_play' WHERE description='hit_into_play_score'") # Merge hits and hits+RBI's
	statcastdb.execute("DELETE FROM statcast WHERE description!='hit_into_play'") # Delete all non-hits
	statcastdb.commit()
	#statcastdb.close()
	print("Database done processing!")

allBatterEntries = statcastdb.execute("SELECT batter FROM statcast").fetchall() # Get all batters
uniqueBatterEntriesSet = set(allBatterEntries) # Filter list of batters such that each batter is there once by making it a set
uniqueBatterEntriesList = list(uniqueBatterEntriesSet) # Convert set to list
playerIDLink = 'http://crunchtimebaseball.com/master.csv' # Set link to scrape
playerID = pd.read_csv(playerIDLink, low_memory=False, encoding='latin-1') # Error if no low_memory
playerIDNumber = list(playerID["mlb_id"]) # List of player ID's
playerIDName = list(playerID["mlb_name"]) # Corresponding list of player names

for batterEntry in tqdm(uniqueBatterEntriesList, desc = 'Player IDs updated', leave = False): # For each batter
	try:
		row = playerIDNumber.index(batterEntry[0]) # Get row index for batter
		# Replace player's ID with player's name
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
	except:
		#print(str("Minor leaguer: " + batterEntry[0]) + " was omitted!")

statcastdb.commit()

statcastdb.execute("CREATE TABLE IF NOT EXISTS statcastNormalized AS SELECT * FROM statcast")	# Clones database if the database doesn't already exist

if normalize: # Checking for flag
	# Attributes to normalize
	normalizeAttributes = ['ax', 'ay', 'az', 'hc_x', 'hc_y', 'launch_angle', 'launch_speed', 'pfx_x', 'pfx_z', 
	'hit_distance_sc', 'vx0', 'vy0', 'vz0', 'release_pos_x', 'release_pos_y', 'release_pos_z']
	for attribute in tqdm(normalizeAttributes, desc = 'Attributes normalized: ', leave = True): # For each attribute
		attributeData = statcastdb.execute("SELECT `index`," + attribute + " FROM statcastNormalized").fetchall() # Get list of entries for attribute
		attributeDataComplete = statcastdb.execute("SELECT " + attribute + " FROM statcastNormalized WHERE " + attribute + " IS NOT NULL").fetchall() # Get list of entries for attribute that aren't null
		attributeDataComplete = list(sum(attributeDataComplete, ())) # Filter list such that each tuple entry is one string entry

		median = statistics.median(attributeDataComplete) # Get median value for attribute
		mean = 0 # Init mean
		for entry in attributeData: # Calculating total value for attribute
			if entry[1] != None:
				mean += entry[1]
			else: # Impute
				mean += median
		mean /= len(attributeData) # Divide by total to get mean
		
		squaredDifferences = 0 # Init squared differences
		missingSquaredDifference = (median - mean) ** 2 # Calculate squared difference since it's the same for all missing values
		for entry in attributeData: # Calculating standard deviation
			if entry[1] != None: # Calculate squared difference if value is not missing
				squaredDifferences += (entry[1] - mean) ** 2 # (value - mean)^2
			else: #impute
				squaredDifferences += missingSquaredDifference
		standardDeviation = math.sqrt(squaredDifferences/len(attributeData)) # Take sqrt of entire thing

		missingZScore = (median - mean) / standardDeviation # the Z score for missing values is always the same, so calculate beforehand
		for entry in tqdm(attributeData, desc = 'Normalizing entries: ', leave = True): # Calculate and set Z score for each entry
			if entry[1] != None: # Calculate Z score if value isn't missing
				zScore = (entry[1] - mean) / standardDeviation
			else:
				zScore = missingSquaredDifference
			#print(attribute + " | " + str(zScore) + " | " + str(entry[0]))
			statcastdb.execute("UPDATE statcastNormalized SET " + attribute + "=? WHERE `index`=?",(zScore,entry[0])) # Replace value for attribute with the Z-score
statcastdb.commit()