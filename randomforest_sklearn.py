import pandas as pd
import sqlalchemy 
import numpy as np
from sklearn.preprocessing import Imputer
from utils import CategoricalEncoder
from sklearn.pipeline import FeatureUnion
from utils import DataFrameSelector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score
import random
from copy import deepcopy
from sklearn.ensemble import BaggingClassifier
from sklearn import model_selection

def main():

	pd.options.mode.chained_assignment = None

	# First we're going to import the database
	db = "raw2017.db"
	table = "statcast"

	engine = sqlalchemy.create_engine("sqlite:///%s" % db, execution_options={"sqlite_raw_colnames": True})
	data = pd.read_sql_table(table, engine)

	# Remove irrelevant features
	del data['des']
	del data['at_bat_number']
	del data['pitch_number']
	data = data[[col for col in data.columns if 'deprecated' not in col]]

	del data['spin_dir']
	del data['umpire']

	del data['type']  # We don't know what this feature is, but they're almost all 'X' so it's irrelevant

	del data['game_year']
	del data['pos2_person_id.1']  # This is a duplicate

	del data['game_pk']  # This is just a game ID

	# We're going to make balls and strikes discrete so the algorithm doesn't define a threshold 
	# that will lose a lot of information
	data['balls'] = data['balls'].astype(str)
	data['strikes'] = data['strikes'].astype(str)

	# BABIP (Batting Average on Balls In Play) is an important stat, but it appears we extracted it wrong
	# so we'll drop it for now
	# TODO Extract BABIP correctly
	del data['babip_value']


	# TODO try models with hit_location and innings (and any other stats with similar distributions) as 
	# both discrete and continuous attributes
	# TODO discuss how to normalize features
	del data['index']

	# Discretize runners on base as well as outs_when_up, pitcher
	for feature in [col for col in data.columns if col.startswith('on')] + ['outs_when_up']:
	    data[feature] = data[feature].astype(str)

	# TODO properly collect ISO and WOBA values
	del data['iso_value']
	del data['woba_denom']
	del data['woba_value']

	hit_types = {'single', 'double', 'triple', 'home_run'}
	labels = np.array(['hit' if event in hit_types else 'not_hit' for event in data['events']])
	del data['events']

	# Further discretization
	data['hit_location'] = data['hit_location'].astype(str)
	data['inning'] = data['inning'].astype(str)
	del data['launch_speed_angle']  # This data appears broken, plus we have it elsewhere
	del data['zone']                # We're not sure what this is lmao
	del data['description']         # TMI
	del data['estimated_ba_using_speedangle']   # These stats are generally useless
	del data['estimated_woba_using_speedangle']
	del data['hit_location']        # This is sort of subjective
	del data ['batter'] # We have batter name
	del data ['pitcher'] # We have pitcher name
	del data ['game_type'] # Potentially important

	# We will normalize the data later, but for now, we need to impute missing values
	incomplete_rows = data[data.isnull().any(axis=1)].head()

	imputer = Imputer(strategy='median')
	# Remove non-numerical features so median can be calculated
	# Finna fill those empties with the median
	data_num = data._get_numeric_data()

	X = imputer.fit_transform(data_num)

	data_tr = pd.DataFrame(X, columns=data_num.columns)

	continuous_attrs = np.array(data_num.columns)
	discrete_attrs = set([attr for attr in data.columns if attr not in continuous_attrs])

	discrete_attrs = np.array(list(discrete_attrs))

	# Turn date attribute into month
	data['game_date'] = data['game_date'].apply(lambda s: s[5:7])


	continuous_attrs = np.array(continuous_attrs)

	# TODO Temporary fix for handling missing discrete values
	for attr in discrete_attrs:
	    data[attr] = data[attr].astype(str)

	continuous_pipeline = Pipeline([
	    ('selector', DataFrameSelector(continuous_attrs)), 
	    ('imputer', Imputer(strategy='median')), 
	    ('std_scaler', StandardScaler())
	])

	discrete_pipeline = Pipeline([
	    ('selector', DataFrameSelector(discrete_attrs)),
	    ('cat_encoder', CategoricalEncoder(encoding='ordinal')) # onehot-dense | ordinal
	])

	full_pipeline = FeatureUnion([
	    ('c_pipeline', continuous_pipeline), 
	    ('d_pipeline', discrete_pipeline)
	])

	data_prepared = full_pipeline.fit_transform(data)

	#print(data_prepared)
	
	#data_prepared_split = list()
	#data_prepared_copy = deepcopy(data_prepared)
	#subsetSize = int(len(data_prepared) / 10)
	#for i in range(10): # 10 subsets
	#	subset = list()
	#	while len(subset) < subsetSize:
	#		index = randrange(len(data_prepared_copy))
	#		subset.append(data_prepared_copy.pop(index))
	#	data_prepared_split.append(subset)
	seed = random.randint(0,999999)
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cart = tree.DecisionTreeClassifier()
	baggedTrees = BaggingClassifier(base_estimator=cart, n_estimators=10, random_state=seed)

	decisionForest = baggedTrees.fit(data_prepared, labels)
	scores = cross_val_score(decisionForest, X=data_prepared, y=labels, scoring='accuracy', cv=kfold, n_jobs=4)
	print(pd.Series(scores).describe())
	
	for i, decisionTree in enumerate(decisionForest.estimators_):
		export_graphviz(decisionTree, out_file='tree' + str(i) + '.dot')

if __name__ == "__main__":
    main()
