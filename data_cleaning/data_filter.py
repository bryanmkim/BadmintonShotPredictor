import pandas as pd
# Load the CSV file
raw_data = pd.read_csv('og_train.csv')

# Slice specific columns by their exact names
# rally will be used 
sliced_df = raw_data[['rally_id', 'ball_round', 'player', 'type', 'landing_x', 'landing_y', 'rally_length']]
sliced_df.to_csv('train.csv', index=False)


