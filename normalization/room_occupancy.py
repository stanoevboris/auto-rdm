from ucimlrepo import fetch_ucirepo

# fetch dataset
room_occupancy_estimation = fetch_ucirepo(id=864)

# data (as pandas dataframes)
X = room_occupancy_estimation.data.features
y = room_occupancy_estimation.data.targets

print(y.groupby('Room_Occupancy_Count').count())
# metadata
# print(room_occupancy_estimation.metadata)
print('test')
# variable information
# print(room_occupancy_estimation.variables)
