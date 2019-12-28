import numpy as np
import scipy as sp
import pandas as pd

from sklearn import datasets as ds
from sklearn import linear_model as lm
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split as tts
from scipy import constants

G = sp.constants.G
c = sp.constants.c
mass_electron = sp.constants.m_e
mass_proton = sp.constants.m_p
mass_neutron = sp.constants.m_n
mass_hygrogen = mass_proton + mass_electron
mass_helium = 2 * mass_hygrogen
mass_moon = 7.342 * 10**22
mass_earth = 5.97237 * 10**24
mass_sun = 1.9885 * 10**30

def velocity_as_function_of_time_and_acceleration (acceleration, time): return(2*acceleration*time)

def momentum_as_function_of_time_and_acceleration (acceleration, time, mass): return( mass * velocity_as_function_of_time_and_acceleration (acceleration, time) )

def distance_from_acceleration(acceleration, time, position = 0): return(acceleration*time**2 + position)

def gravitational_acceleration(mass_Gravitational, radius, mass_Inertial = 1): return(G * mass_Gravitational * mass_Inertial/radius**2)

def radius_independent_acceleration(world_line): return( np.random.uniform(0.0, world_line['Velocity'][len(world_line) - 1]/(2*(len(world_line) - 1))) )

def sample_gravitationally_accelerating_position_velocity_and_momentum(grav_mass, proper_time, initial_radius):
	if(grav_mass < 0):
		print("Gravitational mass cannot be less than 0, None returned.")
		return()
	if(proper_time < 0):
		print("Proper time cannot start before 0, None returned.")
		return()
	if(initial_radius == 0):
		print("Initial radius cannot be 0, None returned.")
		return()
	x = []
	v = []
	p = []
	radius = - 1 * abs(initial_radius)
	for i in range(proper_time):
		x.append(distance_from_acceleration(gravitational_acceleration(grav_mass, radius = radius), i, radius))
		v.append(velocity_as_function_of_time_and_acceleration (gravitational_acceleration(grav_mass, radius = radius), i) )
		p.append(momentum_as_function_of_time_and_acceleration( gravitational_acceleration(grav_mass, radius = radius), i , grav_mass))
		radius = x[i]
		if(x[i] >= 0): break
	df = pd.DataFrame(data = { 'Position' : x, 'Velocity' : v, 'Momentum' : p })
	df.loc[:,'Proper_Time'] = df.index
	df.loc[:,'Radius_Dependent'] = np.array([True] * len(df.index))
	return(df)

def sample_radius_independent_accelerating_position_velocity_and_momentum(grav_mass, proper_time, initial_radius, radius_dependent_world_line):
	if(grav_mass < 0):
		print("Gravitational mass cannot be less than 0, None returned.")
		return()
	if(proper_time < 0):
		print("Proper time cannot start before 0, None returned.")
		return()
	if(initial_radius == 0):
		print("Initial radius cannot be 0, None returned.")
		return()
	x = []
	v = []
	p = []
	radius = - 1 * abs(initial_radius)
	for i in range(proper_time):
		x.append(distance_from_acceleration(radius_independent_acceleration(radius_dependent_world_line), i, radius))
		v.append(velocity_as_function_of_time_and_acceleration (radius_independent_acceleration(radius_dependent_world_line), i) )
		p.append(momentum_as_function_of_time_and_acceleration( radius_independent_acceleration(radius_dependent_world_line), i , grav_mass))
		radius = x[i]
		if(x[i] >= 0): break
	df = pd.DataFrame(data = { 'Position' : x, 'Velocity' : v, 'Momentum' : p })
	df.loc[:,'Proper_Time'] = df.index
	df.loc[:,'Radius_Dependent'] = np.array([False] * len(df.index))
	return(df)

def sample_radius_independent_and_dependent_accelerating_position_velocity_and_momentum(grav_mass, proper_time, initial_radius):
	df1 = sample_gravitationally_accelerating_position_velocity_and_momentum(grav_mass, proper_time, initial_radius)
	df2 = sample_radius_independent_accelerating_position_velocity_and_momentum(grav_mass, proper_time, initial_radius, df1)
	return((df1,df2))

def build_and_score_model(grav_mass, proper_time, initial_radius, n_neighbors = 5):
	WL1, WL2 = sample_radius_independent_and_dependent_accelerating_position_velocity_and_momentum(grav_mass, proper_time, initial_radius)
	data = pd.concat([WL1,WL2]).sample(frac=1)
	X = np.array(data.loc[:, 'Position':'Proper_Time'])
	Y = np.array(data.loc[:, 'Radius_Dependent'])
	nn = KNN(n_neighbors)
	X_train, X_test, Y_train, Y_test = tts(X, Y, stratify = Y)
	nn.fit(X_train,Y_train)
	return(nn.score(X_test, Y_test))

def parameters_significant_at_sigma(grav_mass, proper_time, initial_radius, n_neighbors = 5, sigma = 5):
	scores = []
	for i in range(100): scores.append(build_and_score_model(grav_mass, proper_time, initial_radius, n_neighbors))
	return(scores > sigma*np.std( a = scores, ddof = 1) + .5)

	