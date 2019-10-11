# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 10:46:59 2019

@author: dawig
"""
import numpy as np
import tensorflow as tf
import pandas as pd
import random
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from dollargame import GamePlay, Node, CreateAGame  


"""
This section creates a model for decisions in playing the game.
"""

node_choices = pd.read_csv('C:/Users/dawig/Desktop/AUC/dollar_game_performance\
/node_choice_naive.csv', sep=',',header=None)
"""
When this section is used, it learns from new games played by the neural net model.

node_choices_addon = pd.read_csv('C:/Users/dawig/Desktop/AUC/dollar_game_performance\
/node_choice_neural_net_plays.csv', sep=',',header=None)
node_choices = pd.concat([node_choices,node_choices_addon])
"""



node_choices.columns = ['Steps_to_finish','Primary_size','Secondary_size',\
    'Negative_neighbors','Advantage_over_one_deg', 'Advantage_over_two_deg',\
    'Triangle_loops','Give_or_take','Game_won', 'Negative_load_change', \
    'Node_own_dollars', 'Last_same', 'Two_ago_same']
#Nodes are recoded to penalize losing games, then blended with negative load change
node_choices['Steps_to_finish_helper'] = node_choices['Game_won'].\
                                         apply(lambda x: 300 if x == 0 else -5)
node_choices['Steps_to_finish'] = np.where\
   ((node_choices['Steps_to_finish_helper'] > node_choices['Steps_to_finish'])\
      , node_choices['Steps_to_finish_helper'],node_choices['Steps_to_finish'])
node_choices['Steps_to_finish'] = node_choices['Negative_load_change'] * 100 \
                                        + 300 - node_choices['Steps_to_finish']# Formula makes larger numbers for better moves
node_choices.rename(columns={'Steps_to_finish':'Model_target_variable'}, 
                    inplace=True)
node_choices = node_choices.drop(['Steps_to_finish_helper'], axis = 1)
node_choices = node_choices.drop(['Game_won'], axis = 1)
node_choices = node_choices.drop(['Negative_load_change'], axis = 1)


node_choice_stats = node_choices.describe().transpose()
def norm(x):
  return (x - node_choice_stats['mean'])
normed_data = norm(node_choices)
#normed_data = normed_data   used to pare down data for testing
normed_target = normed_data['Model_target_variable']
normed_indep_variables = normed_data.drop(['Model_target_variable'], axis = 1)
def build_model():
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_dim= 10, name='layer_1'),
        layers.Dense(32, activation='relu', name='layer_2'),
        layers.Dense(1, name='layer_3')
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.005)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mse','accuracy'])
    model.fit(normed_indep_variables, normed_target, epochs=150, batch_size=10, verbose=0)
    scores = model.evaluate(normed_indep_variables, normed_target, verbose=0)
    print('scores')
    print(scores)
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
    return model    
"""
When turned on, this section will create a model.

model = build_model()
model.save('huge_modelb.h5')
"""





#-----------------------------
"""
This section plays the game based on a saved model
"""


model = load_model('huge_modelb.h5')

for j in range(1):
    new_test_game = GamePlay()
    for i in range(100):
        if i % 20 == 0:
            new_test_game.print_current_game_board()
        if new_test_game.finished_game_test() == 1:
            new_test_game.print_current_game_board()
            print('finished')
            print(i)
            print('ha!')
            break
        node_states_for_model = new_test_game.give_game_states()
        node_states_for_model = pd.DataFrame(node_states_for_model)
        node_states_for_model.columns = ['Steps_to_finish','Primary_size',\
                   'Secondary_size','Negative_neighbors','Advantage_over_one_deg',\
               'Advantage_over_two_deg','Triangle_loops','Give_or_take','Game_won'\
               ,'Negative_load_change','Node_own_dollars','Last_same',\
               'Two_ago_same']
        normed_node_states = norm(node_states_for_model)
        normed_node_states = normed_node_states.drop(['Steps_to_finish'], axis = 1)
        normed_node_states = normed_node_states.drop(['Model_target_variable'], axis = 1)
        normed_node_states = normed_node_states.drop(['Game_won'], axis = 1)
        normed_node_states = normed_node_states.drop(['Negative_load_change'], axis = 1)
        move_choice = model.predict(normed_node_states)
        new_test_game.neural_net_game_play(np.argmax(move_choice), i)
        print(np.argmax(move_choice))
    #print(len(move_choice))




