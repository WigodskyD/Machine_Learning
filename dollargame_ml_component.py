# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 10:46:59 2019

@author: dawig
"""
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from dollargame import GamePlay, Node, CreateAGame  


print(tf.__version__)

node_choices = pd.read_csv('C:/Users/dawig/Desktop/AUC/dollar_game_performance/node_choice_naive_with_random.csv', sep=',',header=None)
node_choices.columns = ['Steps_to_finish','Primary_size','Secondary_size',\
                               'Negative_neighbors','Advantage_over_one_deg',\
                      'Advantage_over_two_deg','Triangle_loops','Give_or_take','Game_won']
 #Nodes are recoded to penalize losing games
node_choices['Steps_to_finish_helper'] = node_choices['Game_won'].\
                                         apply(lambda x: 400 if x == 0 else -5)
node_choices['Steps_to_finish'] = np.where\
   ((node_choices['Steps_to_finish_helper'] > node_choices['Steps_to_finish'])\
      , node_choices['Steps_to_finish_helper'],node_choices['Steps_to_finish'])
node_choices = node_choices.drop(['Steps_to_finish_helper'], axis = 1)
node_choices = node_choices.drop(['Game_won'], axis = 1)
node_choice_stats = node_choices.describe().transpose()
def norm(x):
  return (x - node_choice_stats['mean'])
normed_data = norm(node_choices)
#normed_data = normed_data   used to pare down data for testing
normed_target = normed_data['Steps_to_finish']
normed_indep_variables = normed_data.drop(['Steps_to_finish'], axis = 1)
def build_model():
    model = keras.Sequential([
        layers.Dense(12, activation='relu', input_dim= 7, name='layer_1'),
        layers.Dense(12, activation='relu', name='layer_2'),
        layers.Dense(1, name='layer_3')
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.005)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mse','accuracy'])
    model.fit(normed_indep_variables, normed_target, epochs=40, batch_size=10, verbose=0)
    scores = model.evaluate(normed_indep_variables, normed_target, verbose=0)
    print('scores')
    print(scores)
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
    return model    

#model = build_model()
#model.save('model_drop_game_won_colmn.h5')
#model = load_model('my_model.h5')
#model.summary()




"""node_choices_second = node_choices[8110:8130]
#node_choices_second = node_choices_second.drop(['Steps_to_finish'], axis = 1)
normed_data_test = norm(node_choices_second)
example_batch = normed_data_test.drop(['Steps_to_finish'], axis = 1)
example_result = model.predict(example_batch)
print(example_result)"""






"""model.save('my_model.h5')

node_choices_second = node_choices[8000:8100]
normed_data_test = norm(node_choices_second)
example_batch = normed_data_test
example_result = model.predict(example_batch)
print('iii')
print(example_result)
print(normed_data_test)"""
model = load_model('model_drop_game_won_colmn.h5')
new_test_game = GamePlay()
"""node_states_for_model = new_test_game.give_game_states()
node_states_for_model = pd.DataFrame(node_states_for_model)
node_states_for_model.columns = ['Steps_to_finish','Primary_size',\
               'Secondary_size','Negative_neighbors','Advantage_over_one_deg',\
           'Advantage_over_two_deg','Triangle_loops','Give_or_take','Game_won']
normed_node_states = norm(node_states_for_model)
normed_node_states = normed_node_states.drop(['Steps_to_finish'], axis = 1)
normed_node_states = normed_node_states.drop(['Game_won'], axis = 1)
move_choice = model.predict(normed_node_states)
print(move_choice)
print(np.argmax(move_choice))
#-----------------------------
new_test_game.neural_net_game_play(np.argmax(move_choice))"""
#-----------------------------
for i in range(100):
    if i % 10 == 0:
        new_test_game.print_current_game_board()
    if new_test_game.finished_game_test() == 1:
        print('ha!')    
    node_states_for_model = new_test_game.give_game_states()
    node_states_for_model = pd.DataFrame(node_states_for_model)
    node_states_for_model.columns = ['Steps_to_finish','Primary_size',\
               'Secondary_size','Negative_neighbors','Advantage_over_one_deg',\
           'Advantage_over_two_deg','Triangle_loops','Give_or_take','Game_won']
    normed_node_states = norm(node_states_for_model)
    normed_node_states = normed_node_states.drop(['Steps_to_finish'], axis = 1)
    normed_node_states = normed_node_states.drop(['Game_won'], axis = 1)
    move_choice = model.predict(normed_node_states)
    new_test_game.neural_net_game_play(np.argmax(move_choice), i)
    print(np.argmax(move_choice))
    #print(len(move_choice))


