# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 13:43:48 2019
This file implements a version of "The Dollar Game", which is a graph-based 
game featured in "Numberphile" on YouTube.

Classes include Node, GamePlay and CreateAGame.  GamePlay is the primary class 
to utilize.

@author: dawig
"""
import random
import string
import csv
import numpy as np
import time
import math

class Node:
    """
    This class holds the basic nodes for the game.
    """
    def __init__(self, node_name, dollar_amount):
        self.__identity = node_name
        self.__connections = []
        self.__money_in_node = dollar_amount
    def add_connection(self,connect_name):
        self.__connections.append(connect_name)
    def get_dollar_amount(self):
        return self.__money_in_node
    def change_dollar_amount(self,amount):
        self.__money_in_node += amount
    def print_node(self):
        print (self.__identity)
        print (self.__connections)
    
class GamePlay:
    """
    This class creates connections and makes game moves.  Games can be created 
    manually or through auto_game using CreateAGame
    """
    def __init__(self):
        self.__node_set = []                                                   #hold created nodes for graph
        self.__node_alias_set = []                                             #hold names to refer to nodes
        self.__connection_set = []
        self.__game_step = 0
        self.__csv_data = []
        self.auto_game()
        self.__game_states = np.zeros([1,9])                                   #This saves game states to feed in to neural network model
    def node_create(self, node_name, dollar_amount):
        self.__node_alias_set.append(node_name)
        new_node = Node(node_name,dollar_amount)
        self.__node_set.append(new_node)
    def connection_create(self, left_node, right_node):                        
        if left_node in self.__node_alias_set and right_node \
                     in self.__node_alias_set:
            connection_created = [left_node, right_node]
            self.__connection_set.append(connection_created)
        else:
            print('Connection between ' + left_node + ' and ' + right_node\
                  +' cannot be created.')
    def give_dollars(self, node_donor):                                        #this gives 1 dollar to each connected node 
        connected_list = self.find_connected(node_donor)                       #and subtracts from donor
        dollars_to_give = len(connected_list)
        self.__node_set[self.__node_alias_set.index(node_donor)].\
                                        change_dollar_amount(0-dollars_to_give)
        for adjacent_node in connected_list:
            self.__node_set[self.__node_alias_set.index(adjacent_node)].\
                                                        change_dollar_amount(1)
        temp.report_node_states(node_donor, 0)
        self.__game_step +=1
    def take_dollars(self, node_donor):                                        #this takes one dollar from each neighbor
        connected_list = self.find_connected(node_donor)                       
        dollars_to_take = len(connected_list)
        self.__node_set[self.__node_alias_set.index(node_donor)].\
                                          change_dollar_amount(dollars_to_take)
        for adjacent_node in connected_list:
            self.__node_set[self.__node_alias_set.index(adjacent_node)].\
                                                       change_dollar_amount(-1)
        temp.report_node_states(node_donor, 1)
        self.__game_step +=1
    def find_connected(self,node_to_search):                                   #This function helps find which nodes to take from or give to.
        connections_containing_x = []
        for set in self.__connection_set:
            if node_to_search in set[1]:
                connections_containing_x.append(set[0])
            if node_to_search in set[0]:
                connections_containing_x.append(set[1])
        return connections_containing_x 
    def enter_game_moves(self, moves):                                         #This allows someone to input instructions for
        for move in moves:                                                     #a completed game.
            if move[1] == 'G':
                self.give_dollars(move[0])
            else:
                self.take_dollars(move[0])
            self.print_current_game_board()
    def print_current_game_board(self):                                        
        print ('game step ' + str(self.__game_step))
        for node in self.__node_alias_set:
            print(node)
            print(self.__node_set[self.__node_alias_set.index(node)].\
                                                           get_dollar_amount())
    def auto_game(self):                                                       #This creates a game automatically
        auto_g = CreateAGame()
        game_creation_plan = auto_g.return_full_list()
        for i in range(0,len(game_creation_plan[0])):
            self.node_create(game_creation_plan[0][i],game_creation_plan[1][i])
        for i in range(0,len(game_creation_plan[2])):
            self.connection_create(game_creation_plan[2][i][0],\
                                                   game_creation_plan[2][i][1])
    def finished_game_test(self):
        for node in self.__node_set:
            if node.get_dollar_amount() < 0:
                return 0
        return 1
    def get_dollars_from_all(self):                                            #find dollar amounts
        dollar_set = []
        for node in self.__node_set:
            dollar_set.append(node.get_dollar_amount())
        return dollar_set
    def naive_game_play(self):                                                 #before using ml, just give and take from outliers to 
        for i in range(1,150):                                                 #play game
            dollar_set = self.get_dollars_from_all()
            most_dollars = dollar_set.index(max(dollar_set))
            least_dollars = dollar_set.index(min(dollar_set))
            if self.finished_game_test() != 1:
                if i % 5 ==0: 
                    self.take_dollars(self.__node_alias_set[least_dollars])
                elif i % 3 ==0:
                    self.take_dollars(self.__node_alias_set[least_dollars])
                    self.give_dollars(self.__node_alias_set[most_dollars])
                else:
                    self.give_dollars(self.__node_alias_set[most_dollars])
            else:
                break
        game_done = 1
        if i == 149:
            game_done = 0
        self.save_states(0, game_done)                                         #Write current game result to file
    def game_play_with_random_moves(self):                                     #Add random moves to teach the neural net
        for i in range(1,150):                                                 #model possibilities for play
            dollar_set = self.get_dollars_from_all()
            most_dollars = dollar_set.index(max(dollar_set))
            least_dollars = dollar_set.index(min(dollar_set))
            if self.finished_game_test() != 1:
                node_number = random.randrange(len(self.__node_alias_set))  
                node_name = self.__node_alias_set[node_number]
                if i % 5 ==0: 
                    self.take_dollars(self.__node_alias_set[least_dollars])
                elif i % 3 ==0:
                    self.take_dollars(self.__node_alias_set[least_dollars])
                    self.give_dollars(self.__node_alias_set[most_dollars])
                elif i % 7 ==0:
                    self.give_dollars(node_name)
                elif i % 11 ==0:
                    self.take_dollars(node_name)
                else:
                    self.give_dollars(self.__node_alias_set[most_dollars])
            else:
                break
        game_done = 1
        if i == 149:
            game_done = 0
        self.save_states(1, game_done)                                         #Write current game result to file
    def game_play_completely_random_moves(self):                               #Add random moves to teach the neural net
        for i in range(1,100):                                                 #model possibilities for play
            game_done = 1
            if i == 99:
                game_done = 0
            if self.finished_game_test() != 1:
                node_number = random.randrange(len(self.__node_alias_set))  
                node_name = self.__node_alias_set[node_number]
                give_or_take = random.randrange(0,20)
                if give_or_take % 10 == 0:
                    self.take_dollars(node_name)
                else:
                    self.give_dollars(node_name)
            else:
                break
        self.save_states(2, game_done) 
    def report_node_states(self, node_choice, give_take):
        """
        The matrix with game node states will contain the following columns:
        1 - size of primary neighbor set including self
        2 - size of secondary neighbor set including self
        3 - number of negative nodes in primary neighbor set
        4 - = self - average of one degree neighbor set
        5 - = self - average of secondary neighbor set
        6 - Number of tight 3-way node connections with a triangle all connected
        7 - give(0) or take(1)
        8 - finish game or not?
        0 - Remaining steps to finish  -- set in finalize_game_states
        """
        self.find_adjacent_nodes(node_choice, give_take)
    def find_adjacent_nodes(self, node_choice, give_take):                                #Find how many nodes are connected 1 and 2 steps 
        node_length_set = []                                                   #away, mean dollar value for connections, 
        add_to_game_states = np.zeros([1,9])                                   # and number of negative connections 
        for i in range(1):                                                                                    
            one_degree_conn_set = [node_choice]                                                        
            one_degree_conn_set = one_degree_conn_set + \
                                  self.find_connected(node_choice)  
            two_degree_conn_set = one_degree_conn_set[:]
            triangle_connections_count = 0
            one_deg_check_marker = 1
            for item in one_degree_conn_set:
                new_connx = self.find_connected(item)                          #Build 2 degree set
                for conn in new_connx:
                    if conn not in two_degree_conn_set:
                        two_degree_conn_set.append(conn)
                for check in one_degree_conn_set[one_deg_check_marker + 1:]:   #Then search for nodes connected in triangle
                    if self.connected_pair_search \
                            (check, one_degree_conn_set[one_deg_check_marker]):
                        triangle_connections_count  += 1
                one_deg_check_marker += 1
            node_length_set.append((len(one_degree_conn_set),   \
                                                     len(two_degree_conn_set)))
            add_to_game_states[i, 1] = len(one_degree_conn_set)
            add_to_game_states[i, 2] = len(two_degree_conn_set)
            add_to_game_states[i, 3] = \
                                    self.neighbor_avg(one_degree_conn_set,i)[1]
            #Next 4 lines for col 4 and 5 - see docstring
            own_dollars = self.__node_set[self.__node_alias_set. \
                           index(node_choice)].get_dollar_amount()
            add_to_game_states[i, 5] = \
                      own_dollars - self.neighbor_avg(two_degree_conn_set,i)[0]
            add_to_game_states[i, 4] = \
                      own_dollars - self.neighbor_avg(one_degree_conn_set,i)[0]
            add_to_game_states[i, 6] = triangle_connections_count
            add_to_game_states[i, 7] = give_take
        if self.__game_states[0][1] != 0:
            self.__game_states = np.append(self.__game_states, add_to_game_states, axis = 0)
        else:
            self.__game_states = add_to_game_states
        #print(len(self.__game_states))
    def finalize_game_states(self, game_save_type, game_done):                            #prepare game state matrix, then save it
        game_steps_remaining = [*range(len(self.__game_states)-1,-1,-1)]
        self.__game_states[:,0] = game_steps_remaining
        self.__game_states[:,8] = game_done
        file_choices = ('C:/Users/dawig/Desktop/AUC/dollar_game_performance\
/node_choice_naive.csv', 'C:/Users/dawig/Desktop/AUC/dollar_game_performance\
/node_choice_naive_with_random.csv', 'C:/Users/dawig/Desktop/AUC/\
dollar_game_performance/node_choice_completely_random.csv')
        file_to_write = file_choices[game_save_type]
        with open(file_to_write, 'a') as csvFile:
            writer = csv.writer(csvFile, lineterminator = '\n')
            writer.writerows(self.__game_states)  
        print(self.__game_states)    
    def neighbor_avg(self,neighborhood_set,i):                                 #We call this above for 1 and 2 degree neighborhoods
        neg_size = 0
        sum_neighborhood = 0
        for deg in neighborhood_set:                                           #Thus calculates the sum of our
            neg_size = neg_size + int(self.__node_set \
                   [self.__node_alias_set.index(deg)].get_dollar_amount() < -1)#node and its neighbors
            sum_neighborhood += self.__node_set \
                         [self.__node_alias_set.index(deg)].get_dollar_amount()
        
        return (sum_neighborhood/len(neighborhood_set), neg_size)
    def save_states(self, game_save_type, game_done):                          #This saves number of step for finished games
        file_choices = ('C:/Users/dawig/Desktop/AUC/dollar_game_performance\
/naive.csv', 'C:/Users/dawig/Desktop/AUC/dollar_game_performance\
/naive_with_random.csv', 'C:/Users/dawig/Desktop/AUC/dollar_game_performance\
/completely_random.csv')
        self.finalize_game_states(game_save_type, game_done)
        self.__csv_data.append((len(self.__node_set),self.__game_step))
        file_to_write = file_choices[game_save_type]
        with open(file_to_write, 'a') as csvFile:
            writer = csv.writer(csvFile, lineterminator = '\n')
            writer.writerows(self.__csv_data)  
    def connected_pair_search(self, first, second):
        for conn in self.__connection_set:
            if first == conn[0] and second == conn[1]:
                return True
            elif first == conn[1] and second == conn[0]:
                return True
        return False  
    def TEMPprint_game(self):
        print (self.__node_set)                                                  
        print (self.__node_alias_set)
        print (self.__connection_set) 
         
 
            
class CreateAGame:
    """
    This class creates a connected 3 lists for a connected graph.  We pass it 
    into GamePlay to start the game.
    """
    def __init__(self):
        self.__connections_to_create = []
        self.__nodes_to_create = string.ascii_uppercase[:20]
        self.__dollars_for_nodes = []
        self.make_lists()
        self.check_for_connectedness()
        self.assign_dollars()
    def make_lists(self):
        seed = time.time()
        seed = seed - math.floor(seed)
        seed = int(seed * 1000000) % 1000000
        random.seed(seed)
        numbers = random.randrange(3,20)                                       #Create from 3 to 20 nodes
        self.__nodes_to_create = self.__nodes_to_create[0:numbers]
        for name in self.__nodes_to_create:
            numbers = random.randrange(0,3)                                    #How many connections to originate with
            possible_nodes_to_connect = self.__nodes_to_create.replace(name,'')#each node -- 0-2
            for conn in self.__connections_to_create:                          #take out previous connections that include new node
                if conn[1] in name:
                    possible_nodes_to_connect = \
                                  possible_nodes_to_connect.replace(conn[0],'')
            for i in range(0,numbers):                                         #Generate node to connect to
                try:
                    connected_index = \
                             random.randrange(0,len(possible_nodes_to_connect)) 
                    connected_node =  possible_nodes_to_connect[connected_index]
                    connection_set = [name,connected_node]
                    possible_nodes_to_connect = \
                           possible_nodes_to_connect.replace(connected_node,'')
                    self.__connections_to_create.append(connection_set)
                except:                                                        #removes difficulty when possible_nodes_to_connect is empty
                    continue
    def check_for_connectedness(self):                                         #remove any nodes or sets that aren't connected
        queue_pointer = 0                                                      #leaving one connected game
        queue_end = 1
        try:
            connected_nodes = [self.__connections_to_create[0][0]]
        except:
            connected_nodes = ['A']
        queue = ['start_next',connected_nodes[0]]      
        while queue_pointer != queue_end:
            node_to_connect = queue[queue_pointer + 1]
            for conn in self.__connections_to_create:                          
                if node_to_connect in conn[0]\
                                            and conn[1] not in connected_nodes:
                    queue.append(conn[1])
                    connected_nodes.append(conn[1])
                    queue_end +=1
                elif node_to_connect in conn[1]\
                                            and conn[0] not in connected_nodes:
                    queue.append(conn[0])
                    connected_nodes.append(conn[0])
                    queue_end +=1
            queue_pointer +=1  
        i = 0
        while i < len(self.__connections_to_create):                           #Get rid of connection for removed nodes
            if self.__connections_to_create[i][0] not in connected_nodes:
                self.__connections_to_create.remove\
                                              (self.__connections_to_create[i])
                i -= 1
            i += 1           
        self.__nodes_to_create = connected_nodes
    def assign_dollars(self):                                                  #assign dollars after 
        for name in self.__nodes_to_create:
            dollar_start = random.randrange(-3,5)                              
            self.__dollars_for_nodes.append(dollar_start)
        euler_number = len(self.__connections_to_create) - \
                                                len(self.__nodes_to_create) + 1
        dollar_deficit = euler_number - sum(self.__dollars_for_nodes) 
        if dollar_deficit > 0:
            self.__dollars_for_nodes[-1] += dollar_deficit
    def return_full_list(self):
        return_list = [self.__nodes_to_create,self.__dollars_for_nodes,\
                                                  self.__connections_to_create]
        return return_list
    
    
for i in range(1000):
    temp = GamePlay()
    temp.game_play_completely_random_moves()
for i in range(1000):
    temp = GamePlay()
    temp.naive_game_play() 
for i in range(1000):
    temp = GamePlay()
    temp.game_play_with_random_moves()

    
    
    