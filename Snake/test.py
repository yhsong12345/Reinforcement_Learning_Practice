import torch 
import random
import numpy as np
from collections import deque ### Store memories

from snake_game_rl import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer

from agent import Agent



def test():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        #get move 
        final_move = agent.get_action(state_old)

        #perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # save model
            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score: ', score, 
                  'Record: ', record)
            
            ### Plot
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)






if __name__ == '__name__':
    test()