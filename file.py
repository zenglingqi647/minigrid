# import os
# import shutil


# root = '/data1/lzengaf/cs285/proj/goto/storage'
# for dir in os.listdir(root):
#     if os.path.exists(os.path.join(root, dir, 'status.pt')):
#         os.remove(os.path.join(root, dir, 'status.pt'))
#         print('removed status.pt in {}'.format(dir))
#     else:
#         print('------- no status.pt in {}'.format(dir))
#         shutil.rmtree(os.path.join(root, dir))


\paragraph{DQN Planner:} The DQN planner operates as a conventional reinforcement learning agent. It outputs discrete actions, each corresponding to a specific skill and goal combination. The DQN is trained to maximize a reward signal based on the performance of the selected skills in achieving the set goals.

\paragraph{LLM Integration:} The LLM is utilized to provide strategic guidance to the DQN planner. Unlike the DQN, which is queried at fixed environment step intervals, the LLM is queried during the training phase of the DQN planner. The result of LLM is validated such that the words are from a pre-defined dictionary and makes it easier to parse and harder to deviate from the desired results.

\paragraph{Query Mechanism:} The planner queries the DQN at fixed environment step intervals for immediate decision-making. In contrast, the LLM is queried after a predetermined number of DQN queries. 
    
\paragraph{Reward Shaping:} A novel aspect of our approach is the reshaping of the planner reward. A reward bonus is given when the DQN's output matches the LLM's recommendation. The reward calculation is as follows: a match in chosen skills yields a reward of 1; similarly, matches in object color and type each contribute a reward of 1. Non-matches receive no reward. The final reward is the average of these individual rewards, promoting alignment with LLM guidance.

\paragraph{Training and Updating:} The DQN is continuously trained and updated based on the reshaped reward signal. This training process incorporates feedback from both the environment and the LLM, ensuring that the DQN's policy evolves to effectively integrate the strategic guidance provided by the LLM.
