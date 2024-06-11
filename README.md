# OpenAI Gym Risk Environment
## Formerly: GaTech CS7643 DL Group Project 

![image](https://github.gatech.edu/storage/user/51858/files/5b3f41d7-a1e0-4910-b725-0b9973f7034e)

## About          
  - This is Risk represented as a bi-directional graph. It also has 2 basic bots implemented (neutral, low skilled) and an interface for configuring and training a DDQN
  - There are several Jupyer notebooks to show bot and environment functionality, as well as simulation.py which randomly runs a game until completion
  - Because the full (classic map) Risk board is computationally demanding and difficult to test on, it includes 3 smaller boards \[small, medium, large]
  - We applied heuristics to make the environment discrete. All actions result in 100% troop commitment. Also, cards are traded automatically at 5 cards and for the highest amount, and so trading is not an action in this environment. Edits can be made to have a continuous action space and to use trading cards as an action
  - The action space of this environment is to select the territory index where an action starts and then its destination. These are independent actions across two observations. For example, to attack, an action first is made to select the attacking territory. Then, the Agent observes the board state and the territory it selected, and can then choose a territory index as a 2nd action. There is also a skip/end-phase action
  - The observation space includes normalized troop counts across all territories, a binary toggle of the current phase, and the previously selected action. Troops that are the Agent's are positive, the opponent's are negative. This is suitable for 1vs1 but not multi-agent
<img width="605" alt="Screenshot_2023-08-01_at_4 27 59_PM" src="https://github.gatech.edu/storage/user/51858/files/8c062e17-e3ad-4aa2-9270-8d3f4e9d4ce1">

  - A critical aspect is handling of illegal moves. It is up to the user to decide how to handle illegal moves. It is a core problem because the majority of moves are illegal from any given state, especially in the attack or fortify phase if the Agent has already selected a "FROM" territory. The current default is to not change the state and apply a large penalty but we have found this highly problematic. There is research on this topic: [Learn What Not to Learn: Action Elimination with
Deep Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2018/file/645098b086d2f9e1e0e939c27f9f2d6f-Paper.pdf)

  - Last updated 08/02/23      
  
## Areas of improvement       

- Code cleanup & optimization
  - i.e reduce use of nested loops, potentially add hash tables everywhere         
  - reduce redundant computations across similar functions, likely by tracking more data and more precisely updating that data       
  - make variable naming more consistent                 
  - use of dictionaries instead of key lookups everywhere          
- Allow graph pathing in env handling of Agent attack (or handle this on the Agent side) 
  - Currently, only direct neighbors of a selected territory are considered legal in env, forcing Agent to be very literal   
    - Although this may be considered a macro/option level action

## Possible areas of research     
- MARL
  - 3+ player Risk against humans is mixed competitive-cooperative asynchronous multi-Agent with varying reward functions 
  - Risk against humans, if fully developed, requires real-time communication
- State representation
  - Why do differences in the board size explode complexity for a NN but not a human? What heuristics can be used?
    - i.e discretize observation of troop counts in a meaningful way
    - Hierarchical representation
- Self-play     
- More advanced bots     
- Larger boards        
- Credit assignment     
- Reward functions

  ## Code contributors
- Luke Pratt
- Julia Shuieh
- Robert Kiesler
- Ashish Panchal 

  Transferred from GaTech's private repository to this public repository on June 10th, 2024

