from env import RiskEnvFlat, DQNAgent

from pathlib import Path
import matplotlib.pyplot as plt
import gym
import numpy as np
import pandas as pd
import random

import time
import yaml
import os
import statistics
import shutil

import cProfile
import wandb
import psutil
import GPUtil


def evaluate(agent, env, max_actions=500, num_eval=1, show_board=False):
    for i in range(num_eval):
        agent.epsilon = 0
        state, _ = env.reset()
        actions = 0
        total_reward = 0
        illegal_moves = 0
        terminated = False
        np.set_printoptions(precision=2)
        while not terminated and actions < max_actions:
            action = agent.act(state)

            if show_board:
                if env.phase in (0, 1, 3):
                    env.show_board()
                print(state, action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            actions += 1
            total_reward += reward
            if reward == env.invalid_move_penalty:
                illegal_moves += 1
        opponents_alive = 0
        for p in env.players[1:]:
            if p.territory_count > 0:
                opponents_alive += 1
        result = None
        if opponents_alive == 0: result  = "win"
        elif env.agent.territory_count == 0: result = "lose"
        else: result = "draw"
        print(f"Evaluation Reward: {(total_reward):.3f}, Illegal Move Ratio: {illegal_moves/actions:.3f} Result: {result}")

def create_output_graphs(output_dir, num_episodes, num_players, bot_types, size, output_lists):
    sizes = ["Small", "Medium", "Large"]
    bots = ', '.join(set(bot_types))
    output_path = Path(f'{output_dir}/outputs')
    output_path.mkdir(parents=True, exist_ok=True)
    names = ["Average Reward", "Cumulative Reward", "Loss", "Illegal Move Ratio", "Number of Actions", "Number of Turns", "Episode Time", "Action Selection STD", "Skip Action Ratio", "Percentage Map Owned on End"]

    # Determine the layout of the subplots
    num_graphs = len(output_lists)
    num_cols = 5
    num_rows = (num_graphs + num_cols - 1) // num_cols  # Ensure enough rows for all graphs
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(36, 6 * num_rows))  # Adjust size accordingly
    axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate over it

    for ax, output, name in zip(axes, output_lists, names):
        # Calculate the rolling average with a window of 30
        data_series = pd.Series(output)
        rolling_avg = data_series.rolling(window=30).mean()

        # Plot the original data and rolling average on each subplot
        ax.plot(range(1, num_episodes+1), output, label='Original')
        ax.plot(range(1, num_episodes+1), rolling_avg, label='Rolling Average', color='orange')
        ax.set_xlabel("Episode")
        ax.set_ylabel(name)
        ax.set_title(f"{name} - {sizes[size]} Board; {num_players} Players; {bots}")
        ax.legend()

    # Turn off any extra empty subplots
    for i in range(num_graphs, num_rows * num_cols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    # Save the entire figure with all subplots
    plt.savefig(output_path / f"combined_{sizes[size].lower()}_{num_players}_{'_'.join(set(bot_types)).lower()}.png")
    plt.close(fig)

    # in addition to the large graph which is easier viewing for monitoring,
    # create all the smaller graphs which can be better for reporting
    def create_graph(output, name):
        data_series = pd.Series(output)
        rolling_avg = data_series.rolling(window=30).mean()

        # plt.plot(range(1, num_episodes+1), output)
        plt.plot(range(1, num_episodes+1), output, label='Original')
        plt.plot(range(1, num_episodes+1), rolling_avg, label='Rolling Average', color='orange')
        plt.xlabel("Episode")
        plt.ylabel(name)
        plt.title(f"{name} - {sizes[size]} Board; {num_players} Players; {bots} Bots")
        plt.savefig(output_path / f"{sizes[size].lower()}_{num_players}_{'_'.join(set(bot_types)).lower()}_{name.lower().replace(' ', '_')}.png")
        plt.clf()

    for out, name in zip(output_lists, names):
        create_graph(out, name)

def create_bar_graphs(output_dir, num_episodes, num_players, bot_types, size, output_lists):
    # use for composition analysis
    # such as, STD of moves on a phasic basis, or illegal move ratio on phasic basis
    return

def log_system_metrics():
    # Get GPU information
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]  # Assuming a single GPU setup
        wandb.log({
            "CPU Usage (%)": psutil.cpu_percent(),
            "Memory Usage (%)": psutil.virtual_memory().percent,
            "GPU Usage (%)": gpu.load * 100,
            "GPU Memory Usage (%)": gpu.memoryUtil * 100,
            "GPU Temperature (C)": gpu.temperature
        })
    else:
        wandb.log({
            "CPU Usage (%)": psutil.cpu_percent(),
            "Memory Usage (%)": psutil.virtual_memory().percent,
            "GPU Usage (%)": None,
            "GPU Memory Usage (%)": None,
            "GPU Temperature (C)": None
        })

def main(env_name, num_episodes=2000, save_interval=100, load_model=False):
    
    with open('training_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    os.system('clear') # assuming windows, clear debug output

    wandb.init(project="risk-training", entity="xcyang")

    num_episodes = config["num_episodes"]
    save_interval = config["save_interval"]

    num_players = config["num_players"]
    bot_types = config["bot_types"]
    size = config["size"]
    env = gym.make(env_name, env_config={"num_players": num_players, "size": size, "bot_types": bot_types})

    decay_type = config["decay_type"]
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    output_dir = "experiment_results/" + config["experiment_name"]
    checkpoint_path = Path(f'{output_dir}/checkpoints')
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    shutil.copyfile('training_config.yaml', f"{output_dir}/config.yaml")

    #start making 10% random moves, decay to .05%, with 10% soft update
    #note that when the majority of moves are illegal, this is a high epsilon
    #an idea is to have a legality weighted epsilon but aint nobody got time for that
    #
    #execution slows down dramatically with lower epsilons, I think its because the Agent
    #is dumb and just makes illegal moves over and over
    agent = DQNAgent(
        state_size,
        action_size,
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        epsilon=config["epsilon_max"],
        epsilon_min=config["epsilon_min"],
        epsilon_decay=config["epsilon_decay"] if decay_type == "geo" else None,
        learning_rate=config["learning_rate"],
        tau=config["tau"],
        num_layers=config["num_layers"],
        hidden_dim_max=config["hidden_dim_max"],
        hidden_dim_min=config["hidden_dim_min"],
        target_update_freq=config["target_update_freq"]
    )

    if load_model:
        agent.load(checkpoint_path / 'dqn_model_best.pth')
    print("I am using device", agent.device)

    optimize_ratio = config["optimize_ratio"] #how much are we re-using old data
    render = False
    render_frequency = 200000 #out of
    best_reward = -np.inf
    avg_rewards, cum_rewards, losses, illegal_move_ratios, num_actions, num_turns, ep_time, action_stds, skip_action_ratios, map_ownership_ratios, phasic_action_stds = tuple([] for i in range(11))

    num_oscillations = config["num_oscillations"]  # number of full oscillations over num_episodes
    eps = eps_start = config["epsilon_max"]
    eps_end = config["epsilon_min"]
    max_actions = config["max_actions"]

    if not config["eval_only"]:
        # print(env.bot_types)
        # env.show_board()
        for episode in range(num_episodes):

            log_system_metrics()

            if decay_type == "osc":
                amplitude = eps_start - (episode / num_episodes) * (eps_start - eps_end)
                phase = episode / num_episodes * num_oscillations * 2 * np.pi
                eps = eps_end + amplitude * np.sin(phase)
                eps = abs(eps)
                agent.epsilon = eps
            if decay_type == "lin":
                eps -= (eps_start - eps_end) / num_episodes
                agent.epsilon = eps
            state, _ = env.reset()
            terminated = False
            total_reward = 0
            actions = 0
            illegal_moves = 0
            loss = 0
            loss_count = 0
            skip_actions = 0
            action_counts = {i: 0 for i in range(len(env.territories) + 1)}
            #phase_action_counts = {i: {j: 0 for j in range(len(env.territories) + 1)} for i in range(5) # for composition analysis
            phase_counts = {i: 0 for i in range(5)}
            action_hist = []
            skip_action = len(env.territories)

            state_hist = [state]
            start = time.time()

            optim_time = 0


            while not terminated:
                action = agent.act(state)
                if action == skip_action:
                    skip_actions += 1
                action_hist.append(action)
                #phase_action_counts[env.phase][action] += 1
                action_counts[action] += 1
                phase_counts[env.phase] += 1
                next_state, reward, terminated, truncated, _ = env.step(action)
                state_hist.append(next_state)
                if reward == env.invalid_move_penalty:
                    illegal_moves += 1
                agent.remember(state, action, reward, next_state, terminated, truncated)
                state = next_state
                total_reward += reward
                actions += 1 #why the flipping heck is this so high??
                if render and random.randint(1, render_frequency) == render_frequency:
                    env.show_board() #should prolly just save this or something, blocks are kinda annoying
                start_optim = time.time()
                if actions % agent.batch_size == 0:
                    for i in range(optimize_ratio):
                        loss += agent.optimize_network()
                        loss_count += 1
                optim_time += time.time() - start_optim


                if actions == max_actions: #but why is this happening in the first place
                    break
            # Print to check if there's anything fishy
            # if illegal_moves == 0 and actions < 10:
            #     print(action_hist)
            #     for s in state_hist:
            #         print(s.tolist())

            #if optimize_ratio > 1 and (episode + 1) % ((num_episodes/optimize_ratio)) == 0:
            #    optimize_ratio -= 1 # recycle data less as the Agent advances
            if (episode + 1) % save_interval == 0 or episode == 1:
                agent.save(checkpoint_path / f'dqn_model_{episode}.pth')

            if total_reward > best_reward and episode > save_interval:
                best_reward = total_reward
                agent.save(checkpoint_path / f'dqn_model_best_{episode}.pth')  # filename includes episode number
                agent.save(checkpoint_path / f'dqn_model_best.pth')

            ep_time.append(time.time() - start)
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {(total_reward/actions):.4f}, Loss: {(loss/max(loss_count, 1)):.6f}, Epsilon: {agent.epsilon:.4f}, Illegal Move Ratio: {illegal_moves/actions:.3f} Actions: {actions}")
            print(f"Turns passed {env.turns_passed}, Agent territories remaining {env.agent.territory_count}, Opponent territories remaining {len(env.territories) - env.agent.territory_count}")
            action_std = np.std(list(action_counts.values()))
            action_values = list(action_counts.values())
            max_action_count = max(action_counts, key=action_counts.get)
            print(f"Positive: {env.agent.positive_reward_only:.2f}, Negative: {env.agent.negative_reward_only:.2f}, Cumulative {env.agent.cumulative_reward:.2f}")
            print(f"Min Action Count: {min(action_values)}, Median Action Count: {statistics.median(action_values)}, Action {max_action_count}, has the maximum count: {action_counts[max_action_count]}")
            print(f"Placement Phase: {phase_counts[0]}, Attack Source: {phase_counts[1]}, Attack Target {phase_counts[2]}, Fortify From: {phase_counts[3]}, Fortify To: {phase_counts[4]}\n")
            print(f"Total Time: {time.time() - start}, Optimization Time: {optim_time}\n")


            avg_rewards.append(total_reward / actions)
            cum_rewards.append(env.agent.cumulative_reward)
            losses.append(loss/max(loss_count, 1))
            illegal_move_ratios.append(illegal_moves/actions)
            num_actions.append(actions)
            num_turns.append(env.turns_passed + 1)
            action_stds.append(action_std)
            skip_action_ratios.append(skip_actions/actions)
            map_ownership_ratios.append(env.agent.territory_count / len(env.territories))
            #phasic_action_stds.append([np.std(actions) for actions in phase_action_counts_list])

            assert(env.agent.territory_count <= len(env.territories))

        out = (avg_rewards, cum_rewards, losses, illegal_move_ratios, num_actions, num_turns, ep_time, action_stds, skip_action_ratios, map_ownership_ratios)
        create_output_graphs(output_dir, num_episodes, num_players, bot_types, size, out)

    render_final_results = config["render_final_results"]
    if render_final_results:
        agent.load(checkpoint_path / config["load_checkpoint"])
        evaluate(agent, env, config["max_actions"], config["num_eval"], config["show_board"])
    env.close()

if __name__ == "__main__":
    gym.envs.register(id='RiskEnvFlat-v0', entry_point=RiskEnvFlat)
    #main('RiskEnvFlat-v0')
    cProfile.run('main("RiskEnvFlat-v0")', 'profile_stats.prof')
