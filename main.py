import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_adversary_v3


def obs_dict_to_state_vector(observation_dict, agent_order):
    # Extracts observations in the specific order of agents and flattens them
    state = np.concatenate([observation_dict[agent] for agent in agent_order])
    return state


if __name__ == '__main__':
    env = simple_adversary_v3.parallel_env(continuous_actions=True, render_mode="human")
    n_agents = len(env.possible_agents)
    agent_order = env.possible_agents
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space(env.possible_agents[i]).shape[0])
        
    critic_dims = sum(actor_dims)
    
    n_actions = env.action_space(env.possible_agents[0]).shape[0] 
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           fc1=128, fc2=128, alpha=0.1, beta=0.1, scenario='simple')
    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
                                    n_actions, n_agents, batch_size=1024)
    
    PRINT_INTERVAL = 500
    N_GAMES = 50000
    MAX_STEPS = 30
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0
    
    if evaluate:
        maddpg_agents.load_checkpoint()
        
    for i in range(N_GAMES):
        obs, infos = env.reset()
        score = 0
        done = [False] * n_agents
        episode_step = 0
        while not any(done):
            if evaluate:
                print("hi")
                env.render()
                
            actions = maddpg_agents.choose_action(obs)
            observations_, rewards, terminations, truncations, infos = env.step(actions) 
                
    
            actions_list = [actions[agent] for agent in agent_order]           # List of action arrays
            rewards_list = np.array([rewards[agent] for agent in agent_order]) # Shape (n_agents,)
            dones_list = np.array([terminations[agent] or truncations[agent] for agent in agent_order], dtype=bool)
            
            done = truncations.values()
            print(done)
            
            state = obs_dict_to_state_vector(obs, agent_order)
            state_ = obs_dict_to_state_vector(observations_, agent_order)
            print(episode_step)

                
            memory.store_transition(
                raw_obs=[obs[agent] for agent in agent_order],
                state=state,
                action=actions_list,
                reward=rewards_list,
                raw_obs_=[observations_[agent] for agent in agent_order],
                state_=state_,
                done=dones_list
            )
            
            if not evaluate:
                print("HIII")
                maddpg_agents.learn(memory)
            
            obs = observations_
            
            score +=sum(rewards_list)
            total_steps+=1
            episode_step+=1
            
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print(avg_score, best_score)
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', 1, 'average score{:.1f}'.format(avg_score))
            
    
