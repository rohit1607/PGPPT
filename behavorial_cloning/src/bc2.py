import ilpyt
from ilpyt.agents.imitation_agent import ImitationAgent
from ilpyt.algos.bc import BC

env = ilpyt.envs.build_env(env_id='LunarLander-v2',  num_env=16)
net = ilpyt.nets.choose_net(env)
agent = ImitationAgent(net=net, lr=0.0001)

algo = BC(agent=agent, env=env)
algo.train(num_epochs=10000, expert_demos='demos/LunarLander-v2/demos.pkl')
algo.test(num_episodes=100)
print('done')
