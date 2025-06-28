from training import train_agent
from tqdm import tqdm

if __name__ == "__main__":
    for n_episodes in tqdm([1_000, 10_000, 25_000, 50_000, 100_000], leave=False, desc="Episodes"):
        for rgb in tqdm([True, False], leave=False, desc="RGB"):
            for bin_size in tqdm([32, 64], leave=False, desc="Bin Size"):
                agent_name = f'rgb_{rgb}_bin_{bin_size}_n_{n_episodes}'
                agent = train_agent(agent_name=agent_name, 
                                    n_episodes=n_episodes, 
                                    use_rgb=rgb, 
                                    bin_size=bin_size)