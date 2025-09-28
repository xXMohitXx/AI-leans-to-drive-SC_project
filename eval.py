import pickle
import neat
from env import CarEnv

def run_best(config_file, genome_file):
    with open(genome_file, "rb") as f:
        genome = pickle.load(f)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    env = CarEnv(render=True)
    obs = env.reset()
    done = False
    while not done:
        action = net.activate(obs)
        obs, reward, done, _ = env.step(action)

if __name__ == "__main__":
    run_best("config_neat.ini", "saved/best_genome.pkl")
