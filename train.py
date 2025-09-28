import neat
import pygame
import random
from env import CarEnv

MAX_STEPS = 1500
POP_SIZE = 20
ELITE_SIZE = 8
GRAPH_HISTORY = 200  # how many generations to show in graph


def continuous_evolution(config_file):
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Init population
    population = []
    for gid in range(POP_SIZE):
        genome = config.genome_type(gid)
        genome.configure_new(config.genome_config)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        car = CarEnv()
        car.reset(offset=gid)   # spawn spread
        population.append({"genome": genome, "net": net, "car": car, "steps": 0})

    generation = 0
    best_genomes = []
    fitness_history = []  # store (best, avg) tuples

    while True:
        generation += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        screen.fill((0, 0, 0))
        if population:
            screen.blit(population[0]["car"].track_surface, (0, 0))

        alive_population = []
        fitnesses = []

        for entry in population:
            car, genome, net = entry["car"], entry["genome"], entry["net"]

            if car.done or entry["steps"] > MAX_STEPS:
                genome.fitness = (genome.fitness or 0) + car.distance
                best_genomes.append(genome)
                best_genomes = sorted(best_genomes, key=lambda g: g.fitness or 0, reverse=True)[:ELITE_SIZE]

                if best_genomes:
                    p1, p2 = random.sample(best_genomes, 2) if len(best_genomes) >= 2 else (best_genomes[0], best_genomes[0])
                    new_genome = config.genome_type(random.randint(100000, 999999))
                    new_genome.configure_crossover(p1, p2, config.genome_config)
                    new_genome.mutate(config.genome_config)
                    new_net = neat.nn.FeedForwardNetwork.create(new_genome, config)
                    new_car = CarEnv()
                    new_car.reset(offset=random.randint(0, POP_SIZE))  # spread again
                    alive_population.append({"genome": new_genome, "net": new_net, "car": new_car, "steps": 0})
            else:
                obs = car._get_obs()
                action = net.activate(obs)
                _, reward, done, _ = car.step(action)
                genome.fitness = (genome.fitness or 0) + reward
                car.draw(screen)
                entry["steps"] += 1
                alive_population.append(entry)
                fitnesses.append(genome.fitness or 0)

        population = alive_population

        # Stats overlay
        if fitnesses:
            best_fit = max(fitnesses)
            avg_fit = sum(fitnesses)/len(fitnesses)
        else:
            best_fit, avg_fit = 0, 0

        fitness_history.append((best_fit, avg_fit))
        if len(fitness_history) > GRAPH_HISTORY:
            fitness_history.pop(0)

        lines = [
            f"Generation: {generation}",
            f"Alive: {len(population)}",
            f"Best fitness: {best_fit:.1f}",
            f"Avg fitness: {avg_fit:.1f}"
        ]
        for i, ln in enumerate(lines):
            surf = font.render(ln, True, (255,255,255))
            screen.blit(surf, (10, 10+i*20))

        

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    continuous_evolution("config_neat.ini")
