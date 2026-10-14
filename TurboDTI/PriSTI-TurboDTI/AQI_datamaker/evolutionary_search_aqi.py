import sys
sys.path.append("./PriSTI-TurboDTI")

import random
from deap import base, creator, tools, algorithms
import numpy 

###############################
from aqi_datamodule import AQIDataModule
from aqi_lightningmodule import AQILightningModule
import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping   
import torch.nn.functional as F
import torch

device_num = 0
STEPS = 7
parser = argparse.ArgumentParser(description="PriSTI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default=f'cuda:{device_num}', help='Device for Attack')
parser.add_argument(
    "--targetstrategy", type=str, default="hybrid", choices=["hybrid", "random", "historical"]
)
parser.add_argument("--nsample", type=int, default=10)


args = parser.parse_args()
print(args)

path = "./PriSTI-TurboDTI/config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = False
config["model"]["target_strategy"] = args.targetstrategy
config["diffusion"]["adj_file"] = 'AQI36'
config["seed"] = 1




model = AQILightningModule(config,device=args.device,target_dim=36,seq_len=36)
model.generator.load_state_dict(torch.load("./PriSTI-TurboDTI/save/aqi36/model.pth"))
model.n_samples = args.nsample

dm = AQIDataModule()
trainer = pl.Trainer(
    #accelerator="auto",
    devices = [device_num],
)


# Define problem parameters

NUM_POINTS = 98  # Number of optional points, excluding the mandatory points 99 and 0
TOTAL_SELECTED = STEPS - 1  
early_num = 1
mid_num = TOTAL_SELECTED - 1 - 1  # Subtract final_num and early_num
final_num = 1


# Create fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimization problem
creator.create("Individual", list, fitness=creator.FitnessMin)


def individual2list(individual):
    final_list = [99]
    for i in range(len(individual)):
        if individual[i]==1:
            final_list.append(98-i)
    return final_list
# Define how to generate an individual
# Create an individual
def create_individual():
    individual = [0] * (NUM_POINTS)
    chosen_indices_early = [13]
    chosen_indices_mid = random.sample(range(14, 78), mid_num)
    chosen_indices_final = random.sample(range(78, NUM_POINTS), final_num)

    for index in chosen_indices_early:
        individual[index] = 1
    for index in chosen_indices_mid:
        individual[index] = 1
    for index in chosen_indices_final:
        individual[index] = 1
    return creator.Individual(individual)


# Evaluation function
def evaluate(individual):
    seed_everything(1)
    model.list1 = individual2list(individual)
    print(model.list1)
    trainer.test(model=model, dataloaders=dm.val_dataloader())
    MAE = model.MAE
    return (MAE,)


# Breeding-related operations: crossover and mutation
def crossover(ind1, ind2):
    # Perform two-point crossover
    tools.cxTwoPoint(ind1, ind2)
    # Fix individuals to ensure validity
    fix_individual(ind1)
    fix_individual(ind2)
    return ind1, ind2


def mutate(individual):
    # Combine valid candidate points from middle and final segments
    candidate_indices_mid = list(range(14, 78))
    candidate_indices_final = list(range(78, NUM_POINTS))
    candidate_indices = candidate_indices_mid + candidate_indices_final  # Merge candidate points from both segments

    # Randomly select one candidate point to flip
    index_to_flip = random.choice(candidate_indices)
    individual[index_to_flip] = 1 - individual[index_to_flip]  # Flip the selected point

    # Call fix_individual to maintain validity
    fix_individual(individual)
    return individual,  # Return as a tuple




def fix_individual(individual):
    # Define valid candidate index ranges
    candidate_indices_mid = list(range(14, 78))
    candidate_indices_final = list(range(78, NUM_POINTS))
    
    # Get currently selected indices in middle and final segments
    selected_indices_mid = [idx for idx, value in enumerate(individual) if value == 1 and idx in candidate_indices_mid]
    selected_indices_final = [idx for idx, value in enumerate(individual) if value == 1 and idx in candidate_indices_final]

    # Fix the number of selected points in the middle segment
    while len(selected_indices_mid) > mid_num:  # Too many points selected in the middle segment
        to_remove = random.choice(selected_indices_mid)
        individual[to_remove] = 0
        selected_indices_mid.remove(to_remove)
    
    while len(selected_indices_mid) < mid_num:  # Too few points selected in the middle segment
        to_add = random.choice(candidate_indices_mid)
        if to_add not in selected_indices_mid:  # Ensure the point isn't already selected
            individual[to_add] = 1
            selected_indices_mid.append(to_add)

    # Fix the number of selected points in the final segment
    while len(selected_indices_final) > final_num:  # Too many points selected in the final segment
        to_remove = random.choice(selected_indices_final)
        individual[to_remove] = 0
        selected_indices_final.remove(to_remove)

    while len(selected_indices_final) < final_num:  # Too few points selected in the final segment
        to_add = random.choice(candidate_indices_final)
        if to_add not in selected_indices_final:  # Ensure the point isn't already selected
            individual[to_add] = 1
            selected_indices_final.append(to_add)




# Toolbox setup
toolbox = base.Toolbox()
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", crossover)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    # Initialize population
    pop = toolbox.population(n=50)
    
    # Store the top 5 best individuals
    hof = tools.HallOfFame(5)
    
    # Setup statistics tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # Open log file and write header
    with open("logbook.txt", "w") as log_file:
        log_file.write("Generation\tEvaluations\tAvg Fitness\tMin Fitness\tMax Fitness\n")

    # Run genetic algorithm for 10 generations
    pop, log = algorithms.eaSimple(
        pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=10,
        stats=stats, halloffame=hof, verbose=True
    )

    # Save log entries for each generation
    with open("logbook.txt", "a") as log_file:  # Append mode
        for entry in log:
            log_file.write(f"{entry['gen']}\t{entry['nevals']}\t{entry['avg']:.6f}\t{entry['min']:.6f}\t{entry['max']:.6f}\n")

    return pop, log, hof

if __name__ == "__main__":
    # Run main function
    pop, log, hof = main()

    # Print and save the top 5 globally best individuals
    print("Best individuals are:")
    with open("best_overall_5_individuals.txt", "w") as f:
        for individual in hof:
            individual_list = individual2list(individual)
            fitness = individual.fitness.values[0]
            print(f"  Individual: {individual_list} with fitness: {fitness}")
            f.write(f"Individual: {individual_list} with fitness: {fitness:.6f}\n")

    print("STEPS:", STEPS)




