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
parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default=f'cuda:{device_num}', help='Device for Attack')
parser.add_argument(
    "--targetstrategy", type=str, default="hybrid", choices=["hybrid", "random", "historical"]
)
parser.add_argument("--nsample", type=int, default=10)


args = parser.parse_args()
print(args)

path = "./CSDI-TurboDTI/config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = False
config["model"]["target_strategy"] = args.targetstrategy
config["diffusion"]["adj_file"] = 'AQI36'
config["seed"] = 1




model = AQILightningModule(config,device=args.device,target_dim=36,seq_len=36)
model.generator.load_state_dict(torch.load("./CSDI-TurboDTI/save/aqi36_coeffs/model.pth",map_location="cuda"))
model.n_samples = args.nsample
#model.generator.load_state_dict(torch.load("./CSDI-TurboDTI/save/aqi36_coeffs/model.pth",map_location="cuda"))
dm = AQIDataModule()
trainer = pl.Trainer(
    #accelerator="auto",
    devices = [device_num],
)


# Problem parameters definition
NUM_POINTS = 47  # Number of optional points (excluding mandatory points 49 and 0)
TOTAL_SELECTED = 6  # Number of points to select besides point 49
early_num = 1
mid_num = 4
final_num = TOTAL_SELECTED - mid_num - early_num  # Remaining points after subtracting mid_num and early_num
pivot_interval = 3  # (Not currently used)

# Create fitness and individual classes using DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimization problem
creator.create("Individual", list, fitness=creator.FitnessMin)

def individual2list(individual):
    """Convert binary individual representation to list of selected points, always include 49"""
    final_list = [49]
    for i in range(len(individual)):
        if individual[i] == 1:
            final_list.append(48 - i)
    return final_list

# Individual generation function
def create_individual():
    individual = [0] * (NUM_POINTS + 1)
    # Fix early points selection
    chosen_indices_early = [10]
    # Randomly select mid points
    chosen_indices_mid = random.sample(range(11, 39), mid_num)
    # Randomly select final points
    chosen_indices_final = random.sample(range(39, NUM_POINTS + 1), final_num)
    
    # Assign selections
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

# Genetic operations: crossover and mutation
def crossover(ind1, ind2):
    #tools.cxUniform(ind1, ind2, 0.5)
    tools.cxTwoPoint(ind1,ind2)
    fix_individual(ind1)
    fix_individual(ind2)
    return ind1, ind2

def mutate(individual):
    # Merge valid candidate points from middle and final segments
    candidate_indices_mid = list(range(11, 39))
    candidate_indices_final = list(range(39, NUM_POINTS + 1))
    candidate_indices = candidate_indices_mid + candidate_indices_final  # Combine candidate points from both segments

    # Randomly flip one bit in the merged candidate points
    index_to_flip = random.choice(candidate_indices)
    individual[index_to_flip] = 1 - individual[index_to_flip]  # Flip selected bit

    # Call fix_individual to restore legality
    fix_individual(individual)
    return individual,  # Return as a tuple




def fix_individual(individual):
    # Define the valid candidate index ranges
    candidate_indices_mid = list(range(11, 39))
    candidate_indices_final = list(range(39, NUM_POINTS + 1))
    
    # Get currently selected points in middle and final segments
    selected_indices_mid = [idx for idx, value in enumerate(individual) if value == 1 and idx in candidate_indices_mid]
    selected_indices_final = [idx for idx, value in enumerate(individual) if value == 1 and idx in candidate_indices_final]

    # Fix the number of selected points in the middle segment
    while len(selected_indices_mid) > mid_num:  # If more than required points are selected
        to_remove = random.choice(selected_indices_mid)
        individual[to_remove] = 0
        selected_indices_mid.remove(to_remove)
    
    while len(selected_indices_mid) < mid_num:  # If fewer points than required are selected
        to_add = random.choice(candidate_indices_mid)
        if to_add not in selected_indices_mid:  # Ensure the point is not already selected
            individual[to_add] = 1
            selected_indices_mid.append(to_add)

    # Fix the number of selected points in the final segment
    while len(selected_indices_final) > final_num:  # If more than required points are selected
        to_remove = random.choice(selected_indices_final)
        individual[to_remove] = 0
        selected_indices_final.remove(to_remove)

    while len(selected_indices_final) < final_num:  # If fewer points than required are selected
        to_add = random.choice(candidate_indices_final)
        if to_add not in selected_indices_final:  # Ensure the point is not already selected
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
    # Store top 5 individuals
    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # Open log file and write header
    with open("logbook.txt", "w") as log_file:
        log_file.write("Generation\tEvaluations\tAvg Fitness\tMin Fitness\tMax Fitness\n")

    # Run genetic algorithm for 10 generations
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=10, 
                                   stats=stats, halloffame=hof, verbose=True)

    # Append log info for each generation
    with open("logbook.txt", "a") as log_file:
        for entry in log:
            log_file.write(f"{entry['gen']}\t{entry['nevals']}\t{entry['avg']:.6f}\t{entry['min']:.6f}\t{entry['max']:.6f}\n")

    return pop, log, hof



if __name__ == "__main__":
    # Run the main function
    pop, log, hof = main()

    # Print and save the top 5 best individuals overall
    print("Best individuals are:")
    with open("best_overall_5_individuals.txt", "w") as f:
        for individual in hof:
            individual_list = individual2list(individual)
            fitness = individual.fitness.values[0]
            print(f"Individual: {individual_list} with fitness: {fitness}")
            f.write(f"Individual: {individual_list} with fitness: {fitness:.6f}\n")

    





