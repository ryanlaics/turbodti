import random
from deap import base, creator, tools, algorithms
import numpy 

###############################
from pems08_datamodule import PeMS08DataModule
from pems08_lightningmodule import PeMS08LightningModule
import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping   
import torch.nn.functional as F
import torch

device_num = 0
parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="traffic.yaml")
parser.add_argument('--device', default=f'cuda:{device_num}', help='Device for Attack')
############
parser.add_argument(
    "--targetstrategy", type=str, default="block", choices=["mix", "random", "block"]
)
############
parser.add_argument("--nsample", type=int, default=10)


args = parser.parse_args()
print(args)

path = "./CSDI-TurboDTI/config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = False
config["model"]["target_strategy"] = args.targetstrategy
config["diffusion"]["adj_file"] = 'pems-08'
config["seed"] = 1




model = PeMS08LightningModule(config,device=args.device,target_dim=170,seq_len=24)
model.generator.load_state_dict(torch.load("./CSDI-TurboDTI/save/pems08_coeffs_new2.0lr0_20250708_081248/model.pth",map_location="cuda:0"))
model.n_samples = args.nsample
dm = PeMS08DataModule()
trainer = pl.Trainer(
    #accelerator="auto",
    devices = [device_num],
)



NUM_POINTS = 47  
TOTAL_SELECTED = 6  
early_num = 1
mid_num = 4
final_num = TOTAL_SELECTED - mid_num - early_num 


# Define fitness and individual representations for the genetic algorithm
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def individual2list(individual):
    """
    Convert individual representation (bit list) to actual index list,
    flipping order and adjusting by an offset.
    """
    final_list = [49]  # Starting fixed point
    for i in range(len(individual)):
        if individual[i] == 1:
            final_list.append(48 - i)  # Convert index to actual value
    return final_list

def create_individual():
    """
    Randomly create a valid individual with early, mid, and final stage indices selected.
    """
    individual = [0] * (NUM_POINTS + 1)

    # Select fixed early index
    chosen_indices_early = [10]

    # Randomly select mid and final stage indices
    chosen_indices_mid = random.sample(range(11, 39), mid_num)
    chosen_indices_final = random.sample(range(39, NUM_POINTS + 1), final_num)

    # Mark selected positions
    for index in chosen_indices_early:
        individual[index] = 1
    for index in chosen_indices_mid:
        individual[index] = 1
    for index in chosen_indices_final:
        individual[index] = 1

    return creator.Individual(individual)


def evaluate(individual):
    seed_everything(1)
    model.list1 = individual2list(individual)
    print(model.list1)
    trainer.test(model=model,dataloaders=dm.val_dataloader())
    MAE = model.MAE
    return (MAE,)

def crossover(ind1, ind2):
    #tools.cxUniform(ind1, ind2, 0.5)
    tools.cxTwoPoint(ind1,ind2)
    fix_individual(ind1)
    fix_individual(ind2)
    return ind1, ind2


def mutate(individual):

    candidate_indices_mid = list(range(11, 39))
    candidate_indices_final = list(range(39, NUM_POINTS + 1))
    candidate_indices = candidate_indices_mid + candidate_indices_final 

    index_to_flip = random.choice(candidate_indices)
    individual[index_to_flip] = 1 - individual[index_to_flip] 


    fix_individual(individual)
    return individual, 



def fix_individual(individual):
    # Define candidate index ranges for mid and final stage selections
    candidate_indices_mid = list(range(11, 39))
    candidate_indices_final = list(range(39, NUM_POINTS + 1))
    
    # Find already selected indices in mid and final categories
    selected_indices_mid = [idx for idx, value in enumerate(individual) if value == 1 and idx in candidate_indices_mid]
    selected_indices_final = [idx for idx, value in enumerate(individual) if value == 1 and idx in candidate_indices_final]

    # Adjust mid-stage selections to exactly match the required count
    while len(selected_indices_mid) > mid_num:  
        to_remove = random.choice(selected_indices_mid)
        individual[to_remove] = 0
        selected_indices_mid.remove(to_remove)
    
    while len(selected_indices_mid) < mid_num: 
        to_add = random.choice(candidate_indices_mid)
        if to_add not in selected_indices_mid:  
            individual[to_add] = 1
            selected_indices_mid.append(to_add)

    # Adjust final-stage selections to exactly match the required count
    while len(selected_indices_final) > final_num: 
        to_remove = random.choice(selected_indices_final)
        individual[to_remove] = 0
        selected_indices_final.remove(to_remove)

    while len(selected_indices_final) < final_num:  
        to_add = random.choice(candidate_indices_final)
        if to_add not in selected_indices_final: 
            individual[to_add] = 1
            selected_indices_final.append(to_add)





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

    # Hall of Fame to store best individuals
    hof = tools.HallOfFame(5)

    # Statistics for tracking fitness across generations
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # Prepare log file for recording evolution history
    with open("logbook.txt", "w") as log_file:
        log_file.write("Generation\tEvaluations\tAvg Fitness\tMin Fitness\tMax Fitness\n")

    # Run the evolutionary algorithm
    pop, log = algorithms.eaSimple(
        pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=10,
        stats=stats, halloffame=hof, verbose=True
    )

    # Append statistics to log file
    with open("logbook.txt", "a") as log_file: 
        for entry in log:
            log_file.write(f"{entry['gen']}\t{entry['nevals']}\t{entry['avg']:.6f}\t{entry['min']:.6f}\t{entry['max']:.6f}\n")

    return pop, log, hof


if __name__ == "__main__":
    pop, log, hof = main()

    # Print and save the best individuals
    print("Best individuals are:")
    with open("best_overall_5_individuals.txt", "w") as f:
        for individual in hof:
            individual_list = individual2list(individual)
            fitness = individual.fitness.values[0]
            print(f"  Individual: {individual_list} with fitness: {fitness}")
            f.write(f"Individual: {individual_list} with fitness: {fitness:.6f}\n")


    





