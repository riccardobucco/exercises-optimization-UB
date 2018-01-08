import numpy as np
import random
import matplotlib.pyplot as plt

N_OF_GAMES_TO_PLAY = 100
P_C = 0.5 # probability of crossover
P_M = 0.001 # probability of mutation
N_GENERATIONS = 1000 # number of generations
N_INDIVIDUALS = 50 # number of individuals

def initialize(n):
    return np.array([np.random.randint(2, size=(70,)) for i in range(0, n)])

def get_move_chromosome(chromosome, opponent_moves = [], my_moves = []):
    if len(opponent_moves) == 0:
        return random.choice(range(0,2))
    if len(opponent_moves) == 1:
        return chromosome[64 + opponent_moves[0]]
    if len(opponent_moves) == 2:
        return chromosome[66 + 2*opponent_moves[0] + opponent_moves[1]]
    else:
        return chromosome[my_moves[0]*32 + opponent_moves[0]*16 + my_moves[1]*8 + opponent_moves[1]*4 + my_moves[2]*2 + opponent_moves[2]]

def get_move_tit_for_tat(chromosome_last_move = -1):
    if chromosome_last_move == -1:
        return 1
    return chromosome_last_move

def get_score_chromosome(chromosome_move, tit_for_tat_move):
    if chromosome_move == 0 and tit_for_tat_move == 0:
        return 1
    if chromosome_move == 0 and tit_for_tat_move == 1:
        return 3
    if chromosome_move == 1 and tit_for_tat_move == 0:
        return 0
    return 6

def f(chromosome):
    # Evaluate the fitness value of a given chromosome/strategy
    score = 0
    tit_for_tat_moves = []
    chromosome_moves = []
    for i in range(0, N_OF_GAMES_TO_PLAY):
        if i == 0:
            tit_for_tat_moves.append(get_move_tit_for_tat())
            chromosome_moves.append(get_move_chromosome(chromosome))
        else:
            tit_for_tat_move = get_move_tit_for_tat(chromosome_moves[-1])
            chromosome_move = get_move_chromosome(chromosome, tit_for_tat_moves, chromosome_moves)
            tit_for_tat_moves.append(tit_for_tat_move)
            chromosome_moves.append(chromosome_move)
        if i > 2:
            tit_for_tat_moves.pop(0)
            chromosome_moves.pop(0)
        score += get_score_chromosome(chromosome_moves[-1], tit_for_tat_moves[-1])
    return N_OF_GAMES_TO_PLAY*6 - score

def evaluate(P_t):
    return np.array([f(P_t[i]) for i in range(0, len(P_t))])

def select(P_t, fitness_values):
    # Implements roulette wheel with slots
    F = sum(fitness_values)
    p_i = fitness_values/float(F)
    q_i = np.array([sum(p_i[0:i]) for i in range(1, len(p_i)+1)])
    P_t1 = np.array([P_t[np.argmax(q_i>=random.uniform(0, 1))] for i in range(0, len(P_t))])
    return P_t1

def crossover(P_t, p_c):
    idx_crossover = np.array([i for i in range(0, len(P_t)) if random.uniform(0, 1) < p_c])
    if len(idx_crossover)%2 == 1:
        idx_crossover = np.delete(idx_crossover, random.choice(range(0, len(idx_crossover))))
    np.random.permutation(idx_crossover)
    for i in range(0, len(idx_crossover), 2):
        idx_1 = idx_crossover[i]
        idx_2 = idx_crossover[i+1]
        m = len(P_t[0])
        pos = random.choice(range(0, m-2))
        tmp = P_t[idx_1][pos+1:].copy()
        P_t[idx_1][pos+1:] = P_t[idx_1+1][pos+1:]
        P_t[idx_1+1][pos+1:] = tmp

def mutation(P_t, p_m):
    for i,chromosome in enumerate(P_t):
        for j,gene in enumerate(chromosome):
            if random.uniform(0, 1) < p_m:
                P_t[i][j] = (gene + 1) % 2

def alter(P_t, p_c, p_m):
    crossover(P_t, p_c)
    mutation(P_t, p_m)

def genetic_algorithm(n, n_generations, p_c, p_m):
    t = 0
    P_t = initialize(n)
    fitness_values = evaluate(P_t)
    average_fitness_value = []
    average_fitness_value.append(sum(fitness_values)/float(len(fitness_values)))
    best_chromosome = P_t[np.argmax(fitness_values)]
    best_chromosome_fitness_value = max(fitness_values)
    for t in range(1, n_generations):
        P_t = select(P_t, fitness_values)
        alter(P_t, p_c, p_m)
        fitness_values = evaluate(P_t)
        average_fitness_value.append(sum(fitness_values)/float(len(fitness_values)))
        if best_chromosome_fitness_value < max(fitness_values):
            best_chromosome_fitness_value = max(fitness_values)
            best_chromosome = P_t[np.argmax(fitness_values)]
    return average_fitness_value, best_chromosome, best_chromosome_fitness_value

average_fitness_value, best_chromosome, best_chromosome_score = genetic_algorithm(N_INDIVIDUALS, N_GENERATIONS, P_C, P_M)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('generation')
ax.set_ylabel('average fitness value')
ax.plot(range(0,N_GENERATIONS), average_fitness_value)
plt.savefig("average-fitness-value.png")

chromosome_repr = "[ "
for gene in best_chromosome:
    chromosome_repr += str(gene) + " "
chromosome_repr += "]"
print("Best chromosome: " + chromosome_repr)
print("Best chromosome fitness value: " + str(best_chromosome_score))