import random, csv, time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq
import numpy as np
from collections import Counter

__author__ = 'Matt'

class Knapsack:
    def __init__(self, weightlimit):
        self.dataset = []
        self.weightlimit = weightlimit

    def add_to_knapsack(self, Item):
        self.dataset.append(Item)

    def show_knapsack_inventory(self):
        for item in self.dataset:
            print (
            "Nazwa: " + item.name + " Waga: " + str(item.weight) + " Survivalpoints: " + str(item.survivalpoints))

    def generate_chromosome(self, population):
        chromosomes_list = []
        chromosome = []
        for i in range(population):
            for x in range(len(self.dataset)):
                chromosome.append(random.randint(0, 1))
            chromosomes_list.append(chromosome)
            chromosome = []
        return chromosomes_list

    def get_survival_sum(self, chromosome):
        summary = 0
        for i in range(len(chromosome)):
            if chromosome[i] == 1:
                summary += self.dataset[i].survivalpoints
        return summary

    def get_weight_sum(self, chromosome):
        summary = 0
        for i in range(len(chromosome)):
            if chromosome[i] == 1:
                summary += self.dataset[i].weight
        return summary

    def fitness(self, chromosome): #fitness function if ok return value if not 0
        current_solution_survivalpoints = self.get_survival_sum(chromosome)
        current_solution_weight = self.get_weight_sum(chromosome)
        if (current_solution_weight > self.weightlimit):
            return 0
        else:
            return (current_solution_survivalpoints)

    def final_calculation(self, population, iterations, mutationChance, crosstype, selection, elitism): # main function
        is90per = False
        start = time.time()
        chromosomes = self.generate_chromosome(int(population))
        best_fitness = []
        mean_fitness = []
        iterator = 0
        for iterator in range(iterations):
            after_fintess = []
            new_population = []
            for chromosome in chromosomes:
                after_fintess.append(self.fitness(chromosome))
            best_fitness.append((max(after_fintess)))
            if elitism:
                elite = heapq.nlargest(2, after_fintess)
                new_population.append(chromosomes[after_fintess.index(elite[0])]) #elitism
                new_population.append(chromosomes[after_fintess.index(elite[1])])
            mean_fitness.append(np.mean(after_fintess))
            # for i in range(int(round(float(population) / 2))):
            while len(new_population) != int(population):
                parents = []
                for i in range(2):
                    if selection == "roulette":
                        parents.append(chromosomes[self.weighted_choice(self.roulette(after_fintess))])
                    elif selection == "tournament":
                        parents.append(chromosomes[self.tournament(after_fintess)])
                    elif selection == "rank":
                        fitness_sort = sorted(after_fintess)
                        self.weighted_choice(self.ranking_weights(len(fitness_sort)))
                        parents.append(chromosomes[after_fintess.index(fitness_sort
                                                                     [self.weighted_choice
                                                                    (self.ranking_weights(len(after_fintess)))])])
                if crosstype == "onepoint":
                    offsprings = self.crossover(parents)
                elif crosstype == "twopoints":
                    offsprings = self.two_points_crossover(parents)
                if random.random() < float(mutationChance / 100):
                    offsprings = self.chromosome_mutation(offsprings)
                new_population.extend(offsprings)
            chromosomes = new_population
            # if (self.is_it_90_per(chromosomes) == True):
            #     print("90 % of population is similar.")
            #     iterator += 1
            #     break
        iterator += 1
        stop = time.time()
        print("Best score after iterations: ")
        print("Time: " + str(stop - start))
        print self.get_best_res(chromosomes)

        self.draw_fitness_animation(mean_fitness, best_fitness, iterator)
    def is_it_90_per(self, chromosomes): #check if 90 % of population is the same
        ready_chormosomes = []
        chromosomes_to_str = []
        for x in chromosomes:
            chromosomes_to_str.append(map(str, x))
        for i in chromosomes_to_str:
            ready_chormosomes.append(''.join(i))
        if (float(Counter(ready_chormosomes).most_common()[0][1]) / float(len(chromosomes))) >= 0.9:
            return True
        else:
            return False

    def get_best_res(self, chromosomes): #get best result after all iterations (most duplicated)
        ready_chormosomes = []
        chromosomes_to_str = []
        for x in chromosomes:
            chromosomes_to_str.append(map(str, x))
        for i in chromosomes_to_str:
            ready_chormosomes.append(''.join(i))
        return Counter(ready_chormosomes).most_common()[0][0]

    def chromosome_mutation(self, chromosomes): #mutation of chromosomes very low %
        par1 = list(chromosomes[0])
        par2 = list(chromosomes[1])
        mutated = []
        point = random.randint(0, len(self.dataset) - 1)

        if par1[point] == 1:
            par1[point] = 0
        else:
            par1[point] = 1
        mutated.append(par1)
        point = random.randint(0, len(self.dataset) - 1)

        if par2[point] == 1:
            par2[point] = 0
        else:
            par2[point] = 1
        mutated.append(par2)
        print "Chromosomes mutation!"
        return mutated

    def crossover(self, parents): # single point crossover
        sep = random.randint(1, len(self.dataset) - 1)
        offsprings = []
        par1 = list(parents[0])
        par2 = list(parents[1])
        offsprings.append(par1[:sep] + par2[sep:])
        offsprings.append(par2[:sep] + par1[sep:])
        return offsprings

    def two_points_crossover(self, parents): #crossover parents select 2 points parts between two points will be swap
        sep1 = random.randint(1, len(self.dataset) - 2)
        sep2 = random.randint(sep1, len(self.dataset) - 1)
        while (sep1 == sep2 and sep2 > sep1):
            sep2 = random.randint(sep1, len(self.dataset) - 1)
            if (sep2 - sep1 == 1 or sep1 - sep2 == 1):
                sep2 = random.randint(sep1, len(self.dataset) - 1)
        if (sep2 - sep1 == 1 or sep1 - sep2 == 1):
            sep2 = random.randint(sep1, len(self.dataset) - 1)
        offsprings = []
        par1 = list(parents[0])
        par2 = list(parents[1])
        offsprings.append(par1[:sep1] + par2[sep1:sep2] + par1[sep2:])
        offsprings.append(par2[:sep1] + par1[sep1:sep2] + par2[sep2:])
        return offsprings

    def weighted_choice(self, weights): #weighted choice implementation
        total = 0
        winner = 0
        for i, w in enumerate(weights):
            total += w
            if random.random() * total < w:
                winner = i
        return winner

    def roulette(self, chromosomes_list): #making a list of percent values = fit/sum(fitnes) * 100
        percentage = []
        summary = sum(chromosomes_list)
        for i in chromosomes_list:
            percentage.append(float(100 * (float(i) / float(summary))))
        return percentage

    def draw_fitness_animation(self, max, mean , iterations ): #plot drawing
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax2 = fig.add_subplot(1,1,1)
        def animate(i):
            xar = list(i for i in range(iterations))
            yar = max
            zar = mean
            ax1.clear()
            ax1.plot(xar, yar, label="Mean")
            ax2.plot(xar, zar, label="Max")
            handles, labels = ax1.get_legend_handles_labels()
            plt.legend(handles, labels, loc=4)
            plt.title('Plot of fitness function')
            plt.ylabel('Fitness')
            plt.xlabel('Iterations')
            # plt.axis([0,iterations,0,len(max)])
        ani = animation.FuncAnimation(fig, animate)
        plt.show()
    def tournament(self, fitness):
        group = []
        a = random.randint(0, len(fitness) -1 )
        group.append(fitness[a])
        b = random.randint(0, len(fitness) -1 )
        group.append(fitness[b])
        c = random.randint(0, len(fitness) -1 )
        group.append(fitness[c])
        return fitness.index(max(group))

    def ranking_weights(self, population):
        length = population
        weights = []
        for i in range(int(round(length/2))):
            weights.append(float(0.5))
        for i in range(int(round(length/4))):
            weights.append(float(0.25))
        for i in range(int(round(length/8))):
            weights.append(float(0.125))
        for i in range(int(length/16)):
            weights.append(float(0.0625))
        while length != len(weights):
            weights.append(float(0.0001))
        return sorted(weights)

class Item:
    def __init__(self, name, survivalpoints, weight):
        self.name = name
        self.survivalpoints = survivalpoints
        self.weight = weight


def import_csv():
    records = []
    with open('input.csv', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            records.append(row)
    return records

is_run = 1
limit = int(raw_input("Pojemnosc plecaka: "))
knapsack = Knapsack(limit)
while is_run == 1:
    print("1. Dodaj przedmiot\n2. Oblicz najlepszy zestaw\n3. Wyswietl liste przedmiotow\n4. Import z CSV\n5. Zakoncz")
    x = int(raw_input(">> "))
    if x == 1:
        name = raw_input("Nazwa: ")
        survivalpoints = int(raw_input("Survival points: "))
        weight = float((raw_input("Waga: ").replace(",", ".")))
        item = Item(name, survivalpoints, weight)
        knapsack.add_to_knapsack(item)
    elif x == 2:
        population = int(raw_input("Populacja: "))
        iterations = int(raw_input("Ilosc iteracji: "))
        mutation_chance = float(raw_input("Szansa mutacji w %: "))
        crossover = str(raw_input("Krzyzowanie(twopoints / onepoint): "))
        selection = str(raw_input("Metoda selekcji(roulette / tournament / rank ): "))
        elitism = str(raw_input("Elityzm (y / n): "))
        elitism = True if elitism == "y" else False
        knapsack.final_calculation(population, iterations, mutation_chance, crossover, selection, elitism)
    elif x == 3:
        knapsack.show_knapsack_inventory()
    elif x == 4:
        records = import_csv()
        for record in records:
            name = record[0]
            survivalpoints = int(record[1])
            weight = float(record[2].replace(",", "."))
            item = Item(name, survivalpoints, weight)
            knapsack.add_to_knapsack(item)
    elif x == 5:
        is_run = 0
