import random
import numpy as np
import time

class Professor:
    def __init__(self, id):
        self.id = id
        self.connections = []

    def connect(self, connected_id):
        self.connections.append(connected_id)

class Team:
    def __init__(self, members):
        self.members = members

class Population:
    def __init__(self, teams):
        self.teams = teams
        self.number_of_teams = len(teams)

class Genetic_algorithm:
    def __init__(self, data, number_of_professors,population_size, crossover_operation, mutation_operation, stop_condition):
        self.data = data
        self.number_of_professors = number_of_professors
        self.population_size = population_size
        self.crossover_operation = crossover_operation
        self.mutation_operation = mutation_operation
        self.stop_condition = stop_condition
        self.define_professors()
        self.generate_population()
        self.select_best_team()

    def define_professors(self):
        self.professors = []
        for i in range(self.number_of_professors):
            self.professors.append(Professor(i))
        for connection in self.data:
            self.professors[connection[0]].connect(connection[1])

    def calculate_grant(self, team):
        professor_indices = [i for i, x in enumerate(team.members) if x == 1]
        for i in range(len(professor_indices)):
            for j in range(i + 1, len(professor_indices)):
                if professor_indices[j] in self.professors[professor_indices[i]].connections:
                    team.grant = 0
                    return
        team.grant = sum(team.members) * 1000

    def generate_population(self):
        teams = []
        while len(teams) < self.population_size:
            quantity_of_ones = random.randint(1,5)
            team = list(np.zeros(self.number_of_professors))
            for i in range(quantity_of_ones):
                team[random.randint(0, self.number_of_professors - 1)] = 1
            team = Team(team)
            self.calculate_grant(team)
            if team.grant > 0:
                teams.append(team)
        self.population1 = Population(teams)
        return self.population1

    def select_best_team(self):
        self.best_team = sorted(self.population1.teams, key=lambda x: x.grant, reverse=True)
        self.best_team = self.best_team[0]
        return self.best_team

    def crossover_method(self):
        self.crossed_teams = []
        for team1_index in range(len(self.population1.teams)):
            for team2_index in range(team1_index + 1,len(self.population1.teams)):
                team1 = self.population1.teams[team1_index]
                team2 = self.population1.teams[team2_index]
                new_team = team1.members[:len(team1.members) // 2]
                new_team = new_team + team2.members[len(team2.members) // 2:]
                self.crossed_teams.append(Team(new_team))
        self.population2 = Population(self.crossed_teams)
        return self.population2

    def mutation_method(self, number_of_changes):
        for team in self.population2.teams:
            for rand in random.sample(range(len(team.members)), number_of_changes):
                team.members[rand] = (team.members[rand] + 1) % 2
        return self.population2

    def selection_method(self):
        for team in self.population2.teams:
            self.calculate_grant(team)
        self.best_teams = sorted(self.population2.teams, key = lambda x: x.grant, reverse = True)
        self.best_teams = Population(self.best_teams[:self.population_size-1])
        return self.best_teams

    def stop_condition(self, start, condition_value):
        stop = time.time()
        if stop - start > condition_value:
            return False
        else:
            return True

    def iteration(self):
        self.crossover_method()
        self.mutation_method(1)
        self.selection_method()
        self.population1 = self.best_teams
        self.select_best_team()

start = time.time()
file = open("Arxiv_GR_collab_network.txt", "r")
professors_for_mapping = []
connections = []
for line in file:
    connection = line.split()
    if connection[0] != "#":
        connections.append([int(x) for x in connection])
        if int(connection[0]) not in professors_for_mapping:
            professors_for_mapping.append(int(connection[0]))
        if int(connection[1]) not in professors_for_mapping:
            professors_for_mapping.append(int(connection[1]))
file.close()

for i in range(len(connections)):
    for j in range(len(connections[i])):
        connections[i][j] = professors_for_mapping.index(connections[i][j])

algorithm = Genetic_algorithm(connections, len(professors_for_mapping), 10, self.crossover_method(), mutation_method, stop_condition)
while algorithm.stop_condition(start, 60):
        algorithm.iteration()

team_professors = []
for i in range(len(algorithm.best_team.members)):
    if algorithm.best_team.members[i] == 1:
        team_professors.append(professors_for_mapping[i])

print("Wyszukany zespół:\n", team_professors,"\nOtrzymany grant: ", algorithm.best_team.grant)
