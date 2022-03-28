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

    def copy(self):
        team = Team(self.members.copy())
        team.grant = self.grant
        return team

class Population:
    def __init__(self, teams):
        self.teams = teams
        self.number_of_teams = len(teams)

class Genetic_algorithm:
    def __init__(self, data, number_of_professors,population_size, crossover_operation, crossed_number, mutation_operation, number_of_changes, stop_condition, stop_condition_value):
        self.start = time.time()
        self.data = data
        self.number_of_professors = number_of_professors
        self.population_size = population_size
        self.crossover_operation = crossover_operation
        self.crossed_number = crossed_number
        self.mutation_operation = mutation_operation
        self.number_of_changes = number_of_changes
        self.stop_condition = stop_condition
        self.stop_condition_value = stop_condition_value
        self.number_of_iterations = 0
        self.define_professors()
        self.generate_population()
        self.select_best_team_first_use()
        self.ilosc_iteracji = 0
        if self.stop_condition == 'time_limit':
            while self.time_limit():
                self.iteration()
                self.ilosc_iteracji += 1
        elif self.stop_condition == 'iteration_limit':
            while self.iteration_limit():
                self.iteration()
                self.ilosc_iteracji += 1


    def define_professors(self):
        self.professors = []
        for i in range(self.number_of_professors):
            self.professors.append(Professor(i))
        for connection in self.data:
            self.professors[connection[0]].connect(connection[1])

    def calculate_grant(self, team):
        professor_indices = [i for i, x in enumerate(team.members) if x == 1]
        for i in professor_indices:
            for j in self.professors[i].connections:
                if j in professor_indices:
                    team.grant = 0
                    return
        team.grant = sum(team.members) * 1000

    def generate_population(self):
        teams = []
        while len(teams) < self.population_size:
            team = [-1 for _ in range(self.number_of_professors)]
            first_place = random.randint(0, self.number_of_professors - 1)
            team[first_place] = 1
            for connected_index in self.professors[first_place].connections:
                team[connected_index] = 0
            index = first_place + 1
            while index != first_place:
                if team[index] == -1:
                    team[index] = 1
                    for connected_index in self.professors[index].connections:
                        team[connected_index] = 0
                index = (index + 1) % self.number_of_professors
            team = Team(team)
            self.calculate_grant(team)
            teams.append(team)
        self.population1 = Population(teams)
        return self.population1

    def select_best_team_first_use(self):
        self.sorted_teams = sorted(self.population1.teams, key=lambda x: x.grant, reverse=True)
        self.best_team = self.sorted_teams[0].copy()
        return self.best_team

    def select_best_team(self):
        self.sorted_teams = sorted(self.population1.teams, key=lambda x: x.grant, reverse=True)
        if self.best_team.grant < self.sorted_teams[0].grant:
            self.best_team = self.sorted_teams[0].copy()
        return self.best_team

    def half_of_lists(self):
        self.crossed_teams = []
        for team1_index in range(self.crossed_number):
            for team2_index in range(team1_index + 1, self.crossed_number):
                team1 = self.sorted_teams[team1_index]
                team2 = self.sorted_teams[team2_index]
                new_team = team1.members[:len(team1.members) // 2]
                new_team = new_team + team2.members[len(team2.members) // 2:]
                self.crossed_teams.append(Team(new_team))
                self.crossed_teams.append(team1)
                self.crossed_teams.append(team2)
        self.population2 = Population(self.crossed_teams)
        return self.population2

    def random_cut_place(self):
        self.crossed_teams = []
        for team1_index in range(self.crossed_number):
            for team2_index in range(team1_index + 1, self.crossed_number):
                team1 = self.sorted_teams[team1_index]
                team2 = self.sorted_teams[team2_index]
                random_cut = random.randint(0, len(team1.members) - 1)
                new_team = team1.members[:random_cut]
                new_team = new_team + team2.members[random_cut:]
                self.crossed_teams.append(Team(new_team))
                self.crossed_teams.append(team1)
                self.crossed_teams.append(team2)
        self.population2 = Population(self.crossed_teams)
        return self.population2

    def change_individual(self):
        for team in self.population2.teams:
            for rand in random.sample(range(len(team.members)), self.number_of_changes):
                team.members[rand] = (team.members[rand] + 1) % 2
        return self.population2

    def change_n_places(self):
        for i in range(self.number_of_changes):
            random_team = self.population2.teams[random.randint(0, len(self.population2.teams) - 1)]
            random_place = random.randint(0, len(random_team.members) - 1)
            random_team.members[random_place] = (random_team.members[random_place] + 1) % 2
        return self.population2

    def selection_method(self):
        for team in self.population2.teams:
            self.calculate_grant(team)
        self.best_teams = sorted(self.population2.teams, key = lambda x: x.grant, reverse = True)
        self.best_teams = Population(self.best_teams[:self.population_size-1])
        return self.best_teams

    def time_limit(self):
        stop = time.time()
        if stop - self.start < self.stop_condition_value:
            return True
        else:
            return False

    def iteration_limit(self):
        if self.number_of_iterations < self.stop_condition_value:
            return True
        else:
            return False

    def iteration(self):
        if self.crossover_operation == 'half_of_lists':
            self.half_of_lists()
        elif self.crossover_operation == 'random_cut_place':
            self.random_cut_place()
        if self.mutation_operation == 'change_individual':
            self.change_individual()
        elif self.mutation_operation == 'change_n_places':
            self.change_n_places()
        self.selection_method()
        self.population1 = self.best_teams
        self.select_best_team()
        self.number_of_iterations += 1

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

grants = []
for k in range(5):
    algorithm = Genetic_algorithm(connections, len(professors_for_mapping), 100, 'half_of_lists', 20, 'change_n_places', 200, 'time_limit', 1000)

    team_professors = []
    for i in range(len(algorithm.best_team.members)):
        if algorithm.best_team.members[i] == 1:
            team_professors.append(professors_for_mapping[i])

    print("Wyszukany zespół:\n", team_professors,"\nOtrzymany grant: ", algorithm.best_team.grant)
    print('Liczba iteracji: ', algorithm.number_of_iterations)

grants.append(algorithm.best_team.grant)
print(sum(grants) / len(grants))

#Tomasz Gwiazda - "Algorytmy genetyczne"