# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:08:40 2023

@author: Jérôme
"""

import random
import threading
import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import Layout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import FakeSingaporeV2,FakeWashingtonV2,FakeCairoV2

##-------------------------------------------------------
##     Definition de la fonction objetif à minimiser
##-------------------------------------------------------
def fitness(layout):
    init_layout={qr[i]:layout[i] for i in range(len(layout))}
    init_layout=Layout(init_layout)

    pm = generate_preset_pass_manager(3,backend,initial_layout=init_layout)
    pm.layout.remove(1)
    pm.layout.remove(1)

    QC=pm.run(qc)
    return QC.depth()

##-------------------------------------------------------
##     Selection de l'instance du probleme
##-------------------------------------------------------
def instance_characteristic(backend_name,circuit_type,num_qubit):
    if backend_name == "Singapore":
        backend = FakeSingaporeV2()
    elif backend_name == "Cairo":
        backend = FakeCairoV2()
    else :
        backend = FakeWashingtonV2()
        
    l=f"{circuit_type}_indep_qiskit_{num_qubit}"
    qasmfile=f".\Instances\{l.rstrip()}.qasm"  ###### Il est possible de cette ligne soit problèmatique.
    qc=QuantumCircuit().from_qasm_file(qasmfile)
    qr=qc.qregs[0]
    
    return backend,qc,qr

def instance_selection(instance_num):
    if instance_num==1:
        return "Cairo","ghzall",20
    elif instance_num==2:
        return "Wash","ghzall",20
    elif instance_num==3:
        return "Cairo","ghzall",27
    elif instance_num==4:
        return "Wash","ghzall",27
    elif instance_num==5:
        return "Wash","dj",20
    elif instance_num==6:
        return "Cairo","dj",27
    elif instance_num==7:
        return "Wash","ghz",20
    elif instance_num==8:
        return "Wash","ghz",27    
    elif instance_num==9:
        return "Cairo","qaoa",14
    elif instance_num==11:
        return "Singapore","ghzall",19
    elif instance_num==12:
        return "Singapore","dj",19
    elif instance_num==13:
        return "Cairo","ghz",19
    else:
        print("Choix d'une instance inexistance, instance 1 revoyé  par défaut")
        return "Cairo","ghzall",20


##-------------------------------------------------------
##     Pour choisir une instance: 
##     Modifier instance_num ET RIEN D'AUTRE    
##-------------------------------------------------------
instance_num=2     #### Entre 1 et 9 inclue

backend_name,circuit_type,num_qubit=instance_selection(instance_num)
backend,qc,qr=instance_characteristic(backend_name,circuit_type,num_qubit)

n=num_qubit
m=backend.num_qubits

##-------------------------------------------------------
##     A vous de jouer !  
##-------------------------------------------------------

def set_best_init(best):
    global best_solution_instance
    if(best_solution_instance == None or best_solution_instance.cost > best.cost):
            best_solution_instance = best

class Solution:
    solution = None
    cost = None

    def __init__(self,solution=None,cost=None):
        if(solution == None):
            self.solution = random.sample(range(m),n)
        else:
            self.solution = solution

        if(cost == None):
            self.cost = fitness(self.solution)
        else:
            self.cost = cost

    def clone(self):
        return Solution(solution=self.solution.copy(),cost=self.cost)

class ChildSolution(Solution):
    father_solution = None
    mother_solution = None

    def __init__(self,father_solution,mother_solution):
        self.father_solution = father_solution.clone()
        self.mother_solution = mother_solution.clone()

        super().__init__(solution=self.cross_two_points())

    def cross_two_points(self):
        first_point = random.randint(0,n-1)
        second_point = random.randint(first_point,n-1)

        first_part = (self.father_solution.solution.copy())[0:first_point]
        third_part = (self.father_solution.solution.copy())[second_point:n]

        list_tmp_first = [x for x in self.mother_solution.solution if x not in first_part]
        list_tmp_second = [x for x in list_tmp_first if x not in third_part]

        second_part = list_tmp_second[0 : n - len(first_part) - len(third_part)]

        return (first_part + second_part + third_part)
    
    def mutation(self):
        list_possibilities = [x for x in range(m) if x not in self.solution]
        if(list_possibilities != []):
            mutation_value = list_possibilities[random.randint(0,len(list_possibilities)-1)]
            self.solution[random.randint(0,len(self.solution)-1)] = mutation_value

class Population:
    best_solution = None
    pop = list()

    def __init__(self,size):
        self.pop.clear()
        print("Initialisation de la population :")
        self.initialize_solutions(size)

    def show_best_solution(self):
        print("Best Solution : \n")
        print(f"Solution : {self.best_solution.solution} with cost : {self.best_solution.cost} \n")

    def show_actual_population(self):
        print(f"nombre d'individu : {len(self.pop)}")
        for sol in self.pop: 
            print(f"Solution : {sol.solution} with cost : {sol.cost}")
        print(f"\n")

    def is_best_solution(self,new_solution):
        if(self.best_solution == None or new_solution.cost < self.best_solution.cost):
            self.best_solution = new_solution
            set_best_init(self.best_solution)

    def initialize_solutions(self,size):
        for i in range(size):
            print(f"{round((100/size)*i,2)}%",end="\r")
            self.add_solution(Solution())

    def add_solution(self,new_solution):
        self.pop.append(new_solution)
        self.is_best_solution(new_solution)

    def choose_solutions(self,size):
        inverted_costs = [1 / s.cost for s in self.pop]
        total_inverted_cost = sum(inverted_costs)
        probability_cost = [inverted_cost / total_inverted_cost for inverted_cost in inverted_costs]
        return list(np.random.choice(self.pop,size,False,probability_cost))
    
    def reproduce(self,number_of_childs):
        childs = list()

        for i in range(number_of_childs):
            couple = self.choose_solutions(2)

            child = ChildSolution(couple[0],couple[1])

            if(random.randint(0,99) < mutation_probability):
                child.mutation()

            print(f"{i+1}e child :")
            child = best_local_children(child)

            childs.append(child)

        for c in childs:
            self.add_solution(c)

        self.pop = self.choose_solutions(size_population)

def best_permutation_neighbor(current):
    best = current.clone()
    for i in range(n):
        print(f"{round((100/(n))*i,2)}%",end="\r")
        for j in range(n):
            neighbor_sol = current.solution.copy()
            
            neighbor_sol[i],neighbor_sol[j] = neighbor_sol[j],neighbor_sol[i]
            neighbor = Solution(solution=neighbor_sol)
            
            if(neighbor.cost < best.cost):
                best = neighbor.clone()
    return best

def best_inversion_neighbor(current):
    best = current
    for i in range(n-1):
        print(f"{round((50/(n-1))*i,2)}%",end="\r")
        neighbor_sol = current.solution.copy()

        neighbor_sol[i],neighbor_sol[i+1] = neighbor_sol[i+1],neighbor_sol[i]
        neighbor = Solution(solution=neighbor_sol)
        
        if(neighbor.cost < best.cost):
            best = neighbor.clone()
    
    return best

def best_insertion_neighbor(current):
    best = current
    for i in range(n-1):
        print(f"{round(50+(50/(n-1))*i,2)}%",end="\r")
        neighbor_sol = current.solution.copy()

        insert = neighbor_sol.pop(i)
        neighbor_sol.append(insert)
        neighbor = Solution(solution=neighbor_sol)

        if(neighbor.cost < best.cost):
            best = neighbor
    return best

def best_local_children(child):
    best = child.clone()
    current = child
        
    while(True):
        b_time = time.time()
        current = best_inversion_neighbor(current)
        current = best_insertion_neighbor(current)
        print(f"Cette bonne méthode a pris : {round(time.time() - b_time,2)}s")

        if(best.cost > current.cost):
            best = current.clone()
            set_best_init(best)
            print(f">>current change : {best.solution} cost : {best.cost}")
        else:
            print(f">local best : {best.solution} cost : {best.cost}",end="\n\n")
            break
            
    return best

def show_best_now():
    global best_solution_instance
    b_time = time.time()
    while(True):
        input()
        print(f"running time : {round(time.time() - b_time,2)}s")
        if(best_solution_instance != None):
            print(f"best solution is {best_solution_instance.solution} with a cost {best_solution_instance.cost} \n")
        else: 
            print("n'a pas encore trouvé de meilleur solution")
           
best_solution_instance = None
 
t = threading.Thread(target=show_best_now)
t.start()

size_population = 10
    
size_population += int(size_population * (size_population/(m+n)))

half_size_population=size_population//2
number_of_reproduction_per_population = 10
mutation_probability = 25

number_of_population_generate = 0

while True: 
    population = Population(size_population)
    number_of_population_generate += 1
    print(f"Population pool {number_of_population_generate}\n")
    population.show_actual_population()

    for i in range(number_of_reproduction_per_population):
        print(f"{i+1}e reproducing... \n")
        population.reproduce(half_size_population)