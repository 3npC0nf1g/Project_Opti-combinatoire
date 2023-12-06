# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:08:40 2023

@author: Jérôme
"""

import random
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
    elif instance_num==1:
        return "Cairo","ghz",19
    else:
        print("Choix d'une instance inexistance, instance 1 revoyé  par défaut")
        return "Cairo","ghzall",20


##-------------------------------------------------------
##     Pour choisir une instance: 
##     Modifier instance_num ET RIEN D'AUTRE    
##-------------------------------------------------------
instance_num=1     #### Entre 1 et 9 inclue

backend_name,circuit_type,num_qubit=instance_selection(instance_num)
backend,qc,qr=instance_characteristic(backend_name,circuit_type,num_qubit)

n=num_qubit
m=backend.num_qubits

##-------------------------------------------------------
##     A vous de jouer !  
##-------------------------------------------------------

class Solution:
    solution = None
    cost = None

    def __init__(self,solution=None):
        if(solution == None):
            self.solution = random.sample(range(m-1),n)
        else:
            self.solution = solution
        self.cost = fitness(self.solution)

class ChildSolution(Solution):
    def __init__(self,father_solution,mother_solution):
        first_point = random.randint(0,n-1)
        second_point = random.randint(first_point,n-1)

        first_part = (father_solution.solution)[0:first_point]
        third_part = (father_solution.solution)[second_point:n]

        list_temp = [x for x in mother_solution.solution if (x not in first_part) and (x not in third_part)]

        second_part = list_temp[0 : n - len(first_part) - len(third_part)]

        super().__init__(solution=first_part + second_part + third_part)

class Population:
    best_solution = None
    pop = list()

    def __init__(self,size):
        self.pop.clear()
        self.initialize_solutions(size)

    def show_best_solution(self):
        print("Best Solution : \n")
        print(f"Solution : {self.best_solution.solution} with cost : {self.best_solution.cost} \n")

    def show_actual_population(self):
        for sol in self.pop:
            print(f"Solution : {sol.solution} with cost : {sol.cost} \n")

    def is_best_solution(self,new_solution):
        if(self.best_solution == None or new_solution.cost < self.best_solution.cost):
            self.best_solution = new_solution
            return True
        return False

    def initialize_solutions(self,size):
        for i in range(size):
            self.add_solution(Solution())

    def add_solution(self,new_solution):
        self.pop.append(new_solution)
        self.is_best_solution(new_solution)

    def choose_solutions(self,size):
        total_cost = sum([s.cost for s in self.pop])
        probability_cost = [s.cost / total_cost for s in self.pop]
        return list(np.random.choice(self.pop,size,False,probability_cost))
    
    def reproduce(self,number_of_childs):
        childs = list()

        for i in range(number_of_childs):
            couple = self.choose_solutions(2)

            child = ChildSolution(couple[0],couple[1])
            childs.append(child)

        for c in childs:
            self.add_solution(c)

        self.pop = self.choose_solutions(size_population)

size_population=10
half_size_population=size_population//2
number_of_reproduction_per_population = 5
number_of_population_generate = 1

while True: 
    population = Population(size_population)
    print(f"Population pool {number_of_population_generate}\n")
    number_of_population_generate = number_of_population_generate + 1
    population.show_actual_population()
    population.show_best_solution()

    for i in range(number_of_reproduction_per_population):
        print(f"{i+1}e reproducing... \n")
        population.reproduce(half_size_population)
        population.show_best_solution()
    
    print("---------------------------------\n")
    print(f"Actual best solution is {population.best_solution.solution} with a cost {population.best_solution.cost}\n")
    print("---------------------------------\n")