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

# Initialisation de la meilleure solution de l'exécution
def set_best_init(best):
    global best_solution_instance
    if(best_solution_instance == None or best_solution_instance.cost > best.cost):
        best_solution_instance = best

##-------------------------------------------------------
##     Classes
##-------------------------------------------------------
        
# Classe d'une solution et de son coût
class Solution:
    solution = None
    cost = None

    def __init__(self,solution=None,cost=None):
        if(solution == None):
            self.solution = random.sample(range(m),n)           # Génération aléatoire d'une solution valide (pour diversification)
        else:
            self.solution = solution

        if(cost == None):
            self.cost = fitness(self.solution)                  # Calcul du coût de la solution
        else:
            self.cost = cost

    # Clonage d'une solution et son coût
    def clone(self):
        return Solution(solution=self.solution.copy(),cost=self.cost)

# Classe d'une solution résultante d'une reproduction
class ChildSolution(Solution):
    father_solution = None
    mother_solution = None

    def __init__(self,father_solution,mother_solution):
        self.father_solution = father_solution.clone()
        self.mother_solution = mother_solution.clone()

        super().__init__(solution=self.cross_two_points())      # Obtention de la solution enfant par croisement à deux points

    # Croisement à deux points aléatoires
    def cross_two_points(self):
        first_point = random.randint(0,n-1)
        second_point = random.randint(first_point,n-1)

        first_part = (self.father_solution.solution.copy())[0:first_point]
        third_part = (self.father_solution.solution.copy())[second_point:n]

        list_tmp_first = [x for x in self.mother_solution.solution if x not in first_part]
        list_tmp_second = [x for x in list_tmp_first if x not in third_part]

        second_part = list_tmp_second[0 : n - len(first_part) - len(third_part)]

        return (first_part + second_part + third_part)
    
    # Mutation aléatoire d'un élément
    def mutation(self):
        list_possibilities = [x for x in range(m) if x not in self.solution]
        if(list_possibilities != []):
            mutation_value = list_possibilities[random.randint(0,len(list_possibilities)-1)]
            self.solution[random.randint(0,len(self.solution)-1)] = mutation_value

# Classe d'une population (ensemble de solutions) avec sa meilleure solution
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

    # Vérification de meilleure solution ou non
    def is_best_solution(self,new_solution):
        if(self.best_solution == None or new_solution.cost < self.best_solution.cost):
            self.best_solution = new_solution
            set_best_init(self.best_solution)

    # Initialisation de la population par ajout de solutions aléatoires
    def initialize_solutions(self,size):
        for i in range(size):
            print(f"{round((100/size)*i,2)}%",end="\r")
            self.add_solution(Solution())

    def add_solution(self,new_solution):
        self.pop.append(new_solution)
        self.is_best_solution(new_solution)             # Vérification de meilleure solution

    # Choisi un nombre entré de solutions en fonction de probabilités inverse aux coûts
    def choose_solutions(self,size):
        inverted_costs = [1 / s.cost for s in self.pop]
        total_inverted_cost = sum(inverted_costs)
        probability_cost = [inverted_cost / total_inverted_cost for inverted_cost in inverted_costs]
        return list(np.random.choice(self.pop,size,False,probability_cost))                             # Liste de solutions de la population
    
    # Effectue un nombre entré de reproduction entre deux parents
    def reproduce(self,number_of_childs):
        childs = list()

        for i in range(number_of_childs):
            couple = self.choose_solutions(2)                           # Choisi un couple sur base de probabilités

            child = ChildSolution(couple[0],couple[1])                  # Créé un enfant croisé

            # Probabilité que l'enfant mute
            if(random.randint(0,99) < mutation_probability):
                child.mutation()

            # Effectue une recherche locale pour intensifier l'enfant trouvé
            print(f"{i+1}e child :")
            child = best_local_children(child)

            childs.append(child)

        # Ajoute les enfants à la population
        for c in childs:
            self.add_solution(c)

        # Régule la population par probabilités pour ne pas déppasser la taille fixée
        self.pop = self.choose_solutions(size_population)

##-------------------------------------------------------
##     Recherches locales
##-------------------------------------------------------
        
# Recherche locale par permutation
def best_permutation_neighbor(current):
    best = current.clone()
    for i in range(n):
        print(f"{round((100/(n))*i,2)}%",end="\r")
        for j in range(n):
            neighbor_sol = current.solution.copy()
            
            neighbor_sol[i],neighbor_sol[j] = neighbor_sol[j],neighbor_sol[i]               # Échanges entre chaque élément de la solution
            neighbor = Solution(solution=neighbor_sol)
            
            if(neighbor.cost < best.cost):
                best = neighbor.clone()
    return best

# Recherche locale par inversion
def best_inversion_neighbor(current):
    best = current.clone()
    for i in range(n-1):
        print(f"{round((50/(n-1))*i,2)}%",end="\r")
        neighbor_sol = current.solution.copy()

        neighbor_sol[i],neighbor_sol[i+1] = neighbor_sol[i+1],neighbor_sol[i]               # Échanges entre chaque élément consécutif de la solution
        neighbor = Solution(solution=neighbor_sol)
        
        if(neighbor.cost < best.cost):
            best = neighbor.clone()
    
    return best

# Recherche locale par insertion
def best_insertion_neighbor(current):
    best = current.clone()
    for i in range(n-1):
        print(f"{round(50+(50/(n-1))*i,2)}%",end="\r")
        neighbor_sol = current.solution.copy()

        insert = neighbor_sol.pop(i)                            # Déplacement de chaque élément à la fin de la solution (décalage de la solution)
        neighbor_sol.append(insert)
        neighbor = Solution(solution=neighbor_sol)

        if(neighbor.cost < best.cost):
            best = neighbor

    return best

# Recherche locale générale
def best_local_children(child):
    best = child.clone()
    current = child.clone()
    
    # Boucle infinie jusqu'à trouver l'optimum local
    while(True):
        b_time = time.time()                                                                # Double recherche locale chronométrée (inversion -> insertion)
        current = best_inversion_neighbor(current)                                          
        current = best_insertion_neighbor(current)
        print(f"Cette bonne méthode a pris : {round(time.time() - b_time,2)}s")

        if(best.cost > current.cost):                                                   # Vérification d'amélioration locale : Oui -> on continu
            best = current.clone()                                                      #                                      Non -> break (optimum local trouvé)
            set_best_init(best)
            print(f">>current change : {best.solution} cost : {best.cost}")
        else:
            print(f">local best : {best.solution} cost : {best.cost}",end="\n\n")
            break
            
    return best

# Donne l'information sur la meilleure solution de l'exécution à tout moment
def show_best_now():
    global best_solution_instance
    b_time = time.time()
    while(True):
        input()
        print(f"Running time : {round(time.time() - b_time,2)}s")
        if(best_solution_instance != None):
            print(f"The best solution is {best_solution_instance.solution} with a cost {best_solution_instance.cost} \n")
        else: 
            print("N'a pas encore trouvé de meilleur solution")
           
##-------------------------------------------------------
##     Constantes du programme
##-------------------------------------------------------
            
best_solution_instance = None
 
t = threading.Thread(target=show_best_now)
t.start()

# Taille de population en fonction des variables m et n de l'instance (m et n + grand -> population + petite)
size_population = 10
size_population += int(size_population * (size_population/(m+n)))

half_size_population=size_population//2
number_of_reproduction_per_population = 10
mutation_probability = 25

number_of_population_generate = 0

##-------------------------------------------------------
##     Boucle générale du programme
##-------------------------------------------------------

# Boucle infinie du programme
while True: 
    population = Population(size_population)                        # Génération d'une nouvelle population avec affichage
    number_of_population_generate += 1
    print(f"Population pool {number_of_population_generate}\n")
    population.show_actual_population()

    for i in range(number_of_reproduction_per_population):          # Reproduction de la population courante (sélection, croisement, mutation, recherche locale, régulation)
        print(f"{i+1}e reproducing... \n")
        population.reproduce(half_size_population)