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
##  Time Decorator
##-------------------------------------------------------

# decorateur qui permet de voir le temps d'execution d'une fonction
def time_take(func):
    def wrapper(*args, **kwargs):
        b_time = time.time()
        r = func(*args, **kwargs)
        e_time = time.time() - b_time
        print(f"{func.__name__} a pris {round(e_time,2)}s")
        return r
    return wrapper

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
    elif instance_num==10:
        return "Singapore","ghzall",19
    elif instance_num==11:
        return "Singapore","dj",19
    elif instance_num==12:
        return "Cairo","ghz",19
    else:
        print("Choix d'une instance inexistance, instance 1 revoyé  par défaut")
        return "Cairo","ghzall",20


##-------------------------------------------------------
##     Pour choisir une instance: 
##     Modifier instance_num ET RIEN D'AUTRE    
##-------------------------------------------------------
print("\n Choisi l'instance (Entre 1 et 12 inclue) : ",end="")
instance_num=int(input()) # permet de choisir l'instance au début du programme

backend_name,circuit_type,num_qubit=instance_selection(instance_num)
backend,qc,qr=instance_characteristic(backend_name,circuit_type,num_qubit)

n=num_qubit
m=backend.num_qubits

##-------------------------------------------------------
##  Class Solution
##-------------------------------------------------------

class Solution:
    solution = None
    cost = None

    def __init__(self, solution, cost=None):
        self.solution = solution if isinstance(solution,np.ndarray) else np.array(solution) # enregistre la solution donnée, transforme en np.array si pas déjà fait
        self.cost = cost if cost is not None else fitness(self.solution) # calcule le fitness si le cost n'est pas donnée

    def mutation(self): # change un chiffre par un chiffre non-présent dans la liste d'interger
        list_possibilities = [x for x in range(m) if x not in self.solution]
        if(list_possibilities != []):
            mutation_value = list_possibilities[random.randint(0,len(list_possibilities)-1)]
            self.solution[random.randint(0,n-1)] = mutation_value

    #@time_take
    def best_additive_neighbor(self): # Choisi la meilleur solution dans un voisinage [] + i (ex m=20 n=4 => 1 4 5 20 => 2 5 6 0 => 3 6 7 1 => ...)
        npneighbors = np.array([Solution((self.solution + (i+1)) % m) for i in range(m-1)])
        npneighbors = np.append(npneighbors,self)
        return min(npneighbors, key=lambda x: x.cost)
    
    #@time_take
    def best_insertion_neighbor(self):
        npneighbors = np.array(self)
        for i in range(n-2):
            current_solution = self.solution.copy()
            npneighbors = np.append(npneighbors,Solution(np.concatenate([current_solution[(i+1):], current_solution[:(i+1)]])))

        return min(npneighbors, key=lambda x: x.cost)
    
    #@time_take
    def best_inversion_neighbor(self):
        npneighbors = np.array(self)
        for i in range(n-1):
            current_solution = self.solution.copy()
            current_solution[i],current_solution[i+1] = current_solution[i+1],current_solution[i]
            npneighbors = np.append(npneighbors,Solution(current_solution))

        return min(npneighbors, key=lambda x: x.cost)
            
##-------------------------------------------------------
##  Class Population
##-------------------------------------------------------

class Population:
    history_population = None
    current_population = None
    current_population_size = None
    
    def __init__(self, population_size): # enregistre la taille et créer une nouvelle population random de taille choisi
        print(f"\nInit de la population...", end="\n")
        self.current_population_size = population_size
        self.current_population = np.array([Solution(np.random.permutation(m)[:n]) for _ in range(population_size)], dtype=object)
        self.history_population = self.current_population.copy()
        print(f"Population Init", end="\n\n")
        for sol in self.current_population:
            print(f"Initial Population | solution : {sol.solution} cost : {sol.cost}", end="\n")

    @time_take
    def reproduce(self,b_time,time_in_secondes):
        print("\nReproduce...")
        for _ in range(len(self.current_population)// 2 ): # fait 1/4 de la taille de la population de nouveau enfant
            print("Creation d'un enfant", end="\r")
            child = self.cross_two_points(parent=self.choose_solutions(self.current_population_size)) # prend 2 parents pour faire un enfant avec la technique des 2 point
            
            if(random.randint(0,99) < mutation_probability):
                child.mutation()
            
            while(time.time() - b_time < time_in_secondes): # boucle pour avoir le meilleur min entre les voisins
                          
                print("Recherche locale de l'enfant", end="\r")
                if(in_function_n):
                    best = child.best_insertion_neighbor()
                    best = child.best_inversion_neighbor()
                else:
                    best = child.best_additive_neighbor() # fait une recherche local avec l'additive
                
                
                self.history_population = np.append(self.history_population, best) # ajoute le best local dans l'historique
                
                if(best.cost == child.cost): # si le current et le même que le best alors c'est un min local donc sort de la boucle
                    print(f"Minimum locale | solution : {best.solution} cost : {best.cost}", end="\n")
                    self.current_population = np.append(self.current_population, best)
                    break
                else: # sinon recommence avec le nouveau minimum trouvé
                    child = best 
                
            if(time.time() - b_time >= time_in_secondes):   # si le temps est dépasser sors du for
                    break 
                  
        self.current_population = self.choose_solutions(self.current_population_size) # Ajuste la taille de la pop pour reter à sa taille initiale
            
    def cross_two_points(self,parent):
        first_point = np.random.randint(0, n-1) # choisi le 1er point de coupe
        second_point = np.random.randint(first_point, n-1) # choisi le 2e point de coupe
        who = np.random.randint(0, 1) # choisi qui va être le pére et la mére

        first_part = parent[who].solution[:first_point] # prend la 1er et 3e partie au pére
        third_part = parent[who].solution[second_point:n]

        second_part = np.setdiff1d(parent[1-who].solution, np.concatenate([first_part, third_part]))[:n - len(first_part) - len(third_part)]
        # prend la 2e partie en retirant les doublons de la 1er et 3e partie et coupe tout le trop
        
        return Solution(np.concatenate([first_part, second_part, third_part])) # retourne une nouvelle sol avec les infos des parties

    def choose_solutions(self,size): # permet de chosir un nombre de membre dans la population actuelle (plus le coût est petit, plus il a de chance)
        inverted_costs = [1 / s.cost for s in self.current_population]
        total_inverted_cost = sum(inverted_costs)
        probability_cost = [inverted_cost / total_inverted_cost for inverted_cost in inverted_costs]
        return np.random.choice(self.current_population,size,False,probability_cost)

    def show_current_best_solution(self): # donne la meilleur solution de toute la population
        return min(self.history_population, key=lambda x: x.cost)

##-------------------------------------------------------
##  Function thread show max
##-------------------------------------------------------

def show_best_now():
    global population
    b_time = time.time()
    while(True):
        input()
        print(f"running time : {round(time.time() - b_time,2)}s")
        current_best_solution = population.show_current_best_solution()
        print(f"Meilleur solution : {current_best_solution.solution} cost {current_best_solution.cost}")


##-------------------------------------------------------
##  Main
##-------------------------------------------------------

size_population = 10
mutation_probability = 25
time_in_secondes = 3*60
in_function_n = True
b_time = time.time()

population = Population(size_population) # Création de la population

t = threading.Thread(target=show_best_now,daemon=True)
t.start() # lancement du thread pour voir le meilleur

while(time.time() - b_time < time_in_secondes): # fin de la boucle après Xsecondes
    population.reproduce(b_time,time_in_secondes) # lance une reporduction
    
print("=========================\n Temps alloué fini\n========================= ")
current_best_solution = population.show_current_best_solution() #fin du programme et affiche la meilleur sol
print(f"Meilleur solution : {current_best_solution.solution} cost {current_best_solution.cost}") 