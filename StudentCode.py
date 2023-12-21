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
##     Definition de la fonction objectif à minimiser
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
##     Sélection de l'instance du probleme
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
instance_num=int(input()) # Permet de choisir l'instance au début du programme

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
        self.solution = solution if isinstance(solution,np.ndarray) else np.array(solution) # Enregistre la solution donnée, transforme en np.array si pas déjà fait
        self.cost = cost if cost is not None else fitness(self.solution) # Calcule le fitness si le cost n'est pas donné

    def mutation(self): # Change un chiffre par un chiffre non-présent dans la liste de possibilités entières
        list_possibilities = [x for x in range(m) if x not in self.solution]
        if(list_possibilities != []):
            mutation_value = list_possibilities[random.randint(0,len(list_possibilities)-1)]
            self.solution[random.randint(0,n-1)] = mutation_value
    
    def best_insertion_neighbor(self):  # Effectue des découpes en 1 point pour obtenir le voisinage
        npneighbors = np.array(self)
        for i in range(n-2):
            current_solution = self.solution.copy()
            npneighbors = np.append(npneighbors,Solution(np.concatenate([current_solution[(i+1):], current_solution[:(i+1)]])))

        return min(npneighbors, key=lambda x: x.cost)
    
    def best_inversion_neighbor(self):  # Effectue des inversions d'éléments consécutifs pour obtenir le voisinage
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
    
    def __init__(self, population_size): # Enregistre la taille et créé une nouvelle population aléatoire de taille choisie
        print(f"\nInitialisation de la population...", end="\n")
        self.current_population_size = population_size
        self.current_population = np.array([Solution(np.random.permutation(m)[:n]) for _ in range(population_size)], dtype=object)
        self.history_population = self.current_population.copy()
        print(f"Population Init", end="\n\n")
        for sol in self.current_population:
            print(f"Initial Population | solution : {sol.solution} cost : {sol.cost}", end="\n")

    def reproduce(self,b_time,time_in_secondes):
        print("\nReproduce...")
        for _ in range(size_population // 4 ): # Nombre de reproductions faisant 1/4 de la taille de la population
            print("Création d'un enfant", end="\r")
            child = self.cross_two_points(parent=self.choose_solutions(self.current_population_size)) # Prend 2 parents pour faire un enfant avec la technique des 2 points
            
            if(random.randint(0,99) < mutation_probability):
                child.mutation()
            
            while(time.time() - b_time < time_in_secondes): # Boucle pour avoir le meilleur min entre les voisins
                          
                print("Recherche locale de l'enfant", end="\r")
                best = child.best_insertion_neighbor()
                best = child.best_inversion_neighbor()
                
                self.history_population = np.append(self.history_population, best) # Ajoute le best local dans l'historique
                
                if(best.cost == child.cost): # Si le current est le même que le best alors c'est un min local -> sort de la boucle
                    print(f"Minimum local | solution : {best.solution} cost : {best.cost}", end="\n")
                    self.current_population = np.append(self.current_population, best)
                    break
                else: # Sinon recommence avec le nouveau minimum trouvé
                    child = best 
                
            if(time.time() - b_time >= time_in_secondes):   # Si le temps est dépassé sort de la boucle
                    break 
                  
        self.current_population = self.choose_solutions(self.current_population_size) # Régule la taille de la pop pour conserver sa taille initiale
            
    def cross_two_points(self,parent):
        first_point = np.random.randint(0, n-1) # Choisi le 1er point de coupe
        second_point = np.random.randint(first_point, n-1) # Choisi le 2e point de coupe
        who = np.random.randint(0, 1) # Choisi qui va être le père et la mère

        first_part = parent[who].solution[:first_point] # Prend la 1ère et 3e partie au père
        third_part = parent[who].solution[second_point:n]

        second_part = np.setdiff1d(parent[1-who].solution, np.concatenate([first_part, third_part]))[:n - len(first_part) - len(third_part)]
        # Prend la 2ème partie en retirant les doublons de la 1ère et 3ème partie et coupe tout le trop
        
        return Solution(np.concatenate([first_part, second_part, third_part])) # Retourne une nouvelle solution avec les infos des parties

    def choose_solutions(self,size): # Permet de chosir un nombre de membres dans la population actuelle (plus le coût est petit, plus il a de chance d'être choisi)
        inverted_costs = [1 / s.cost for s in self.current_population]
        total_inverted_cost = sum(inverted_costs)
        probability_cost = [inverted_cost / total_inverted_cost for inverted_cost in inverted_costs]
        return np.random.choice(self.current_population,size,False,probability_cost)

    def show_current_best_solution(self): # Donne la meilleure solution de toute la population
        return min(self.history_population, key=lambda x: x.cost)

##-------------------------------------------------------
##  Function thread show max
##-------------------------------------------------------

def show_best_now():
    global population
    b_time = time.time()
    while(True):
        input()
        print(f"Running time : {round(time.time() - b_time,2)}s")
        current_best_solution = population.show_current_best_solution()
        print(f"Meilleure solution : {current_best_solution.solution} cost {current_best_solution.cost}")

##-------------------------------------------------------
##  Main
##-------------------------------------------------------

size_population = 25
mutation_probability = 25
time_in_secondes = 8*60
b_time = time.time()

population = Population(size_population) # Création de la population

t = threading.Thread(target=show_best_now,daemon=True)
t.start() # Lancement du thread pour voir le meilleur

while(time.time() - b_time < time_in_secondes): # Fin de la boucle après X secondes
    population.reproduce(b_time,time_in_secondes) # Lance une reproduction
    
print("=========================\n Temps alloué écoulé\n========================= ")
current_best_solution = population.show_current_best_solution() # Fin du programme et affiche la meilleure solution
print(f"Meilleure solution : {current_best_solution.solution} cost {current_best_solution.cost}")

fichier = open(f"GROUPE4_Instance_{instance_num}.txt","w")
for s in current_best_solution.solution:
    fichier.write(f"{s} ")
fichier.write(f"{current_best_solution.cost}")
fichier.close()