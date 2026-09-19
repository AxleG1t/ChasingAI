import pygame
from sys import exit
import random
import numpy as np
import os

pygame.init()
screen = pygame.display.set_mode((800, 600))
background = pygame.Surface((800, 600))
background.fill('cadetblue')
clock = pygame.time.Clock()


# ==========================================================================================
# VIEWMODE = TRUE --> Import best and see the screen
# VIEWMODE = False --> Do not import best fish accross run, don't see anything on the screen
# ==========================================================================================

VIEWMODE = True

# track best ever brain
best_ever = 0

# Roullete list
Roullete = list()

# Player Class
# Every sprite class needs an image and a rect
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((60, 60))
        self.image.fill('Blue')
        self.rect = self.image.get_rect(center = (100, 100))
        self.pos = pygame.Vector2(self.rect.center)
        self.maxmovespeed = 5
        self.brain = SimpleBrain(4, 6, 2)
        self.aliveStatus = True
        self.fitness = 0

    def setBrain(self, newBrain):
        self.brain = newBrain

    def getBrain(self):
        return self.brain

    def getFitness(self):
        return self.fitness

    def isAlive(self):
        return self.aliveStatus

    # Seperate collision detection is not possible
    # because it is done in sequential frames
    # let shark call reset position upon collision
    def suicide(self):
        self.rect.center = (100, 100)
        self.pos = pygame.Vector2(self.rect.center)
        self.aliveStatus = False

    # Function to call
    def Think(self, SharkX, SharkY, SharkRect):
        self.fitness += 1
        if(self.fitness >= 7200):
            self.suicide()
            return

        senses = np.array([
            (SharkX - self.rect.centerx) / 800, 
            (SharkY - self.rect.centery) / 600,     
            self.rect.centerx / 800, 
            self.rect.centery / 600])

        out = self.brain.Process(senses)
        self.pos.x += out[0] * self.maxmovespeed
        self.pos.y += out[1] * self.maxmovespeed

        self.rect.center = self.pos
        self.rect.clamp_ip(screen.get_rect())
        self.pos = pygame.Vector2(self.rect.center)  


    # Needed for Shark, don't change
    def getPosition(self):
        return self.rect.center

    def getRect(self):
        return self.rect


# Enemy Class
class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((60, 60))
        self.image.fill('Red')
        self.rect =  self.image.get_rect(center = (750, 400))
        self.movespeed = 1.7

    def update(self, player_x_position, player_y_position, player):
        # move right
        if(self.rect.centerx < player_x_position and self.rect.right + self.movespeed <= 800):
            self.rect.x += self.movespeed
        elif(self.rect.centerx > player_x_position and self.rect.left - self.movespeed >= 0):
            self.rect.x -= self.movespeed

        if(self.rect.centery < player_y_position and self.rect.top + self.movespeed <= 600):
            self.rect.y += self.movespeed
        elif(self.rect.centery > player_y_position and self.rect.bottom - self.movespeed >= 0):
            self.rect.y -= self.movespeed

        if(self.rect.colliderect(player.getRect())):
            self.rect.center = (700, 500)
            player.suicide()

    def getRect(self):
        return self.rect


def crossover(A, B):
    genomeA = A.get_genome()
    genomeB = B.get_genome()
    point = random.randint(1, len(genomeA) - 1)
    newGenome = np.concatenate([genomeA[:point], genomeB[point:]])
    returnBrain = SimpleBrain(A.InputN, A.HiddenN, A.OutputN)
    returnBrain.set_genome(newGenome)
    return returnBrain
    
def mutate(A, rate):
    genome = A.get_genome()
    for i in range(len(genome)):
        if(random.random() < rate):
            genome[i] += np.random.randn() * 0.1
    A.set_genome(genome)

def breed(roullete, n):
    roullete = sorted(roullete, key = lambda pair: pair[0])
    denominator = n * (n + 1) / 2
    fishes = [fish for _, fish in roullete]
    probability = [(i + 1) / denominator for i in range(n)]


    sum = 0
    cumulative = []
    for p in probability:
        sum += p
        cumulative.append(sum)

    return_population = []
    for _ in range(n - 2):
        parentA, parentB = random.choices(fishes, weights=probability, k = 2)
        childBrain = crossover(parentA.getBrain(), parentB.getBrain())
        mutate(childBrain, 0.1)
        child = Player()
        child.setBrain(childBrain)
        return_population.append(child)

    # elite
    _, bestfi = roullete[-1]
    elite = Player()
    elite.getBrain().set_genome(bestfi.getBrain().get_genome().copy())
    return_population.append(elite)

    sampleBrain = elite.getBrain()

    # mutated elite
    elite_mutated = Player()
    variant_brain = SimpleBrain(sampleBrain.InputN, sampleBrain.HiddenN, sampleBrain.OutputN)
    variant_brain.set_genome(bestfi.getBrain().get_genome().copy())
    mutate(variant_brain, 0.1)
    elite_mutated.setBrain(variant_brain)
    return_population.append(elite_mutated)

    return return_population


class SimpleBrain() :
    def __init__(self, InputNodes, HiddenNodes, OutputNodes) :
        self.InputN = InputNodes
        self.HiddenN = HiddenNodes
        self.OutputN = OutputNodes
        self.layer1 = np.random.randn(self.InputN, self.HiddenN)
        self.layer2 = np.random.randn(self.HiddenN, self.OutputN) 

    def Process(self, senses):
        layer1Output = np.tanh(np.matmul(senses, self.layer1))
        layer2Output = np.tanh(np.matmul(layer1Output, self.layer2))
        return layer2Output

    def get_genome(self):
        return np.concatenate([self.layer1.flatten(), self.layer2.flatten()])
    
    def set_genome(self, flat):
        self.layer1 = flat[:(self.InputN * self.HiddenN)].reshape(self.InputN, self.HiddenN).copy()
        self.layer2 = flat[(self.InputN * self.HiddenN):].reshape(self.HiddenN, self.OutputN).copy()

enemy = pygame.sprite.GroupSingle()
enemy.add(Enemy())

#Initialize the n population
population = list()
for i in range(100):
    population.append(Player())

if(VIEWMODE):
    if(os.path.exists('best_fish.npy')):
        population[0].getBrain().set_genome(np.load('best_fish.npy'))
        print("Best fish succesfully loaded")


# initialize first fish
current_player = 0
player = pygame.sprite.GroupSingle()
player.add(population[current_player])


GenerationCount = 0

while True:

    for events in pygame.event.get():
        if events.type == pygame.QUIT:
            pygame.quit()
            exit()


    # If it is dead
    # Record it's survival time (fitness)
    # stash that in a group
    if(not population[current_player].isAlive()):
        # viewing purposes
        # print("Fish has fitness level : ", population[current_player].getFitness())

        # append it's fitness and the actual object to roullete
        Roullete.append((player.sprite.getFitness(), player.sprite))
        current_player += 1

        if(current_player == 100):
            GenerationCount += 1
            # Generation Recap
            fits = [f for f, _ in Roullete]
            print(f"Gen {GenerationCount}: best = {max(fits)}, avg = {sum(fits)/len(fits)}")

            gen_best_f, gen_best_fi = max(Roullete, key=lambda pair: pair[0])
            if(gen_best_f > best_ever):
                best_ever = gen_best_f
                np.save('best_fish.npy', gen_best_fi.getBrain().get_genome())
                print(f" all new time best: {best_ever} - saved")

            # went through all the population
            # reset
            # breed
            population = breed(Roullete, 100)
            Roullete.clear()
            current_player = 0

        player = pygame.sprite.GroupSingle()
        player.add(population[current_player])

    # Thinking Process
    player.sprite.Think(enemy.sprite.rect.centerx, enemy.sprite.rect.centery, enemy.sprite.getRect())
    enemy.update(*player.sprite.getPosition(), player.sprite)

    # Visualization
    if VIEWMODE:
        screen.blit(background, (0, 0))
        player.draw(screen)
        enemy.draw(screen)
        pygame.display.update()
        clock.tick(60)