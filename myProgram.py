import numpy as np
from random import randrange

def initQ(size):
    return np.zeros((size,size,size,size))

def initPai(size):
    return np.zeros((size,size,size,size,4))


dirs = [[-1, 0], [0, 1], [1, 0], [0, -1]]
step_limit = 1000
n = 8
Q = initQ(n)
pai = initPai(n)

def move(ax, ay, bx, by, pai, mode='naive'):
    temp_q  = []
    temp_pai = pai[ax][ay][bx][by]
    new_ax = ax
    new_ay = ay
    new_bx = bx
    new_by = by
    if(temp_pai[0]==0):
        new_dir = randrange(4)
    else:
        rand = randrange(10)
        max_prob = max(temp_pai)
        if(rand<1):
            sub_rand = randrange(4)
            while(temp_pai[sub_rand]==max_prob):
                sub_rand = randrange(4)
            new_dir = sub_rand
        else:
            new_dir = temp_pai.index(max_prob)
    new_ax = ax + dirs[new_dir][0]
    new_ay = ay + dirs[new_dir][1]
    if(mode=='naive'):
        if (new_ax < 0 or new_ay < 0 or new_ax > 7 or new_ay > 7):
            new_ax = ax
            new_ay = ay # back to the same place
        elif (new_ax == bx and new_ay == by):
            new_bx = bx + dirs[new_dir][0]
            new_by = by + dirs[new_dir][1]
        return ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, -1
    else:
        if (new_ax < 0 or new_ay < 0 or new_ax > 7 or new_ay > 7):
            new_ax = ax
            new_ay = ay # back to the same place
            return ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, -1;  # back to the same place
        elif (new_ax == bx and new_ay == by):
            new_bx = bx + dirs[new_dir][0]
            new_by = by + dirs[new_dir][1]
            if (new_bx < 0 or new_by < 0 or new_bx > 7 or new_by > 7):
                return ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, 10
            else:
                if (abs(new_bx - (n - 1) / 2) + abs(new_by - (n - 1) / 2) > abs(bx - (n - 1) / 2) + abs(
                        by - (n - 1) / 2)):
                    return ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, 1
                else:
                    return ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, -1
        else:
            return ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, -1
def MEUpdate(SARS):
    return 1

def QLearning(alpha, epsilon, mode):
    return 1

def MCEpisode(alpha, epsilon, pai,mode = 'naive'):
    SARs = []
    ax = 0
    bx = 0
    ay = 0
    by = 0
    while(ax==bx and ay==by):
        bx = randrange(8)
        by = randrange(8)
        ax = randrange(8)
        ay = randrange(8)
    i = 0
    while(i<1000):
        ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, reward = move(ax,ay,bx,by,pai,mode)
        SARs.append([ax,ay,bx,by,new_dir,reward])
        if (new_bx < 0 or new_by < 0 or new_bx > 7 or new_by > 7):
            break
        else:
            ax = new_ax
            ay = new_ay
            bx = new_bx
            by = new_by
            i = i+1
    if(i!=1000):
        return MCUpdate(SARs)
    else:
        return 0

def learn( alpha, epsilon, pai,mode='naive', method = 'MC'):
    reward = 0
    i = 1
    while(i<10000):
        if(method == 'MC'):
            reward = MCEpisode(alpha,epsilon,pai,mode)
        else:
            reward = QLearning(alpha, epsilon, mode)#TODO change it to TD
        i = i + 1
    return reward


def main():
    print(learn(0.1,0.1,pai,'complex','MC'))

if __name__ == '__main__':
    main()

print("__name__ value: ", __name__)