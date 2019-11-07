import numpy as np
dirs = [[-1,0],[0,1],[1,0],[0,-1]]
step_limit = 1000
n = 8
initQ(8)
initPai(8)

def initQ(size):
    Q = np.zeros((size,size,size,size))

def initPai(size):
    pai = np.zeros((size,size,size,size,4))

def move(ax, ay, bx, by, mode='naive'):
    R = -1
    temp_q  = []
    temp_pai = pai[ax][ay][bx][by]
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
    new_ax = ax + dirs[new_dir]
    new_ay = ay + dirs[new_dir]
    if(mode=='naive'):
        if (new_ax < 0 or new_ay < 0 or new_ax > 7 or new_ay > 7):
            return ax, ay, bx, by, new_dir, ax, ay, bx, by, -1;  # back to the same place
        elif (new_ax == bx and new_ay == by):
            new_bx = bx + dirs[new_dir]
            new_by = by + dirs[new_dir]
            if (new_bx < 0 or new_by < 0 or new_bx > 7 or new_by > 7):
                return ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, 10
        else:
            return ax, ay, bx, by, new_dir, new_ax, new_ay, bx, by, -1
    else:
        if (new_ax < 0 or new_ay < 0 or new_ax > 7 or new_ay > 7):
            return ax, ay, bx, by, new_dir, ax, ay, bx, by, -1;  # back to the same place
        elif (new_ax == bx and new_ay == by):
            new_bx = bx + dirs[new_dir]
            new_by = by + dirs[new_dir]
            if (new_bx < 0 or new_by < 0 or new_bx > 7 or new_by > 7):
                return ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, 10
            else:
                if (abs(new_bx - (n - 1) / 2) + abs(new_by - (n - 1) / 2) > abs(bx - (n - 1) / 2) + abs(
                        by - (n - 1) / 2)):
                    return ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, 1
        else:
            return ax, ay, bx, by, new_dir, new_ax, new_ay, bx, by, -1

def MCEpisode(alpha, epsilon, complex_reward = false):
    dir = -1

    bx = randrange(8)
    by = randrange(8)
    bcoord = [bx,by]
    ax = randrange(8)
    ay = randrange(8)
    acoord = [ax,ay]
    while(i<1000):


    #Randomly assign  the bomb and the agent
