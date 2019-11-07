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

def move(ax, ay, bx, by):
    R = -1
    temp_q  = []
    temp_pai = pai[ax][ay][bx][by]
    for i in range(0,len(dirs)):
        new_ax = ax+dirs[i]
        new_ay = by+dirs[i]
        if (new_ax < 0 or new_ay < 0 or new_ax > 7 or new_ay > 7):
            temp_q.append(0)
        else:
            if(new_x==bx and new_y == by):
                new_bx = bx + dirs[i]
                new_by = by + dirs[i]

    policy = pai[ax][ay][bx][by]
    prob = -1
    for dir in range(0,len(policy)):
        if(policy[dir]>0.25):
            prob = policy[dir]



def MCEpisode(alpha, epsilon, complex_reward = false):
    dir = -1

    bx = randrange(8)
    by = randrange(8)
    bcoord = [bx,by]
    ax = randrange(8)
    ay = randrange(8)
    acoord = [ax,ay]
    while(i<1000):
        i = i+1
        action = randrange(4)
        ax = ax + dirs[action][0]
        ay = ay + dirs[action][1]
        if(ax<0 or ay<0 or ax>7 or ay >7):
            ax = ax - dirs[action][0]
            ay = ay - dirs[action][1]
            reward = -1
            #record ths s, a pair
            continue
        if(ax==bx and ay==by):
            if (bx < 0 or by < 0 or bx > 7 or by > 7):
                reward = 10
                # record the s, a, r pair
                continue;
            dis_o = abs(bx-3.5)+abs(by-3.5)
            bx = bx + dirs[action][0]
            by = by + dirs[action][1]
            dis_n = abs(bx-3.5)+abs(by-3.5)
            if(dis_n>dis_o):
                reward = 1
                #record this s, a ,r pair
                continue
            else:
                reward = -1
                #record...
                continue

    #Randomly assign  the bomb and the agent
