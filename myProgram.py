import numpy as np
dirs = [[-1,0],[0,1],[1,0],[0,-1]]
step_limit = 1000
n = 8
initQ(8)

def initQ(size):
    Q = np.zeros((size,size,size,size))

def MCEpisode(alpha, epsilon, complex_reward = false):
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
