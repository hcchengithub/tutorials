
import peforth; peforth.ok(loc=locals(),cmd="include xray.f")

# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# 10 - scatter
"""
Please note, this script is for python3+.
If you are using python2+, please modify it accordingly.
Tutorial reference:
http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
"""

import matplotlib.pyplot as plt
import numpy as np

n = 1024    # data size
X = np.random.normal(0, 1, n)
Y = np.random.normal(0, 1, n)
T = np.arctan2(Y, X)    # for color later on

plt.scatter(X, Y, s=75, c=T, alpha=.5)

plt.xlim(-1.5, 1.5)
plt.xticks(())  # ignore xticks
plt.ylim(-1.5, 1.5)
plt.yticks(())  # ignore yticks

plt.show()

peforth.ok('11> ')


'''

import peforth; peforth.ok(loc=locals(),cmd="include xray.f")
peforth.ok('11> ')

    \ Lab of copy-paste  一張漂亮的商標圖案！
        <accept> <text> 
        locals().update(harry_port());  # bring in all things
        # ---------------------------------------------------------------------------
        n = 100    # data size
        X = np.random.normal(0, 1, n)                # <------- 0 不是起點，而是中位數
        Y = np.random.normal(0, 1, n)                # <------- 0 不是起點，而是中位數   
        T = np.arctan(Y, X)    # for color later on
        plt.scatter(X, Y, s=75, c=T, alpha=.5)
        plt.xlim(-10.5, 11.5)
        plt.xticks(())  # ignore xticks
        plt.ylim(-10.5, 11.5)
        plt.yticks(())  # ignore yticks
        plt.show()
        # ---------------------------------------------------------------------------
        </text> -indent py: exec(pop())
        </accept> dictate 

'''