# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# 12 - contours
"""
Please note, this script is for python3+.
If you are using python2+, please modify it accordingly.
Tutorial reference:
http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
"""

import matplotlib.pyplot as plt
import numpy as np

def f(x,y):
    # the height function
    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)

n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X,Y = np.meshgrid(x, y)

# use plt.contourf to filling contours
# X, Y and value for (X,Y) point
plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap=plt.cm.hot)

# use plt.contour to add contour lines
C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)
# adding label
plt.clabel(C, inline=True, fontsize=10)

plt.xticks(())
plt.yticks(())
plt.show()

'''

import peforth; peforth.ok(loc=locals(),cmd="include xray.f")
peforth.ok('11> ')
    
    words
    ... snip ....
    peforth plt np n X Y1 Y2 x y
    11> n . cr
    12
    11> X . cr
    [ 0  1  2  3  4  5  6  7  8  9 10 11]
    11> Y1 . cr
    [ 0.8179656   0.49323403  0.4284618   0.68595935  0.51233127  0.51762588
      0.4130633   0.37405451  0.22465381  0.15635563  0.16062196  0.04750518]
    11> Y2 . cr
    [ 0.55254726  0.55435316  0.42639684  0.37550261  0.6354508   0.33628373
      0.39926031  0.27926584  0.19029853  0.19743172  0.15091777  0.06192359]
    11> x . cr
    11
    11> y . cr
    0.0619235874571
    11>

    \ range(n) 與 np.arange(n) 的差別
    11> py> range(10) . cr
    range(0, 10)
    11> np :> arange(10) . cr
    [0 1 2 3 4 5 6 7 8 9]
    11> np :> arange(10) type . cr
    <class 'numpy.ndarray'>
    11> py> range(10) type . cr
    <class 'range'>
    11>    
    
    \ Lab of copy-paste
        <accept> <text> 
        locals().update(harry_port());  # bring in all things
        # ---------------------------------------------------------------------------
        plt.figure()
        n = 12
        X = np.arange(n)
        Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
        Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

        plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
        plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

        for x, y in zip(X, Y1):
            # ha: horizontal alignment
            # va: vertical alignment
            #plt.text(x, y + 0.1, '%.2f' % y, ha='center', va='bottom')
            plt.text(x, y + 0.1, '%.2f' % y)  # 柱狀圖本身是以中心點展開柱子的寬度
            pass                              # plt.text() 也是一樣，指定 ha va 與位置的關係看要怎麼算 

        for x, y in zip(X, Y2):
            # ha: horizontal alignment
            # va: vertical alignment
            plt.text(x, -y - 0.1, '%.2f' % y, ha='center', va='top')

        plt.xlim(-.5, n)
        plt.xticks(())
        plt.ylim(-1.25, 1.25)
        plt.yticks(())

        plt.show()
        # ---------------------------------------------------------------------------
        </text> -indent py: exec(pop())
        </accept> dictate 

    \ zip() 把多個 interater 一次取出    
    11> py> zip(v('X'),v('Y1')) py: help(pop())
    Help on zip object:
    Help on zip object:
    class zip(object)
     |  zip(iter1 [,iter2 [...]]) --> zip object
     |
     |  Return a zip object whose .__next__() method returns a tuple where
     |  the i-th element comes from the i-th iterable argument.  The .__next__()
     |  method continues until the shortest iterable in the argument sequence
     |  is exhausted and then it raises StopIteration.
        
'''
