# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# 4 - figure
"""
Please note, this script is for python3+.
If you are using python2+, please modify it accordingly.
Tutorial reference:
http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2

plt.figure()
plt.plot(x, y1)


plt.figure(num=3, figsize=(8, 5),)
plt.plot(x, y2)
# plot the second curve in this figure with certain parameters
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
plt.show()
import peforth;peforth.ok(loc=locals(),cmd=":> [0] inport")


'''
    \ Shows y1 and y2 plot on two different figure (window) at the same time
    OK plt :: figure() x y1 plt :: plot(pop(1),pop())
    OK plt :: figure() x y2 plt :: plot(pop(1),pop()) plt :: show()

    \ show() y1 and y2 plots in the same figure
    OK x y1 plt :: plot(pop(1),pop())
    OK x y2 plt :: plot(pop(1),pop()) plt :: show()

    \ show() y1 and pause, close the y1 plot then y2 plot will be shown
    OK x y1 plt :: plot(pop(1),pop()) plt :: show() 
    OK x y2 plt :: plot(pop(1),pop()) plt :: show()
    OK
'''

