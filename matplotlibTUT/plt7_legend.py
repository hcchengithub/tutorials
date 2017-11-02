
import peforth; peforth.ok(loc=locals(),cmd="include xray.f")

# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# 7 - legend
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
# set x limits
plt.xlim((-1, 2))
plt.ylim((-2, 3))

# set new ticks  我想 tick 就是 x-axis 上 plot() 的點位， x 的 linespace 是 -3～3 但點位是 -1～2
new_ticks = np.linspace(-1, 2, 5)
plt.xticks(new_ticks)
# set tick labels
plt.yticks([-2, -1.8, -1, 1.22, 3],
           [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])

l1, = plt.plot(x, y2, label='linear line')
l2, = plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--', label='square line')

# plt.legend(loc='upper right')
plt.legend(handles=[l1, l2], labels=['up', 'down'],  loc='best')
# the "," is very important in here l1, = plt... and l2, = plt... for this step

"""legend( handles=(line1, line2, line3),
           labels=('label1', 'label2', 'label3'),
           'upper right')
    The *loc* location codes are::

          'best' : 0,          (currently not supported for figure legends)
          'upper right'  : 1,
          'upper left'   : 2,
          'lower left'   : 3,
          'lower right'  : 4,
          'right'        : 5,
          'center left'  : 6,
          'center right' : 7,
          'lower center' : 8,
          'upper center' : 9,
          'center'       : 10,"""

plt.show()
import peforth;peforth.ok('11> ')

'''
# set new ticks  我想 tick 就是 x-axis 上 plot 用的點位， x 的 linespace 是 -3～3 但點位是 -1～2

    \ 確定 xticks 就是這麼設定的
    new_ticks plt :: xticks(pop(),['aa','bb','cc','dd','ee'])
    plt :: show()

    \ 試打 legend 也成功了！
    plt ::~ legend(handles=[v('l1'),v('l2')], labels=['gigi','heihei'],loc='upper right')
    plt :: show()

    \ 查看 plot() return value 
    11> dropall plt :> plot() .s
          0: [] (<class 'list'>)
    11> dropall x y2 plt :> plot(pop(1),pop()) .s
          0: [<matplotlib.lines.Line2D object at 0x00000203C6CDB5C0>] (<class 'list'>)

    \ 這麼奇怪的寫法，答案揭曉。。。。      
    l1, = plt.plot(x, y2, label='linear line')
    l2, = plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--', label='square line')
    \ 。。。。 看這實驗就懂了：
    dropall <py>
        x, = [11]   # 加個逗點，是要取出 list 裡面的 cell 
        y = [22]    # 不加逗點，就是整個 list 
        push(x)
        push(y)
    </py> .s
          0:          11           Bh (<class 'int'>)
          1: [22] (<class 'list'>)

    \ 可以用 peforth 任意玩耍了 
    x y2 plt :>~ plot(pop(1), pop(), label='aaa') 
    :> [0] to l1
    x y1 plt :>~ plot(pop(1), pop(), color='red', linewidth=1.0, linestyle='--', label='bbb') 
    :> [0] to l2
    l1 l2 plt ::~ legend(handles=[pop(1),pop()], labels=['gigi','heihei'],loc='lower right')
    plt :: show()
      
    legend 設定過了就一直跟著這個 figure() 上不會變
          
'''
