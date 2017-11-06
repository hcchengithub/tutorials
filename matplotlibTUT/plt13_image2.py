import peforth; peforth.ok(loc=locals(),cmd="include xray.f")

'''
    \ Study 三寶
    \ 1. DOS Box title 
    \ 2. Breakpoint
    \ 3. Lab of copy-paste
        <accept> <text> 
        # ---------------------------------------------------------------------------
        # image data
        a = np.array(np.arctan(np.linspace(-1, 1, 9))).reshape(3,3)
        plt.imshow(a, interpolation='nearest', cmap='cool', origin='lower')
        plt.colorbar(shrink=.7)
        plt.xticks(())
        plt.yticks(())
        plt.show()
        # ---------------------------------------------------------------------------
        </text> -indent py: exec(pop(),harry_port()) \ If only globals is given, locals defaults to it.
        </accept> dictate 
'''

# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# 13 - image
"""
Please note, this script is for python3+.
If you are using python2+, please modify it accordingly.
"""

import matplotlib.pyplot as plt
import numpy as np

# image data
a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
              0.365348418405, 0.439599930621, 0.525083754405,
              0.423733120134, 0.525083754405, 0.651536351379]).reshape(3,3)

"""
for the value of "interpolation", check this:
http://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html
for the value of "origin"= ['upper', 'lower'], check this:
http://matplotlib.org/examples/pylab_examples/image_origin.html
"""
plt.imshow(a, interpolation='nearest', cmap='bone', origin='lower')
plt.colorbar(shrink=.92)

plt.xticks(())
plt.yticks(())
plt.show()

peforth.ok('11> ',cmd="parent inport")

'''
    imshow() image show
    colorbar() 類似前面介紹過的 legend , 圖例
    
    11> a :> shape . cr
    (3, 3)
    11> a . cr
    [[ 0.31366083  0.36534842  0.42373312]
     [ 0.36534842  0.43959993  0.52508375]
     [ 0.42373312  0.52508375  0.65153635]]
    11> a :> reshape(-1,1) . cr
    [[ 0.31366083]
     [ 0.36534842]
     [ 0.42373312]
     [ 0.36534842]
     [ 0.43959993]
     [ 0.52508375]
     [ 0.42373312]
     [ 0.52508375]
     [ 0.65153635]]
    11> a :> reshape(-1,) . cr
    [ 0.31366083  0.36534842  0.42373312  0.36534842  0.43959993  0.52508375
      0.42373312  0.52508375  0.65153635]
    11> a :> reshape(-1,0) . cr
    Failed in </py> (compiling=False): cannot reshape array of size 9 into shape (0)

11>
'''