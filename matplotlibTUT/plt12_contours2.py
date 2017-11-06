import peforth; peforth.ok(loc=locals(),cmd="include xray.f")

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
peforth.ok('11> ')


'''
    alpha 是透明度
    color map = cmap = cm , 有各式各樣的 color map , plt.cm.hot 是其中一種
                            看 help(plt.cm) 就知道了

    11> words
    ... snip ....
    peforth plt np f n x y X Y C
    11>

    \ f(x,y) 是個很優美的曲面。老師說不要糾結於這個曲面。
    11>   f . cr
    <function f at 0x000002850BC43E18>
    11>

    11> n . cr
    256
    11>

    \ 觀察 X Y 就可以知道 meshgrid 是啥了， -3～3 從左下角延伸到右上角，相當於
    \ 把一維的 X 軸擴展成二維的 地板平面，這就是 meshgrid

    11> X . cr
    [[-3.         -2.97647059 -2.95294118 ...,  2.95294118  2.97647059  3.        ]
     [-3.         -2.97647059 -2.95294118 ...,  2.95294118  2.97647059  3.        ]
     [-3.         -2.97647059 -2.95294118 ...,  2.95294118  2.97647059  3.        ]
     ...,
     [-3.         -2.97647059 -2.95294118 ...,  2.95294118  2.97647059  3.        ]
     [-3.         -2.97647059 -2.95294118 ...,  2.95294118  2.97647059  3.        ]
     [-3.         -2.97647059 -2.95294118 ...,  2.95294118  2.97647059  3.        ]]

    11> X type . cr
    <class 'numpy.ndarray'>

    11> X :> shape . cr
    (256, 256)
    11>

    OK Y . cr
    [[-3.         -3.         -3.         ..., -3.         -3.         -3.        ]
     [-2.97647059 -2.97647059 -2.97647059 ..., -2.97647059 -2.97647059
      -2.97647059]
     [-2.95294118 -2.95294118 -2.95294118 ..., -2.95294118 -2.95294118
      -2.95294118]
     ...,
     [ 2.95294118  2.95294118  2.95294118 ...,  2.95294118  2.95294118
       2.95294118]
     [ 2.97647059  2.97647059  2.97647059 ...,  2.97647059  2.97647059
       2.97647059]
     [ 3.          3.          3.         ...,  3.          3.          3.        ]]
    OK

    11> Y type . cr
    <class 'numpy.ndarray'>
    11>

    11> Y :> shape . cr
    (256, 256)
    11>
                                   n. 轮廓；等高线；周线；电路；概要  
    11> C type . cr                vvvvvvv
    <class 'matplotlib.contour.QuadContourSet'>
    11>

    11> C dir . cr
    ['_A', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_add_label', '_auto', '_autolev', '_check_xyz', '_contour_args', '_contour_generator', '_contour_level_args', '_corner_mask', '_get_allsegs_and_allkinds', '_get_label_clabeltext', '_get_label_text', '_get_lowers_and_uppers', '_initialize_x_y', '_levels', '_make_paths', '_maxs', '_mins', '_process_args', '_process_colors', '_process_levels', '_process_linestyles', '_process_linewidths', '_transform', '_use_clabeltext', 'add_checker', 'add_label', 'add_label_clabeltext', 'add_label_near', 'allkinds', 'allsegs', 'alpha', 'antialiased', 'autoscale', 'autoscale_None', 'ax', 'calc_label_rot_and_inline', 'callbacksSM', 'changed', 'check_update', 'cl', 'cl_cvalues', 'cl_xy', 'clabel', 'cmap', 'collections', 'colorbar', 'colors', 'contour_doc', 'cvalues', 'extend', 'extent', 'filled', 'find_nearest_contour', 'get_alpha', 'get_array', 'get_clim', 'get_cmap', 'get_label_coords', 'get_label_width', 'get_real_label_width', 'get_text', 'get_transform', 'hatches', 'labelCValueList', 'labelCValues', 'labelFmt', 'labelFontProps', 'labelFontSizeList', 'labelIndiceList', 'labelLevelList', 'labelManual', 'labelMappable', 'labelTexts', 'labelTextsList', 'labelXYs', 'labels', 'layers', 'legend_elements', 'levels', 'linestyles', 'linewidths', 'locate_label', 'locator', 'logscale', 'monochrome', 'nchunk', 'norm', 'origin', 'pop_label', 'print_label', 'rightside_up', 'set_alpha', 'set_array', 'set_clim', 'set_cmap', 'set_label_props', 'set_norm', 'tcolors', 'tlinewidths', 'to_rgba', 'too_close', 'update_dict', 'vmax', 'vmin', 'zmax', 'zmin']
    11>

    11> C . cr
    <matplotlib.contour.QuadContourSet object at 0x000002850FCAED30>
    11>

    11> n . cr
    256
    11>

    11> x . cr
    [-3.         -2.97647059 -2.95294118 -2.92941176 -2.90588235 -2.88235294
     -2.85882353 -2.83529412 -2.81176471 -2.78823529 -2.76470588 -2.74117647
      .... snip .....
      2.78823529  2.81176471  2.83529412  2.85882353  2.88235294  2.90588235
      2.92941176  2.95294118  2.97647059  3.        ]

    11> y . cr
    [-3.         -2.97647059 -2.95294118 -2.92941176 -2.90588235 -2.88235294
     -2.85882353 -2.83529412 -2.81176471 -2.78823529 -2.76470588 -2.74117647
     -2.71764706 -2.69411765 -2.67058824 -2.64705882 -2.62352941 -2.6
     -2.57647059 -2.55294118 -2.52941176 -2.50588235 -2.48235294 -2.45882353
     -2.43529412 -2.41176471 -2.38823529 -2.36470588 -2.34117647 -2.31764706
     -2.29411765 -2.27058824 -2.24705882 -2.22352941 -2.2        -2.17647059
     .... snip .....
      2.92941176  2.95294118  2.97647059  3.        ]
    11>


    \ Lab of copy-paste
        <accept> <text> 
        # ---------------------------------------------------------------------------
        def f(x,y):
            # the height function
            #return (1 - x / 2 + x**7 + y**5) * np.exp(-x**2 -y**2) # 漂亮曲面
            return (x) # 簡化曲面，看這個就懂了
            # return (np.sin(x)) # 沒比單 x 好
            # 總之 f(x,y) 只是提供一個 z 軸上的高度而已

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

        # plt.xticks(())
        # plt.yticks(())
        plt.show()
        # ---------------------------------------------------------------------------
        </text> -indent py: exec(pop(),harry_port())
        </accept> dictate 

'''
