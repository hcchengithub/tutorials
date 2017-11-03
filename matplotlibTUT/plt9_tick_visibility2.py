
import peforth; peforth.ok(loc=locals(),cmd="include xray.f")

# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# 9 - tick_visibility
"""
Please note, this script is for python3+.
If you are using python2+, please modify it accordingly.
Tutorial reference:
http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y = 0.1*x

plt.figure()
plt.plot(x, y, linewidth=10)
plt.ylim(-2, 2)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))


for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(12)
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.7))
plt.show()

peforth.ok('11> ')

'''

    \ python list's and tuple's can add but not set's
    11> py> [11,22,33] py> [44,55,66] + . cr
    [11, 22, 33, 44, 55, 66]
    11> py> (11,22,33) py> (44,55,66) + . cr
    (11, 22, 33, 44, 55, 66)
    11> py> {11,22,33} py> {44,55,66} + . cr
    Word in phaseB <Word '+'>: unsupported operand type(s) for +: 'set' and 'set'

    \ 看看 ax.get_xticklabels() 是啥東西？ 怎能相加？ 
    11> ax :> get_yticklabels() type . cr
    <class 'matplotlib.cbook.silent_list'>
    11> ax :> get_yticklabels()[0] . cr
    Text(0,-2,'−2.0')
    11> ax :> get_yticklabels()[1] . cr
    Text(0,-1.5,'−1.5')
    11> ax :> get_yticklabels() py> len(pop()) . cr
    9
    11> ax :> get_xticklabels() py> len(pop()) . cr
    9
    11> ax :> get_xticklabels() ax :> get_yticklabels() + py> len(pop()) . cr
    18
    11> ax :> get_yticklabels()[0] type . cr
    <class 'matplotlib.text.Text'>
    
    11> ax :> get_yticklabels()[0] dir . cr
    ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', 
    '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', 
    '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', 
    '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', 
    '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 
    '__weakref__', '_agg_filter', '_alpha', '_animated', '_axes', 
    '_bbox_patch', '_cached', '_clipon', '_clippath', '_color', '_contains', 
    '_draw_bbox', '_fontproperties', '_get_dist_to_box', '_get_layout', 
    '_get_multialignment', '_get_rendered_text_width', 
    '_get_wrap_line_width', '_get_wrapped_text', '_get_xy_display', '_gid', 
    '_horizontalalignment', '_label', '_linespacing', '_mouseover', 
    '_multialignment', '_oid', '_path_effects', '_picker', '_prop_order', 
    '_propobservers', '_rasterized', '_remove_method', '_renderer', 
    '_rotation', '_rotation_mode', '_set_gc_clip', '_sketch', '_snap', 
    '_stale', '_sticky_edges', '_text', '_transform', '_transformSet', 
    '_update_clip_properties', '_url', '_usetex', '_verticalalignment', 
    '_visible', '_wrap', '_x', '_y', 'add_callback', 'aname', 'axes', 
    'clipbox', 'contains', 'convert_xunits', 'convert_yunits', 'draw', 
    'eventson', 'figure', 'findobj', 'format_cursor_data', 'get_agg_filter', 
    'get_alpha', 'get_animated', 'get_axes', 'get_bbox_patch', 
    'get_children', 'get_clip_box', 'get_clip_on', 'get_clip_path', 
    'get_color', 'get_contains', 'get_cursor_data', 'get_family', 
    'get_figure', 'get_font_properties', 'get_fontfamily', 'get_fontname', 
    'get_fontproperties', 'get_fontsize', 'get_fontstretch', 'get_fontstyle', 
    'get_fontvariant', 'get_fontweight', 'get_gid', 'get_ha', 
    'get_horizontalalignment', 'get_label', 'get_name', 'get_path_effects', 
    'get_picker', 'get_position', 'get_prop_tup', 'get_rasterized', 
    'get_rotation', 'get_rotation_mode', 'get_size', 'get_sketch_params', 
    'get_snap', 'get_stretch', 'get_style', 'get_text', 'get_transform', 
    'get_transformed_clip_path_and_affine', 'get_unitless_position', 
    'get_url', 'get_usetex', 'get_va', 'get_variant', 
    'get_verticalalignment', 'get_visible', 'get_weight', 
    'get_window_extent', 'get_wrap', 'get_zorder', 'have_units', 'hitlist', 
    'is_figure_set', 'is_math_text', 'is_transform_set', 'mouseover', 
    'pchanged', 'pick', 'pickable', 'properties', 'remove', 
    'remove_callback', 'set', 'set_agg_filter', 'set_alpha', 'set_animated', <------- alpha 
    'set_axes', 'set_backgroundcolor', 'set_bbox', 'set_clip_box',  <----------- bbox
    'set_clip_on', 'set_clip_path', 'set_color', 'set_contains',  <------------- color 
    'set_family', 'set_figure', 'set_font_properties', 'set_fontname', 
    'set_fontproperties', 'set_fontsize', 'set_fontstretch', 'set_fontstyle', 
    'set_fontvariant', 'set_fontweight', 'set_gid', 'set_ha', 
    'set_horizontalalignment', 'set_label', 'set_linespacing', 'set_ma', 
    'set_multialignment', 'set_name', 'set_path_effects', 'set_picker', 
    'set_position', 'set_rasterized', 'set_rotation', 'set_rotation_mode', 
    'set_size', 'set_sketch_params', 'set_snap', 'set_stretch', 'set_style', 
    'set_text', 'set_transform', 'set_url', 'set_usetex', 'set_va', 
    'set_variant', 'set_verticalalignment', 'set_visible', 'set_weight', 
    'set_wrap', 'set_x', 'set_y', 'set_zorder', 'stale', 'stale_callback', 
    'sticky_edges', 'update', 'update_bbox_position_size', 'update_from', 
    'zorder']
    11>

'''
