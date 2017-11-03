
import peforth; peforth.ok(loc=locals(),cmd="include xray.f")


# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# 6 - axis setting
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
plt.plot(x, y2)
# plot the second curve in this figure with certain parameters
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
# set x limits
plt.xlim((-1, 2))
plt.ylim((-2, 3))

# set new ticks
new_ticks = np.linspace(-1, 2, 5)
plt.xticks(new_ticks)
# set tick labels
plt.yticks([-2, -1.8, -1, 1.22, 3],
           ['$really\ bad$', '$bad$', '$normal$', '$good$', '$really\ good$'])
# to use '$ $' for math text and nice looking, e.g. '$\pi$'

# gca = 'get current axes'    
ax = plt.gca()

'''
# set_ticks_position() and set_position() both 或之一設過以後再用 peforth
# 改，軸線都亂掉，可能要先把設過的消掉才行。

ax.spines['right'].set_color('none')  # spines 是四邊的邊框 Morvan 說的
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
# ACCEPTS: [ 'top' | 'bottom' | 'both' | 'default' | 'none' ]

ax.spines['bottom'].set_position(('data', 0))
# the 1st is in 'outward' | 'axes' | 'data'
# axes: percentage of y axis
# data: depend on y data

ax.yaxis.set_ticks_position('left')
# ACCEPTS: [ 'left' | 'right' | 'both' | 'default' | 'none' ]

ax.spines['left'].set_position(('data',0))
'''

import peforth;peforth.ok('11> ',loc=locals(),cmd=":> [0] inport")
plt.show()
import peforth;peforth.ok('22> ',loc=locals(),cmd=":> [0] inport")


'''
    OK plt :> gca py: help(pop())
        OK plt :> gca py: help(pop())
        Help on function gca in module matplotlib.pyplot:

        gca(**kwargs)
            Get the current :class:`~matplotlib.axes.Axes` instance on the
            current figure matching the given keyword args, or create one.

            Examples
            --------
            To get the current polar axes on the current figure::

                plt.gca(projection='polar')

            If the current axes doesn't exist, or isn't a polar one, the appropriate
            axes will be created and then returned.

            See Also
            --------
            matplotlib.figure.Figure.gca : The figure's gca method.
    
    OK words
    ...snip... plt np x y1 y2 new_ticks ax peforth
    OK __main__ dir . cr
    ['__annotations__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'ax', 'new_ticks', 'np', 'peforth', 'plt', 'x', 'y1', 'y2']
    OK ax dir . cr
    ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_add_text', '_adjustable', '_agg_filter', '_alpha', '_anchor', '_animated', '_aspect', '_autoscaleXon', '_autoscaleYon', '_axes', '_axes_class', '_axes_locator', '_axisbelow', '_cachedRenderer', '_clipon', '_clippath', '_connected', '_contains', '_current_image', '_cursorProps', '_facecolor', '_frameon', '_gci', '_gen_axes_patch', '_gen_axes_spines', '_get_axis_list', '_get_legend_handles', '_get_lines', '_get_patches_for_fill', '_get_view', '_gid', '_gridOn', '_hold', '_init_axis', '_label', '_left_title', '_make_twin_axes', '_mouseover', '_navigate', '_navigate_mode', '_oid', '_originalPosition', '_path_effects', '_pcolorargs', '_picker', '_position', '_process_unit_info', '_prop_order', '_propobservers', '_rasterization_zorder', '_rasterized', '_remove_method', '_right_title', '_sci', '_set_artist_props', '_set_gc_clip', '_set_lim_and_transforms', '_set_view', '_set_view_from_bbox', '_shared_x_axes', '_shared_y_axes', '_sharex', '_sharey', '_sketch', '_snap', '_stale', '_sticky_edges', '_subplotspec', '_tight', '_transform', '_transformSet', '_update_line_limits', '_update_patch_limits', '_update_transScale', '_url', '_use_sticky_edges', '_visible', '_xaxis_transform', '_xcid', '_xmargin', '_yaxis_transform', '_ycid', '_ymargin', 'acorr', 'add_artist', 'add_callback', 'add_collection', 'add_container', 'add_image', 'add_line', 'add_patch', 'add_table', 'aname', 'angle_spectrum', 'annotate', 'apply_aspect', 'arrow', 'artists', 'autoscale', 'autoscale_view', 'axes', 'axesPatch', 'axhline', 'axhspan', 'axis', 'axison', 'axvline', 'axvspan', 'bar', 'barbs', 'barh', 'bbox', 'boxplot', 'broken_barh', 'bxp', 'callbacks', 'can_pan', 'can_zoom', 'change_geometry', 'cla', 'clabel', 'clear', 'clipbox', 'cohere', 'colNum', 'collections', 'containers', 'contains', 'contains_point', 'contour', 'contourf', 'convert_xunits', 'convert_yunits', 'csd', 'dataLim', 'drag_pan', 'draw', 'draw_artist', 'end_pan', 'errorbar', 'eventplot', 'eventson', 'figbox', 'figure', 'fill', 'fill_between', 'fill_betweenx', 'findobj', 'fmt_xdata', 'fmt_ydata', 'format_coord', 'format_cursor_data', 'format_xdata', 'format_ydata', 'get_adjustable', 'get_agg_filter', 'get_alpha', 'get_anchor', 'get_animated', 'get_aspect', 'get_autoscale_on', 'get_autoscalex_on', 'get_autoscaley_on', 'get_axes', 'get_axes_locator', 'get_axis_bgcolor', 'get_axisbelow', 'get_children', 'get_clip_box', 'get_clip_on', 'get_clip_path', 'get_contains', 'get_cursor_data', 'get_cursor_props', 'get_data_ratio', 'get_data_ratio_log', 'get_default_bbox_extra_artists', 'get_facecolor', 'get_fc', 'get_figure', 'get_frame_on', 'get_geometry', 'get_gid', 'get_images', 'get_label', 'get_legend', 'get_legend_handles_labels', 'get_lines', 'get_navigate', 'get_navigate_mode', 'get_path_effects', 'get_picker', 'get_position', 'get_rasterization_zorder', 'get_rasterized', 'get_renderer_cache', 'get_shared_x_axes', 'get_shared_y_axes', 'get_sketch_params', 'get_snap', 'get_subplotspec', 'get_tightbbox', 'get_title', 'get_transform', 'get_transformed_clip_path_and_affine', 'get_url', 'get_visible', 'get_window_extent', 'get_xaxis', 'get_xaxis_text1_transform', 'get_xaxis_text2_transform', 'get_xaxis_transform', 'get_xbound', 'get_xgridlines', 'get_xlabel', 'get_xlim', 'get_xmajorticklabels', 'get_xminorticklabels', 'get_xscale', 'get_xticklabels', 'get_xticklines', 'get_xticks', 'get_yaxis', 'get_yaxis_text1_transform', 'get_yaxis_text2_transform', 'get_yaxis_transform', 'get_ybound', 'get_ygridlines', 'get_ylabel', 'get_ylim', 'get_ymajorticklabels', 'get_yminorticklabels', 'get_yscale', 'get_yticklabels', 'get_yticklines', 'get_yticks', 'get_zorder', 'grid', 'has_data', 'have_units', 'hexbin', 'hist', 'hist2d', 'hitlist', 'hlines', 'hold', 'ignore_existing_data_limits', 'images', 'imshow', 'in_axes', 'invert_xaxis', 'invert_yaxis', 'is_figure_set', 'is_first_col', 'is_first_row', 'is_last_col', 'is_last_row', 'is_transform_set', 'ishold', 'label_outer', 'legend', 'legend_', 'lines', 'locator_params', 'loglog', 'magnitude_spectrum', 'margins', 'matshow', 'minorticks_off', 'minorticks_on', 'mouseover', 'mouseover_set', 'name', 'numCols', 'numRows', 'patch', 'patches', 'pchanged', 'pcolor', 'pcolorfast', 'pcolormesh', 'phase_spectrum', 'pick', 'pickable', 'pie', 'plot', 'plot_date', 'properties', 'psd', 'quiver', 'quiverkey', 'redraw_in_frame', 'relim', 'remove', 'remove_callback', 'reset_position', 'rowNum', 'scatter', 'semilogx', 'semilogy', 'set', 'set_adjustable', 'set_agg_filter', 'set_alpha', 'set_anchor', 'set_animated', 'set_aspect', 'set_autoscale_on', 'set_autoscalex_on', 'set_autoscaley_on', 'set_axes', 'set_axes_locator', 'set_axis_bgcolor', 'set_axis_off', 'set_axis_on', 'set_axisbelow', 'set_clip_box', 'set_clip_on', 'set_clip_path', 'set_color_cycle', 'set_contains', 'set_cursor_props', 'set_facecolor', 'set_fc', 'set_figure', 'set_frame_on', 'set_gid', 'set_label', 'set_navigate', 'set_navigate_mode', 'set_path_effects', 'set_picker', 'set_position', 'set_prop_cycle', 'set_rasterization_zorder', 'set_rasterized', 'set_sketch_params', 'set_snap', 'set_subplotspec', 'set_title', 'set_transform', 'set_url', 'set_visible', 'set_xbound', 'set_xlabel', 'set_xlim', 'set_xmargin', 'set_xscale', 'set_xticklabels', 'set_xticks', 'set_ybound', 'set_ylabel', 'set_ylim', 'set_ymargin', 'set_yscale', 'set_yticklabels', 'set_yticks', 'set_zorder', 'specgram', 'spines', 'spy', 'stackplot', 'stale', 'stale_callback', 'start_pan', 'stem', 'step', 'sticky_edges', 'streamplot', 'table', 'tables', 'text', 'texts', 'tick_params', 'ticklabel_format', 'title', 'titleOffsetTrans', 'transAxes', 'transData', 'transLimits', 'transScale', 'tricontour', 'tricontourf', 'tripcolor', 'triplot', 'twinx', 'twiny', 'update', 'update_datalim', 'update_datalim_bounds', 'update_datalim_numerix', 'update_from', 'update_params', 'use_sticky_edges', 'viewLim', 'violin', 'violinplot', 'vlines', 'xaxis', 'xaxis_date', 'xaxis_inverted', 'xcorr', 'yaxis', 'yaxis_date', 'yaxis_inverted', 'zorder']
    OK

    \ 瞭解 set_color() 之前先查有沒有 get_color() ? 
    OK ax :> spines['right'] dir . cr
    ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_adjust_location', '_agg_filter', '_alpha', '_animated', '_antialiased', '_axes', '_bounds', '_calc_offset_transform', '_capstyle', '_clipon', '_clippath', '_combined_transform', '_contains', '_dashes', '_dashoffset', '_edge_default', '_edgecolor', '_ensure_position_is_set', '_facecolor', '_fill', '_gid', '_hatch', '_hatch_color', '_joinstyle', '_label', '_linestyle', '_linewidth', '_mouseover', '_oid', '_original_edgecolor', '_original_facecolor', '_patch_transform', '_patch_type', '_path', '_path_effects', '_picker', '_position', '_process_radius', '_prop_order', '_propobservers', '_rasterized', '_recompute_transform', '_remove_method', '_set_edgecolor', '_set_facecolor', '_set_gc_clip', '_sketch', '_smart_bounds', '_snap', '_spine_transform', '_stale', '_sticky_edges', '_transform', '_transformSet', '_url', '_us_dashes', '_visible', 'add_callback', 'aname', 'axes', 'axis', 'circular_spine', 'cla', 'clipbox', 'contains', 'contains_point', 'convert_xunits', 'convert_yunits', 'draw', 'eventson', 'figure', 'fill', 'findobj', 'format_cursor_data', 'get_aa', 'get_agg_filter', 'get_alpha', 'get_animated', 'get_antialiased', 'get_axes', 'get_bounds', 'get_capstyle', 'get_children', 'get_clip_box', 'get_clip_on', 'get_clip_path', 'get_contains', 'get_cursor_data', 'get_data_transform', 'get_ec', 'get_edgecolor', 'get_extents', 'get_facecolor', 'get_fc', 'get_figure', 'get_fill', 'get_gid', 'get_hatch', 'get_joinstyle', 'get_label', 'get_linestyle', 'get_linewidth', 'get_ls', 'get_lw', 'get_patch_transform', 'get_path', 'get_path_effects', 'get_picker', 'get_position', 'get_rasterized', 'get_sketch_params', 'get_smart_bounds', 'get_snap', 'get_spine_transform', 'get_transform', 'get_transformed_clip_path_and_affine', 'get_url', 'get_verts', 'get_visible', 'get_window_extent', 'get_zorder', 'have_units', 'hitlist', 'is_figure_set', 'is_frame_like', 'is_transform_set', 'linear_spine', 'mouseover', 'pchanged', 'pick', 'pickable', 'properties', 'register_axis', 'remove', 'remove_callback', 'set', 'set_aa', 'set_agg_filter', 'set_alpha', 'set_animated', 'set_antialiased', 'set_axes', 'set_bounds', 'set_capstyle', 'set_clip_box', 'set_clip_on', 'set_clip_path', 'set_color', 'set_contains', 'set_ec', 'set_edgecolor', 'set_facecolor', 'set_fc', 'set_figure', 'set_fill', 'set_gid', 'set_hatch', 'set_joinstyle', 'set_label', 'set_linestyle', 'set_linewidth', 'set_ls', 'set_lw', 'set_patch_circle', 'set_patch_line', 'set_path_effects', 'set_picker', 'set_position', 'set_rasterized', 'set_sketch_params', 'set_smart_bounds', 'set_snap', 'set_transform', 'set_url', 'set_visible', 'set_zorder', 'spine_type', 'stale', 'stale_callback', 'sticky_edges', 'update', 'update_from', 'validCap', 'validJoin', 'zorder']
    OK ax :> spines['right'].get_edgecolor . cr
    <bound method Patch.get_edgecolor of <matplotlib.spines.Spine object at 0x000001781A80B668>>
    
    \ 被設成 none 都是 0
    OK ax :> spines['right'].get_edgecolor() . cr
    (0.0, 0.0, 0.0, 0.0)

    \ 改成 green 之後，證實讀回的方式是這個
    
    ax :: spines['right'].set_color('green')
    ax :> spines['right'].get_edgecolor() \ (0.0, 0.5019607843137255, 0.0, 1.0)

    ax :: spines['top'].set_color('yellow')
    ax :> spines['top'].get_edgecolor() \ (1.0, 1.0, 0.0, 1.0)
    
    \ continue 查看效果 ==> Good!
    \ 然後到處改一改。。。
    
    ax :: xaxis.set_ticks_position('top')
    ax :: spines['top'].set_position(('data',0))

    ax :: yaxis.set_ticks_position('right')
    ax :: spines['right'].set_position(('data',0))
    
    
'''
