#!/usr/bin/python3
# -*- coding: utf-8 -*-


from bokeh.io import show
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.plotting import figure


def plot_bokeh_horiz_barchart(
    df,
    right,
    y,
    yrange,
    bar_height,
    tooltip,
    ptitle,
    bar_fill_color="#4287f5",
    bar_line_color="#4287f5",
    fig_size=(600, 450),
):
    source = ColumnDataSource(data=df)
    hover = HoverTool(mode="mouse")
    hover.tooltips = tooltip
    p = figure(
        y_range=yrange,
        plot_width=fig_size[0],
        plot_height=fig_size[1],
        title=ptitle,
        tools="",
        toolbar_location=None,
    )
    p.hbar(
        y=y,
        right=right,
        height=bar_height,
        source=source,
        fill_color=bar_fill_color,
        line_color=bar_line_color,
    )
    p.add_tools(hover)
    configure_bokeh_chart_properties([p])
    show(p)


def configure_bokeh_chart_properties(charts_list):
    for _, p in enumerate(charts_list):
        p.ygrid.minor_grid_line_color = "lightgrey"
        p.ygrid.minor_grid_line_alpha = 0.5
        p.toolbar.autohide = False
        p.xaxis.axis_label_text_font_size = "14pt"
        p.yaxis.axis_label_text_font_size = "14pt"
        p.xaxis.major_label_text_font_size = "12pt"
        p.yaxis.major_label_text_font_size = "12pt"
        p.title.text_font_size = "12pt"
        p.axis.major_label_text_color = "black"
        p.axis.major_tick_line_color = "black"
        p.axis.minor_tick_line_color = "black"
