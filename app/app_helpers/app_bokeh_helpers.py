#!/usr/bin/python3
# -*- coding: utf-8 -*-


import bokeh.models as bhm
import panel as pn
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure


def configure_bokeh_chart_properties(charts_list):
    for _, p in enumerate(charts_list):
        p.ygrid.minor_grid_line_color = "lightgrey"
        p.ygrid.minor_grid_line_alpha = 0.5
        p.toolbar.autohide = False
        p.yaxis.major_label_text_font_size = "12pt"
        p.xaxis.major_label_text_font_size = "12pt"
        p.title.text_font_size = "14pt"
        p.xaxis.axis_label_text_font_size = "14pt"
        p.yaxis.axis_label_text_font_size = "14pt"
        # p.xaxis.major_label_text_color = "black"
        # p.yaxis.major_label_text_color = "black"
        # p.xaxis.axis_label_text_color = "black"
        # p.yaxis.axis_label_text_color = "black"
        # p.xaxis.axis_label_text_font_style = "bold"


def create_bokeh_panes(dict_of_charts):
    bokeh_panes_dict = {
        f"bokeh_pane_{k}": pn.pane.Bokeh(v) for k, v in dict_of_charts.items()
    }
    return bokeh_panes_dict


def bokeh_horiz_bar_chart(
    data_source,
    y,
    right,
    yrange,
    height,
    color,
    tools,
    ptitle,
    toolbar_location=None,
    tooltips=None,
    fig_size=(250, 325),
):
    bar_chart = figure(
        y_range=yrange,
        plot_width=fig_size[0],
        plot_height=fig_size[1],
        title=ptitle,
        tools=tools,
        tooltips=tooltips,
        toolbar_location=toolbar_location,
    )
    bar_chart.hbar(
        y=y,
        right=right,
        height=height,
        source=data_source,
        # color=color,
        fill_color=color,
        line_color=color,
    )
    return bar_chart


def bokeh_horiz_spreads_chart(
    data_source,
    y,
    left,
    right,
    x_range,
    yrange,
    height,
    fill_color,
    line_color,
    tools,
    ptitle,
    toolbar_location=None,
    tooltips=None,
    fig_size=(250, 325),
):
    spreads_chart = figure(
        y_range=yrange,
        x_range=x_range,
        plot_width=fig_size[0],
        plot_height=fig_size[1],
        title=ptitle,
        tools=tools,
        tooltips=tooltips,
        toolbar_location=toolbar_location,
    )
    spreads_chart.hbar(
        y=y,
        left=left,
        right=right,
        height=height,
        fill_color=fill_color,
        line_color=line_color,
        source=data_source,
    )
    return spreads_chart
