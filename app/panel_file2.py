#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Space News Article Topic Predictor Dashboard."""


import calendar
import os
from datetime import date

import app_helpers.app_bokeh_helpers as abh
import app_helpers.app_dashboard_helpers as adh
import app_helpers.app_data_retrieval_helpers as adrh
import bokeh.models as bhm
import pandas as pd
import panel as pn
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.palettes import turbo
from bokeh.themes import Theme
from bokeh.transform import factor_cmap

PROJ_ROOT_DIR = os.getcwd()

panel_template = pn.template.MaterialTemplate(
    title="Topics from The Guardian's News Articles about Space Science"
)

pn.config.sizing_mode = "stretch_width"

curdoc().theme = Theme(filename="templates/bokeh_theme.yaml")

DATA_DIR = os.path.join(os.path.dirname(__file__))
data_filepath = os.path.join(PROJ_ROOT_DIR, "data", "dashboard_data.h5")
jinja2_templates_dir = os.path.join(PROJ_ROOT_DIR, "templates")

TOPICS_BY_RESIDUAL = [
    "Scientific Research about Dark Matter",
    "Search for E.T. life",
    "SpaceX Rocket Testing Reports",
    "Neil Armstrong",
    "Mars mission updates",
    "Topic 0",
    "Achievements of Hubble Space Telescope",
    "ISS updates",
    "Black Holes",
    "Anticipating or Reporting on Eclipses",
    "Dinosaurs",
    "Space Debris from Satellites",
    "Events relating to Virgin Galactic",
    "Report on Detection of Gravitational Waves",
    "Sun's influence on life across the Solar System",
    "Topic 33",
    "Reports about Problem of Global Warming",
    "Neuroscience Research",
    "Spacecraft Imaging of Dwarf distant planet Pluto",
    "On the Search for and Detection of Neutrinos",
    "Evidence of Water in the Solar System",
    "About Exploring the Moon",
]
TOPIC_O_TERMS = [
    "peopl",
    "thing",
    "think",
    "just",
    "time",
    "veri",
    "know",
    "work",
    "make",
    "look",
]
TOPIC_0_NER = [
    "VR",
    "Overview",
    "Nasa",
    "Pratscher",
    "Apollo",
    "The Orbital Perspective",
    "the University of Missouri",
]
beginning_date = date(2019, 11, 2)
ending_date = date(2020, 2, 28)

# set up widgets
topic_term = bhm.Select(
    title="Topic Name",
    value=TOPICS_BY_RESIDUAL[5],
    options=TOPICS_BY_RESIDUAL,
)
daterange = bhm.DateRangeSlider(
    title="Dates",
    value=(date(2019, 11, 2), date(2020, 2, 28)),
    start=date(2019, 11, 2),
    end=date(2020, 2, 28),
)
progress = pn.widgets.Progress(name="Progress", value=100, width=250)

# set up data sources
source = bhm.ColumnDataSource(data=dict(count=[], weekday=[]))
source_month = bhm.ColumnDataSource(data=dict(count=[], month=[]))
source_topics = bhm.ColumnDataSource(data=dict(count=[], topic=[]))
source_weights = bhm.ColumnDataSource(
    data=dict(term_weight=[], term=[], topic=[])
)
source_ner = bhm.ColumnDataSource(
    data=dict(entity_count=[], entity=[], topic=[])
)
df_groups = adh.load_data(data_filepath)
group = df_groups.groupby("topic")
source_group = ColumnDataSource(data=group)
tools = ""

# set up plots
days = [calendar.day_name[i - 1] for i in range(1, 8)][::-1]
weekday_freq_bar_chart = abh.bokeh_horiz_bar_chart(
    source,
    "weekday",
    "count",
    FactorRange(factors=days),
    1.0,
    "blue",
    tools,
    "News Articles by Day of Week",
    None,
    None,
    (275, 325),
)
topic_freq_bar_chart = abh.bokeh_horiz_bar_chart(
    source_topics,
    "topic",
    "count",
    FactorRange(factors=TOPICS_BY_RESIDUAL),
    1.0,
    "darkred",
    tools,
    "News Articles by Topic",
    None,
    None,
    (275, 650),
)
term_weights_bar_chart = abh.bokeh_horiz_bar_chart(
    source_weights,
    "term",
    "term_weight",
    TOPIC_O_TERMS,
    0.85,
    "#2FAA96",
    tools,
    "",
    None,
    adh.generate_tooltip(
        data_list=[["term", "Name", 14], ["term_weight", "Weight", 14]],
        tool_title_fontsize=16,
        tool_title="Term Weight",
    ),
    (275, 325),
)
month_freq_bar_chart = abh.bokeh_horiz_bar_chart(
    source_month,
    "month",
    "count",
    ["Nov", "Dec", "Jan", "Feb"],
    1.0,
    "blue",
    tools,
    "News Articles by Month",
    None,
    None,
    (275, 325),
)
entity_counts_bar_chart = abh.bokeh_horiz_bar_chart(
    source_ner,
    "entity",
    "entity_count",
    TOPIC_0_NER,
    0.95,
    "#694489",
    tools,
    "",
    None,
    adh.generate_tooltip(
        data_list=[
            ["entity", "Name", 14],
            ["entity_count", "Occurrences", 14],
        ],
        tool_title_fontsize=16,
        tool_title="topic",
        hover_template="hover_custom_title.j2",
    ),
    (275, 650),
)
index_cmap = factor_cmap(
    "topic",
    palette=turbo(len(TOPICS_BY_RESIDUAL)),
    factors=TOPICS_BY_RESIDUAL,
    end=1,
)
spreads_chart = abh.bokeh_horiz_spreads_chart(
    source_group,
    "topic",
    "resid_perc25_min",
    "resid_perc75_max",
    (0.79, 1.0),
    TOPICS_BY_RESIDUAL,
    0.85,
    index_cmap,
    index_cmap,
    tools,
    "Topic Residual Spreads",
    toolbar_location=None,
    tooltips=adh.generate_tooltip(
        data_list=[
            ["resid_perc25_mean", "25th Percentile", 14],
            ["resid_perc75_mean", "75th Percentile", 14],
        ],
        tool_title_fontsize=16,
        tool_title="topic",
        hover_template="hover_custom_title.j2",
    ),
    fig_size=(275, 650),
)

term_weights_bar_chart.xaxis.axis_label = "Term Weight"
term_weights_bar_chart.xaxis.axis_label_standoff = 0
entity_counts_bar_chart.xaxis.axis_label_standoff = 0

abh.configure_bokeh_chart_properties(
    [
        weekday_freq_bar_chart,
        topic_freq_bar_chart,
        term_weights_bar_chart,
        month_freq_bar_chart,
        entity_counts_bar_chart,
        spreads_chart,
    ]
)
bokeh_panes_dict = abh.create_bokeh_panes(
    {
        "weekday_freq_bar_chart": weekday_freq_bar_chart,
        "topic_freq_bar_chart": topic_freq_bar_chart,
        "term_weights_bar_chart": term_weights_bar_chart,
        "month_freq_bar_chart": month_freq_bar_chart,
        "entity_counts_bar_chart": entity_counts_bar_chart,
        "spreads_chart": spreads_chart,
    }
)


def topic_term_change(attrname, old, new):
    update()


def daterange_change(attrname, old, new):
    update()


def update():
    topic_selected = topic_term.value
    start_date, end_date = daterange.value_as_date
    # print(str(start_date), str(end_date))
    adh.perform_updates(
        start_date,
        end_date,
        progress,
        topic_selected,
        data_filepath,
        term_weights_bar_chart,
        entity_counts_bar_chart,
        topic_freq_bar_chart,
        source,
        source_weights,
        source_ner,
        source_month,
        source_topics,
        beginning_date,
        ending_date,
        topic_term,
    )


topic_term.on_change("value", topic_term_change)
daterange.on_change("value", daterange_change)

# set up layout
widgets = column(topic_term, daterange)

# initialize
update()

cb = pn.state.add_periodic_callback(update, 200, timeout=5000)

desc_sidebar_pane = adh.generate_sidebar_html()

panel_template.sidebar.append(widgets)
panel_template.sidebar.append(progress)
panel_template.sidebar.append(desc_sidebar_pane)
panel_template.main.append(
    pn.Row(
        pn.Column(
            pn.Row(
                bokeh_panes_dict["bokeh_pane_weekday_freq_bar_chart"],
                bokeh_panes_dict["bokeh_pane_month_freq_bar_chart"],
            ),
            pn.Row(bokeh_panes_dict["bokeh_pane_term_weights_bar_chart"]),
            pn.Row(bokeh_panes_dict["bokeh_pane_spreads_chart"]),
        ),
        pn.Column(
            pn.Row(bokeh_panes_dict["bokeh_pane_topic_freq_bar_chart"]),
            pn.Row(bokeh_panes_dict["bokeh_pane_entity_counts_bar_chart"]),
        ),
    )
)

panel_template.servable()
