#!/usr/bin/python3
# -*- coding: utf-8 -*-


import os
from typing import List

import altair as alt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

SMALL_SIZE = 26
MEDIUM_SIZE = 28
BIGGER_SIZE = 30
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes\n",
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title\n",
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels\n",
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels\n",
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels\n",
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize\n",
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title\n",
sns.set_style("darkgrid", {"legend.frameon": False}),
sns.set_context("talk", font_scale=0.95, rc={"lines.linewidth": 2.5})
alt.renderers.set_embed_options(
    padding={"left": 0, "right": 0, "bottom": 0, "top": 0}
)


def combine_string_and_nan_columns(df):
    """Combine text columns that could also contain NaNs"""
    arr = df.add(", ").fillna("").values.tolist()
    s = pd.Series(
        ["".join(x).strip(", ") for x in arr], index=df.index
    ).replace("^$", np.nan, regex=True)
    s = s.where(s.notnull(), None)
    return s


def plot_horiz_bar(
    df,
    ptitle,
    y_tick_mapper_list,
    fig_size=(8, 10),
    xspacer=0.001,
    yspacer=0.1,
    ytick_font_size=18,
    title_font_size=20,
    annot_font_size=16,
    n_bars=10,
    n_plots=2,
    n_cols=2,
    show_bar_labels=True,
) -> plt.figure:
    """Plot horizontal bar chart, with labeled bar values"""
    # Get top n words per topic
    df_view_words = []
    for col in df:
        if df[col].min() < 0:
            terms = (
                df[col]
                .abs()
                .sort_values(ascending=False)
                .index.tolist()[:n_bars][::-1]
            )
        else:
            terms = (
                df[col]
                .sort_values(ascending=False)
                .index.tolist()[:n_bars][::-1]
            )
        df_view_words.append(terms)

    # Get DataFrame corresponding to top n words per topic
    df_view = pd.DataFrame(
        {
            key: sorted(np.abs(list(value.values())), reverse=False)
            for key, value in df.to_dict().items()
        }
    ).tail(n_bars)

    axes = df_view.plot(
        kind="barh",
        logx=False,
        align="center",
        legend=False,
        edgecolor="black",
        width=0.8,
        figsize=(40, 35),
        title=None,
        subplots=True,
        layout=(5, 3),
    )
    plt.subplots_adjust(
        left=0.125,  # the left side of the subplots of the figure
        right=1.0,  # the right side of the subplots of the figure
        bottom=0.1,  # the bottom of the subplots of the figure
        top=1.0,  # the top of the subplots of the figure
        wspace=0.35,  # the amount of width res. for blank space bw subplots
        hspace=0.1,  # the amount of height res. for white space bw subplots
    )
    for ax, y_tick_labels, title_text in zip(
        [a for ax in axes for a in ax], df_view_words, y_tick_mapper_list
    ):
        plt.draw()
        x_tick_labels = [item.get_text() for item in ax.get_xticklabels()]
        if x_tick_labels:
            ax.set_xticklabels(
                x_tick_labels, fontsize=SMALL_SIZE, rotation=0, ha="center"
            )
        ax.set_yticklabels(
            y_tick_labels, fontsize=SMALL_SIZE, rotation=0, ha="right"
        )
        ax.set_title(title_text, fontweight="bold", fontsize=BIGGER_SIZE)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
    fig = plt.gcf()
    return fig


def pipe_get_topics(
    X,
    pipe,
    n_topics_wanted,
    df,
    y_tick_mapper_list,
    vt_df_index=None,
    analysis_type="lsa",
    number_of_words_per_topic_to_show=5,
    show_bar_labels=True,
):
    """Pipeline-based workflow to get topics"""
    doc_topic = pipe.fit_transform(X)

    if analysis_type == "lsa":
        explained_variance_ratio = pipe.named_steps[
            analysis_type
        ].explained_variance_ratio_
        ptitle = [
            f"Topic {k+1} ({(explained_variance_ratio[k] * 100):.1f}%)"
            for k in range(n_topics_wanted)
        ]
    else:
        ptitle = [f"Topic {k+1}" for k in range(n_topics_wanted)]

    topic_word = pd.DataFrame(
        pipe.named_steps[analysis_type].components_.round(3),
        index=[f"component_{k+1}" for k in range(n_topics_wanted)],
        columns=pipe.named_steps["vectorizer"].get_feature_names(),
    )
    # print(topic_word)

    fig = plot_horiz_bar(
        topic_word.T,
        ptitle=ptitle,
        y_tick_mapper_list=y_tick_mapper_list,
        fig_size=(60, 8),
        xspacer=0.001,
        yspacer=0.3,
        ytick_font_size=18,
        title_font_size=20,
        annot_font_size=16,
        n_bars=number_of_words_per_topic_to_show,
        n_plots=topic_word.T.shape[1],
        n_cols=n_topics_wanted,
        show_bar_labels=show_bar_labels,
    )

    vt_index = vt_df_index if vt_df_index else [k[:30] for k in X]
    Vt = pd.DataFrame(
        doc_topic.round(5),
        index=vt_index,
        columns=[f"component_{k+1}" for k in range(n_topics_wanted)],
    )
    Vt.index.name = "document"
    Vt = Vt.div(Vt.sum(axis=1), axis=0)

    Vt.insert(0, "text", X)
    df_with_topics = (
        df.set_index("text")
        .merge(
            Vt.reset_index().set_index("text"),
            how="inner",
            left_index=True,
            right_index=True,
        )
        .reset_index(drop=False)
    )
    # print(df_with_topics)

    mask = df_with_topics.columns[
        df_with_topics.columns.str.contains("component")
    ]
    df_with_topics["most_popular_topic"] = df_with_topics[mask].idxmax(axis=1)
    df_with_topics = df_with_topics.sort_values(by=["year"], ascending=True)
    # df_with_topics["decade"] = df_with_topics["year"] // 10 * 10

    df_topics_by_year = (
        df_with_topics.groupby(["year"])["most_popular_topic"]
        .apply(set)
        .apply(list)
        .reset_index()
    )
    df_topics_by_year["all_most_popular_topics"] = df_topics_by_year[
        "most_popular_topic"
    ].str.join(", ")

    dict_mapper = {
        f"component_{k+1}": elem for k, elem in enumerate(y_tick_mapper_list)
    }
    value_mapper = {"most_popular_topic": dict_mapper}
    df_with_topics = df_with_topics.replace(value_mapper)
    df_with_topics = df_with_topics.rename(columns=dict_mapper)
    df_topics_by_year["all_most_popular_topics"] = df_topics_by_year[
        "all_most_popular_topics"
    ].replace(dict_mapper, regex=True)

    return df_with_topics, df_topics_by_year, topic_word, fig


def manual_get_topics(
    X,
    model,
    vectorizer,
    n_topics_wanted,
    df,
    y_tick_mapper_list,
    vt_df_index=None,
    analysis_type="lsa",
    number_of_words_per_topic_to_show=5,
    show_bar_labels=True,
) -> List:
    """Manual workflow to get topics"""
    doc_word = vectorizer.fit_transform(X)
    doc_topic = model.fit_transform(doc_word)

    if analysis_type == "lsa":
        explained_variance_ratio = model.explained_variance_ratio_
        ptitle = [
            f"Topic {k+1} ({(explained_variance_ratio[k] * 100):.1f}%)"
            for k in range(n_topics_wanted)
        ]
    else:
        ptitle = [f"Topic {k+1}" for k in range(n_topics_wanted)]

    topic_word = pd.DataFrame(
        model.components_.round(3),
        index=[f"component_{k+1}" for k in range(n_topics_wanted)],
        columns=vectorizer.get_feature_names(),
    )
    # print(topic_word)

    fig = plot_horiz_bar(
        topic_word.T,
        ptitle=ptitle,
        y_tick_mapper_list=y_tick_mapper_list,
        fig_size=(37, 6),
        xspacer=0.001,
        yspacer=0.3,
        ytick_font_size=18,
        title_font_size=20,
        annot_font_size=16,
        n_bars=number_of_words_per_topic_to_show,
        n_plots=topic_word.T.shape[1],
        n_cols=n_topics_wanted,
        show_bar_labels=show_bar_labels,
    )

    vt_index = vt_df_index if vt_df_index else [k[:30] for k in X]
    Vt = pd.DataFrame(
        doc_topic.round(5),
        index=vt_index,
        columns=[f"component_{k+1}" for k in range(n_topics_wanted)],
    )
    Vt.index.name = "document"
    Vt = Vt.div(Vt.sum(axis=1), axis=0)

    Vt.insert(0, "text", X)
    df_with_topics = (
        df.set_index("text")
        .merge(
            Vt.reset_index().set_index("text"),
            how="inner",
            left_index=True,
            right_index=True,
        )
        .reset_index(drop=False)
    )
    # print(df_with_topics)

    mask = df_with_topics.columns[
        df_with_topics.columns.str.contains("component")
    ]
    df_with_topics["most_popular_topic"] = df_with_topics[mask].idxmax(axis=1)
    df_with_topics = df_with_topics.sort_values(by=["year"], ascending=True)
    # df_with_topics["decade"] = df_with_topics["year"] // 10 * 10

    df_topics_by_year = (
        df_with_topics.groupby(["year"])["most_popular_topic"]
        .apply(set)
        .apply(list)
        .reset_index()
    )
    df_topics_by_year["all_most_popular_topics"] = df_topics_by_year[
        "most_popular_topic"
    ].str.join(", ")

    dict_mapper = {
        f"component_{k+1}": elem for k, elem in enumerate(y_tick_mapper_list)
    }
    value_mapper = {"most_popular_topic": dict_mapper}
    df_with_topics = df_with_topics.replace(value_mapper)
    df_with_topics = df_with_topics.rename(columns=dict_mapper)
    df_topics_by_year["all_most_popular_topics"] = df_topics_by_year[
        "all_most_popular_topics"
    ].replace(dict_mapper, regex=True)

    return [df_with_topics, df_topics_by_year, fig]


def get_main_topic_percentage(
    df: pd.DataFrame, group_by_col: str, quant_col_idx_start: int = 1
) -> pd.DataFrame:
    """
    Get a DataFrame of percentage of document belonging to most popular topic
    """
    # Get DataFrame of means
    df_percentages = (
        df.iloc[:, quant_col_idx_start:-1].groupby([group_by_col]).mean()
    )
    # Get DataFrame of stddevs
    df_percentages_std = (
        df.iloc[:, quant_col_idx_start:-1].groupby([group_by_col]).std()
    )
    # Append most popular topic to DataFrame of means
    df_percentages["most_popular_topic"] = df_percentages.idxmax(axis=1)
    most_pop_topic_pct = (
        (df_percentages.iloc[:, :-1].max(axis=1) * 100).round(2).astype(str)
    )
    df_percentages["most_popular_topic"] = (
        df_percentages["most_popular_topic"] + " (" + most_pop_topic_pct
    )
    # Append std of most popular topic to DataFrame of means
    for (_, row_std), (l, row_mean) in zip(
        df_percentages_std.iterrows(), df_percentages.iterrows()
    ):
        row_val = row_mean["most_popular_topic"]
        #     print(
        #         row_mean["most_popular_topic"].split(" (")[0],
        #         f"+/- {(row_std[row_val.split(' (')[0]] * 100).round(2)})",
        #     )
        row_std = f"+/- {(row_std[row_val.split(' (')[0]] * 100).round(2)})"
        df_percentages.loc[l, "most_popular_topic"] = (
            row_mean["most_popular_topic"] + row_std
        )
    return df_percentages


def altair_datetime_heatmap(
    df,
    x,
    y,
    xtitle,
    ytitle,
    tooltip,
    cmap,
    legend_title,
    color_by_col,
    yscale,
    axis_tick_font_size=12,
    axis_title_font_size=16,
    title_font_size=20,
    legend_fig_padding=10,  # default is 18
    y_axis_title_alignment="left",
    fwidth=300,
    fheight=535,
    file_path=os.path.join(
        os.path.abspath(os.getcwd()), "reports", "figures", "my_heatmap.html"
    ),
    save_to_html=False,
    sort_y=[],
    sort_x=[],
    dx=5,
    offset=10,
    plot_titleFontSize=16,
):
    """Generate a datetime heatmap with Altair"""
    # sorty = sort_y if sort_y else None
    # sortx = sort_x if sort_x else None

    base = alt.Chart(title=ytitle)
    hmap = (
        base.mark_rect(fontSize=title_font_size)
        .encode(
            alt.X(
                x,
                title="",
                sort=sort_x,
                axis=alt.Axis(
                    labelAngle=0,
                    tickOpacity=0,
                    domainWidth=0,
                    domainColor="black",
                    labelFontSize=axis_tick_font_size,
                    titleFontSize=axis_title_font_size,
                ),
            ),
            alt.Y(
                y,
                title="",
                sort=sort_y,
                axis=alt.Axis(
                    titleAngle=0,
                    titleAlign=y_axis_title_alignment,
                    tickOpacity=0,
                    domainWidth=0,
                    domainColor="black",
                    titleX=-10,
                    titleY=-10,
                    labelFontSize=axis_tick_font_size,
                    titleFontSize=axis_title_font_size,
                ),
            ),
            color=alt.Color(
                color_by_col,
                scale=alt.Scale(type=yscale, scheme=cmap),
                legend=alt.Legend(
                    title=legend_title,
                    orient="right",  # default is "right"
                    labelFontSize=axis_tick_font_size,
                    titleFontSize=axis_title_font_size,
                    offset=legend_fig_padding,
                ),
            ),
            tooltip=tooltip,
        )
        .properties(width=fwidth, height=fheight)
    )
    heatmap = (
        alt.layer(hmap, data=df)
        .configure_title(
            fontSize=plot_titleFontSize,
            anchor="start",
            dx=dx,
            offset=offset,
        )
        .configure_axisY(labelLimit=450, labelAlign="right")
    )
    if not os.path.exists(file_path) and save_to_html:
        heatmap.save(file_path)
    return heatmap


def plot_horiz_bar_gensim(
    model, id2word_dictionary, mapper_dict, fig_size=(40, 35)
) -> plt.figure:
    tableau10_colors = list(mcolors.TABLEAU_COLORS.values())
    colors = tableau10_colors + tableau10_colors[: len(mapper_dict)]
    fig = plt.figure(constrained_layout=True, figsize=fig_size)
    gs = fig.add_gridspec(5, 3)
    axs = []
    for rc in range(5):
        for cc in range(3):
            axs.append(fig.add_subplot(gs[rc, cc]))
    for topic_id, ccolor in zip(range(len(mapper_dict)), colors):
        df_topic_probs_per_topic = pd.DataFrame(
            model.get_topic_terms(topicid=topic_id, topn=10, normalize=True),
            columns=["word_id", "probability"],
        )
        df_topic_probs_per_topic.insert(
            0,
            "topic",
            mapper_dict[topic_id],
        )
        df_topic_probs_per_topic.insert(
            1,
            "word",
            df_topic_probs_per_topic["word_id"].apply(
                lambda x: id2word_dictionary[x]
            ),
        )
        df_view = df_topic_probs_per_topic.pivot(
            index="topic", columns="word", values="probability"
        ).T.sort_values(
            by=mapper_dict[topic_id],
            ascending=True,
        )
        ax = axs[topic_id]
        _ = df_view.plot(
            kind="barh",
            logx=False,
            align="center",
            legend=False,
            edgecolor="black",
            width=0.8,
            title=None,
            color=ccolor,
            ax=ax,
        )
        ax.set_yticklabels(
            df_view.index.tolist(), fontsize=SMALL_SIZE, rotation=0, ha="right"
        )
        ax.set_ylabel(None)
        ax.set_xticklabels([])
        ax.set_title(
            mapper_dict[topic_id],
            fontweight="bold",
            fontsize=BIGGER_SIZE,
        )
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
    return fig


def get_top_words_per_topic(
    model_fitted,
    vectorizer_fitted,
    mapper_dict: dict,
    num_top_words: int = 10,
    num_topics: int = 15,
    rename_col_names: bool = False,
) -> List:
    topic_word_best = pd.DataFrame(
        model_fitted.components_,
        index=[f"component_{k+1}" for k in range(num_topics)],
        columns=vectorizer_fitted.get_feature_names(),
    )
    top_words = {}
    top_vals = {}
    for topic, words_ in topic_word_best.T.items():
        top10 = words_.nlargest(num_top_words).index
        vals = words_.loc[top10].values
        top_vals[topic] = vals
        top_words[topic] = top10.tolist()
    df_top_words_per_topic = pd.DataFrame(top_words)
    df_top_vals_per_topic = pd.DataFrame(top_vals)
    if rename_col_names:
        df_top_words_per_topic = df_top_words_per_topic.rename(
            columns=mapper_dict
        )
        df_top_vals_per_topic = df_top_vals_per_topic.rename(
            columns=mapper_dict
        )
    return [topic_word_best, df_top_words_per_topic, df_top_vals_per_topic]


def get_docs_with_topics_v2(
    corpus: list, model_transformed, mapper_dict: dict, df: pd.DataFrame
) -> pd.DataFrame:
    corpus_label = [t[:30] + "..." for t in corpus]
    df_with_topics = pd.DataFrame(
        data=model_transformed,
        columns=mapper_dict.values(),
        index=corpus_label,
    )
    df_with_topics = df_with_topics.div(df_with_topics.sum(axis=1), axis=0)
    df_with_topics["most_popular_topic"] = df_with_topics.idxmax(axis=1)
    df_with_topics.insert(0, "text", df["text"].to_numpy())
    df_with_topics.insert(1, "year", df["year"].to_numpy())
    df_with_topics.insert(2, "document", range(len(df_with_topics)))
    return df_with_topics


def altair_boxplot_sorted(
    df,
    xvar="X",
    yvar="Y",
    sortvar="median(Z)",
    ptitle="Title here",
    labelFontSize=12,
    titleFontSize=12,
    plot_titleFontSize=16,
    dx=30,
    offset=-5,
    x_tick_label_angle=-45,
    horiz_bar_chart=False,
    axis_range=[0, 1],
    fig_size=(450, 250),
):
    sortvar = yvar if not sortvar else sortvar
    df2 = pd.DataFrame({col: vals[yvar] for col, vals in df.groupby(xvar)})
    base = alt.Chart(df, title=ptitle).mark_boxplot(
        color="#4287f5"
    )  # #3b5b92 or #4287f5
    if horiz_bar_chart:
        # print("Horizontal")

        if "median" in sortvar:
            boxes_sorted_by_median = (
                df2.median(axis=0).sort_values(ascending=False).index.tolist()
            )
        else:
            boxes_sorted_by_median = np.sort(df[xvar].unique())[::-1].tolist()
        chart = base.encode(
            x=alt.X(
                f"{sortvar}:Q",
                title="",
                scale=alt.Scale(domain=(axis_range[0], axis_range[1])),
            ),
            y=alt.Y(f"{xvar}:N", title=xvar, sort=boxes_sorted_by_median),
        )
        x_tick_label_angle = 0
        chart = chart.configure_axisY(
            labelLimit=450, labelAlign="right", titleAngle=0, titleFontSize=0
        )
    else:
        # print("Vertical")
        if "median" in sortvar:
            boxes_sorted_by_median = (
                df2.median(axis=0).sort_values(ascending=True).index.tolist()
            )
        else:
            boxes_sorted_by_median = df[xvar].unique().tolist()
        chart = base.encode(
            x=alt.X(f"{xvar}:N", title="", sort=boxes_sorted_by_median),
            y=alt.Y(
                f"{sortvar}:Q",
                title=xvar,
                scale=alt.Scale(domain=(axis_range[0], axis_range[1])),
            ),
        )
        chart = chart.configure_axisY(titleFontSize=0, titleAngle=0)
    chart = (
        chart.properties(width=fig_size[0], height=fig_size[1])
        .configure_axis(
            labelFontSize=labelFontSize, titleFontSize=titleFontSize
        )
        .configure_axisX(labelAngle=x_tick_label_angle)
        .configure_title(
            fontSize=plot_titleFontSize, anchor="start", dx=dx, offset=offset
        )
    )
    return chart


def boxplot_sorted(
    df,
    by,
    column,
    ptitle,
    ha="left",
    font_size=12,
    x_tick_angle=0,
    sort_by_median=True,
    vert=True,
    fig_size=(12, 4),
):
    _, ax = plt.subplots(figsize=fig_size)
    props = dict(boxes="#4287f5", whiskers="k", medians="cyan", caps="k")
    df2 = pd.DataFrame({col: vals[column] for col, vals in df.groupby(by)})
    if sort_by_median:
        meds = df2.median().sort_values(ascending=True)
        df2[meds.index].boxplot(
            rot=x_tick_angle,
            fontsize=font_size,
            vert=vert,
            color=props,
            patch_artist=True,
            ax=ax,
        )
    else:
        df2.boxplot(
            rot=x_tick_angle,
            fontsize=font_size,
            vert=vert,
            color=props,
            patch_artist=True,
            ax=ax,
        )
    if vert:
        ax.set_xticklabels(ax.get_xticklabels(), ha=ha)
    ax.set_title(
        ptitle,
        fontsize=14,
        loc="left",
        fontweight="bold",
    )


def altair_plot_grid_by_column(
    df,
    xvar,
    yvar,
    col2grid,
    space_between_plots=5,
    row_size=3,
    labelFontSize=14,
    titleFontSize=14,
    fig_size=(100, 200),
):
    columns = []
    chunks = (df[col2grid].nunique() - 1) // row_size + 1
    for i in range(chunks):
        # print(i * row_size, (i + 1) * row_size)
        rows = []
        row_mul_start = i * row_size
        row_mul_stop = (i + 1) * row_size
        for y in df[col2grid].unique()[row_mul_start:row_mul_stop]:
            row_chart = (
                alt.Chart(df[df[col2grid] == y], title=str(y))
                .mark_bar()
                .encode(
                    x=alt.X(f"{xvar}:Q", title=""),
                    y=alt.Y(f"{yvar}:N", title="", sort="-x"),
                    color=alt.Color(
                        f"{col2grid}:N",
                        scale=alt.Scale(scheme="tableau20"),
                        legend=None,
                    ),
                )
                .properties(width=fig_size[0], height=fig_size[1])
            )
            rows.append(row_chart)
        col_chart = alt.vconcat(*rows)  # .resolve_scale(color="independent")
        columns.append(col_chart)
    combo = (
        alt.hconcat(*columns).configure_concat(spacing=space_between_plots)
        # .resolve_scale(color="independent")
        .configure_axis(
            labelFontSize=labelFontSize, titleFontSize=titleFontSize
        )
    )
    return combo


def altair_plot_line_chart(
    df,
    xvar,
    yvar,
    ptitle,
    labelFontSize=12,
    titleFontSize=14,
    plot_titleFontSize=16,
    linewidth=3,
    dx=0,
    offset=5,
    x_tick_label_angle=-45,
    marker_size=200,
    y_axis_range=[0, 1],
    fig_size=(750, 250),
):
    chart = (
        alt.Chart(df, title=ptitle)
        .mark_line(point=True, strokeWidth=linewidth)
        .encode(
            x=alt.X(f"{xvar}:N", title=""),
            y=alt.Y(
                f"{yvar}:Q",
                title="",
                scale=alt.Scale(domain=(y_axis_range[0], y_axis_range[1])),
            ),
        )
        .properties(width=fig_size[0], height=fig_size[1])
        .configure_axis(
            labelFontSize=labelFontSize, titleFontSize=titleFontSize
        )
        .configure_axisX(labelAngle=x_tick_label_angle)
        .configure_point(size=marker_size)
        .configure_mark(color="#4287f5")
        .configure_title(
            fontSize=plot_titleFontSize,
            anchor="start",
            dx=dx,
            offset=offset,
        )
    )
    return chart


def plot_single_point_annotated_line_chart(
    df, ptitle, y_annot, fig_size=(12, 4)
):
    _, ax = plt.subplots(figsize=fig_size)
    df.plot(color="#4287f5", ax=ax)
    ax.get_legend().remove()
    ax.set_xlabel(None)
    ax.annotate(
        f"{df.idxmax()[0]}={y_annot}",
        xy=(df.idxmax()[0], y_annot),
        xytext=(df.idxmax()[0], y_annot),
        textcoords="offset points",
        arrowprops=dict(facecolor="black", shrink=0.05),
        bbox=dict(boxstyle="round,pad=0.3", fc=(0.8, 0.9, 0.9), ec="b", lw=2),
        fontsize=16,
    )
    ax.set_title(
        ptitle,
        fontsize=14,
        loc="left",
        fontweight="bold",
    )
    ax.grid(which="both", axis="both", color="lightgrey")


def altair_plot_bar_chart_value_counts(
    df,
    ptitle,
    xvar,
    yvar,
    labelFontSize=12,
    titleFontSize=12,
    plot_titleFontSize=16,
    dx=30,
    offset=-5,
    x_tick_label_angle=-45,
    horiz_bar_chart=False,
    horiz_label_limit=180,
    fig_size=(650, 250),
):
    chart = (
        alt.Chart(
            df,
            title=ptitle,
        )
        .mark_bar()
        .encode(x=alt.X(f"{xvar}", title=""), y=alt.Y(f"{yvar}", title=""))
        .configure_mark(color="#4287f5")
        .configure_axis(
            labelFontSize=labelFontSize, titleFontSize=titleFontSize
        )
        .configure_axisX(labelAngle=x_tick_label_angle)
        .properties(width=fig_size[0], height=fig_size[1])
        .configure_title(
            fontSize=plot_titleFontSize,
            anchor="start",
            dx=dx,
            offset=offset,
        )
    )
    if horiz_bar_chart:
        x_tick_label_angle = 0
        chart = chart.configure_axisX(labelAngle=0).configure_axisY(
            labelLimit=horiz_label_limit, labelAlign="right"
        )
    return chart


def altair_plot_horiz_bar_chart(
    data,
    ptitle="Most Popular Topic by Year",
    xvar="url",
    yvar="year",
    xtitle="Occurrences",
    labelFontSize=12,
    titleFontSize=14,
    plot_titleFontSize=16,
    text_var="abc",
    tooltip=["abc", "xyz"],
    dx=35,
    offset=0,
    horiz_label_limit=0,
    sort_y="-y",
    fig_size=(600, 450),
):
    bars = (
        alt.Chart(data, title=ptitle)
        .mark_bar()
        .encode(
            x=alt.X(f"{xvar}:Q", title=xtitle),
            y=alt.Y(f"{yvar}:N", title="", sort=sort_y),
            color=alt.Color(f"{text_var}:N", legend=None),
            tooltip=tooltip,
        )
        .properties(height=fig_size[0], width=fig_size[1])
        .configure_axis(
            labelFontSize=labelFontSize,
            titleFontSize=titleFontSize,
        )
        .configure_title(
            fontSize=plot_titleFontSize, anchor="start", dx=dx, offset=offset
        )
    )
    if horiz_label_limit > 0:
        bars = bars.configure_axisY(
            labelLimit=horiz_label_limit, labelAlign="right"
        )
    return bars


def altair_plot_histogram_grid_by_column(
    data,
    xvar,
    yvar,
    col2grid,
    pairs_of_lists,
    space_between_plots=5,
    row_size=3,
    labelFontSize=14,
    titleFontSize=14,
    fig_size=(100, 200),
):
    columns = []
    chunks = (len(pairs_of_lists) - 1) // row_size + 1
    for i in range(chunks):
        # print(i * row_size, (i + 1) * row_size)
        rows = []
        row_mul_start = i * row_size
        row_mul_stop = (i + 1) * row_size
        for y in pairs_of_lists[row_mul_start:row_mul_stop]:
            # print(i, y)
            row_chart = (
                alt.Chart(data[data[col2grid].str.contains("|".join(y))])
                .mark_bar()
                .encode(
                    alt.X(f"{xvar}:Q", title="", bin=alt.Bin(maxbins=20)),
                    alt.Y(f"{yvar}", title=""),
                    alt.Color(
                        f"{col2grid}:N",
                        legend=None,
                    ),
                    tooltip=[col2grid, yvar],
                )
                .properties(width=200, height=200)
            )
            rows.append(row_chart)
        column_chart = alt.vconcat(*rows).resolve_scale(color="independent")
        columns.append(column_chart)
    combo = (
        alt.hconcat(*columns).configure_concat(spacing=space_between_plots)
        # .resolve_scale(color="independent")
        .configure_axis(
            labelFontSize=labelFontSize, titleFontSize=titleFontSize
        )
    )
    return combo


def altair_plot_triangular_heatmap(
    data,
    ptitle,
    xvar,
    yvar,
    zvar,
    xtitle,
    ytitle,
    tooltip,
    axis_tick_font_size=14,
    axis_title_font_size=16,
    plot_titleFontSize=12,
    dx=0,
    offset=0,
    show_triangle="upper",
    fig_size=(600, 600),
):
    chart = (
        alt.Chart(
            data,
            title=ptitle,
        )
        .mark_rect()
        .encode(
            x=alt.X(
                f"{xvar}:O",
                title=xtitle,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(f"{yvar}:O", title=ytitle),
            color=alt.Color(
                f"{zvar}:Q", scale=alt.Scale(scheme="yelloworangered")
            ),
            tooltip=tooltip,
        )
        .properties(width=fig_size[1], height=fig_size[1])
        .configure_title(
            fontSize=plot_titleFontSize,
            anchor="start",
            dx=dx,
            offset=offset,
        )
        .configure_axis(
            labelFontSize=axis_tick_font_size,
            titleFontSize=axis_title_font_size,
        )
    )
    if show_triangle == "lower":
        chart = chart.transform_filter(f"datum.{xvar} <= datum.{yvar}")
    elif show_triangle == "upper":
        chart = chart.transform_filter(f"datum.{xvar} >= datum.{yvar}")
    return chart
