#!/usr/bin/env python3

from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# from sklearn.model_selection import train_test_split

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
):
    """Plot horizontal bar chart, with labeled bar values"""
    # Get top n words per topic
    df_view_words = [
        df[col].sort_values(ascending=False).index.tolist()[:n_bars][::-1]
        for col in df
    ]

    # Get DataFrame corresponding to top n words per topic
    df_view = pd.DataFrame(
        {
            key: sorted(value.values(), reverse=False)
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
    # X_train, X_test = train_test_split(X, test_size=0.33, random_state=42)
    # _ = pipe.fit_transform(X_train)
    # doc_topic = pipe.transform(X_test)
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
):
    """Pipeline-based workflow to get topics"""
    # X_train, X_test = train_test_split(X, test_size=0.33, random_state=42)
    # doc_word_train = vectorizer.fit_transform(X_train)
    # doc_word = vectorizer.transform(X_test)
    # _ = model.fit_transform(doc_word_train)
    # doc_topic = model.fit_transform(doc_word)

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

    return df_with_topics, df_topics_by_year, fig


def get_main_topic_percentage(df, group_by_col, quant_col_idx_start=1):
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
    file_path=Path().cwd() / "reports" / "figures" / "my_heatmap.html",
    save_to_html=False,
    sort_y=[],
    sort_x=[],
):
    """Generate a datetime heatmap with Altair"""
    # sorty = sort_y if sort_y else None
    # sortx = sort_x if sort_x else None

    base = alt.Chart()
    hmap = (
        base.mark_rect(fontSize=title_font_size)
        .encode(
            alt.X(
                x,
                title=xtitle,
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
                title=ytitle,
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
    heatmap = alt.layer(hmap, data=df)
    if not file_path.is_file() and save_to_html:
        heatmap.save(str(file_path))
    return heatmap
