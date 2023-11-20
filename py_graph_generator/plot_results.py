import argparse
import os.path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

rc_fonts = {
    "font.family": "serif",
    "font.size": 20,
    'figure.figsize': (5, 3),
    "text.usetex": True,
    'text.latex.preamble':
        r"""
        \usepackage{libertine}
        \usepackage[libertine]{newtxmath}
        """
}

mpl.rcParams.update(rc_fonts)

__font_4 = 4
__font_6 = 6
__font_7 = 7
__font_8 = 8
__font_20 = 20
__font_24 = 24
__font_36 = 36
__font_44 = 44
__font_16 = 16
__font_12 = 12
__font_30 = 30
__font_26 = 26

OUT_DIR = "../out/plots/"


def scatterplot(plot_file: str, x: str, y: str, hue: str, df: pd.DataFrame, title: str = None):
    """outputs boxplot for a single problem instance"""

    # df is sorted by solverName
    # x := df colum name n (graph sizes)
    # y := df colum name eTotal (elapsed milliseconds total)
    # hue := df column name solverName (e.g. cpu1, kernel1, ..., kernel9)

    print(plot_file)
    x_data = df[x]
    # need to apply logarithm, otherwise we run into troubles
    y_data = np.log(df[y])
    solver_name = df[hue]
    fig, ax = plt.subplots(figsize=(__font_8, __font_8))
    ax.tick_params(labelsize=__font_24)
    ax.xaxis.get_offset_text().set_fontsize(__font_24)

    color_palette = sns.color_palette("tab10")
    sns.set_palette(palette=color_palette)
    sns.set_style(style="ticks")

    scatter_plot = sns.scatterplot(x_data, y_data, solver_name, style=solver_name, s=100)
    scatter_plot.set(xlabel="$|V|$")
    scatter_plot.set(ylabel="Time elapsed in [ms]")
    if title:
        scatter_plot.set(title=title)
    plt.tight_layout()

    plt.savefig(plot_file)
    # plt.show()
    plt.close()


def lineplot(plot_file: str, x: str, y: str, hue: str, df: pd.DataFrame, title: str = None):
    """outputs shaded line plot for a single problem instance"""

    print(plot_file)
    x_data = df[x]
    y_data = df[y]
    solver_name = df[hue]
    fig, ax = plt.subplots(figsize=(__font_8, __font_8))
    ax.tick_params(labelsize=__font_24)

    color_palette = sns.color_palette("tab10")
    sns.set_palette(palette=color_palette)
    sns.set_style(style="ticks")

    line = sns.lineplot(x=x_data, y=y_data, hue=solver_name, style=solver_name, linewidth=3)

    line.set_xlabel("$|V|$", fontsize=__font_24)
    line.set_ylabel(y, fontsize=__font_24)
    line.set(ylabel="Time elapsed in [ms]")
    if title:
        line.set(title=title)
    plt.tight_layout()

    plt.savefig(plot_file)
    # plt.show()
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, dest='file_name', action="store",
                        help="Filename out.csv")
    parser.add_argument("-d", "--dir", type=str, default=OUT_DIR, dest='out_dir', action="store",
                        help="Directory containing out.csv")
    return parser.parse_args()


def parse_graph_file_name(row) -> dict:
    """extracts key values from graph file name"""
    filename = row["graphFileName"]
    values = filename.split('_')
    if filename.startswith("graph_custom"):
        kv = {
            "type": values[1],
            "n": int(values[3]),
            "density": float("0." + values[5][1:]),
            "seed": int(values[7])
        }
    elif filename.startswith("graph_erdos"):
        kv = {
            "type": values[1],
            "n": int(values[3]),
            "p": float("0." + values[5][1:]),
            "seed": int(values[7])
        }
    elif filename.startswith("graph_scale"):
        kv = {
            "type": values[1],
            "n": int(values[3]),
            "alpha": float("0." + values[5][1:]),
            "beta": float("0." + values[7][1:]),
            "gamma": float("0." + values[9][1:]),
            "delta_in": float("0." + values[11][1:]),
            "delta_out": float("0." + values[13][1:]),
            "seed": int(values[15]),
        }
    elif filename.startswith("graph_tree"):
        kv = {
            "type": values[1],
            "n": int(values[3]),
            "seed": int(values[5])
        }
    else:
        raise NotImplementedError("unknown graph filename")
    return kv


def add_columns_to_df(df) -> pd.DataFrame:
    """adds columns to df extracted from graph file name"""
    df_new = df.copy(deep=True)

    # add additional columns to new df
    additional_columns = ["type", "n", "p", "density", "alpha", "beta", "gamma", "delta_in", "delta_out", "seed"]
    for col in additional_columns:
        df_new[col] = -1

    # extract values from graph file name and add to column
    for index, row in df.iterrows():
        kv = parse_graph_file_name(row)
        for k, v in kv.items():
            df_new.loc[index, k] = v
    return df_new


def get_plot_file_name(out_dir, plot_type, graph_type, p, density, alpha, beta, gamma, delta_in, delta_out):
    """returns plot file name for problem instance"""
    if graph_type == "custom":
        file_name = f"custom_density-{str(density).replace('.', '-')}"
    elif graph_type == "erdos":
        file_name = f"erdos-renyi_p-{str(p).replace('.', '-')}"
    elif graph_type == "scale":
        file_name = f"scale-free_a-{str(alpha).replace('.', '-')}_b-{str(beta).replace('.', '-')}_" \
                    f"g-{str(gamma).replace('.', '-')}_din-{str(delta_in).replace('.', '-')}_" \
                    f"dout-{str(delta_out).replace('.', '-')}"
    elif graph_type == "tree":
        file_name = f"random_tree"
    else:
        raise NotImplementedError("graph type unknown")
    return os.path.join(out_dir, f"{plot_type}_{file_name}.png")


def filter_solver(solver_list: list, df: pd.DataFrame):
    df_new = df.copy(deep=True)
    df_new = df_new[df_new["solverName"].isin(solver_list)]
    return df_new


def create_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


def get_subtitle(graph_type, p, density, alpha, beta, gamma, delta_in, delta_out):
    """returns subtitle for problem instance"""
    if graph_type == "custom":
        subtitle = f"Custom Graph with density={density}"
    elif graph_type == "erdos":
        subtitle = f"Erd\\\"os Renyí Graph with p={p}"
    elif graph_type == "scale":
        subtitle = f"Scale-free Graph with $\\alpha={alpha}$, $\\beta={beta}$, $\\gamma={gamma}$, " \
                   f"$\\delta_{{in}}={delta_in}$, $\\delta_{{out}}={delta_out}$"
    elif graph_type == "tree":
        subtitle = f"Random Tree Graph"
    else:
        raise NotImplementedError("graph type unknown")
    return f"\\tiny{{\\\\}}\\Large{{{subtitle}}}\\tiny{{ \\\\ \\_}}"


def main(csv_file, out_dir):
    create_dir(out_dir)
    df = add_columns_to_df(pd.read_csv(csv_file, header=0, sep=","))

    # group df as problem instances (each problem instance is defined by graph type and its parameters)
    # each group contains the results for all tested kernels and all graph sizes
    group_by = ["type", "p", "density", "alpha", "beta", "gamma", "delta_in", "delta_out"]
    df_problem_instances = df.groupby(by=group_by)
    for group_keys, df_group in df_problem_instances:
        graph_type, p, density, alpha, beta, gamma, delta_in, delta_out = group_keys
        p_file1 = get_plot_file_name(out_dir, "scatter", graph_type, p, density, alpha, beta, gamma, delta_in,
                                     delta_out)
        p_file2 = get_plot_file_name(out_dir, "line", graph_type, p, density, alpha, beta, gamma, delta_in,
                                     delta_out)

        df_group.sort_values(by="solverName", ignore_index=True, inplace=True)

        # ["cpu1", "kernel1", "kernel2", "kernel3", "kernel4", "kernel5", "kernel6", "kernel7", "kernel8", "kernel9"]
        # df_group = filter_solver(["kernel7", "kernel8", "kernel9"], df_group)

        subtitle = get_subtitle(graph_type, p, density, alpha, beta, gamma, delta_in, delta_out)

        title = f"Total Execution Time (smaller instances)\\\\{subtitle}"
        scatterplot(plot_file=p_file1, x="n", y="eTotal", hue="solverName", df=df_group, title=title)
        lineplot(plot_file=p_file2, x="n", y="eTotal", hue="solverName", df=df_group, title=title)


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args.file_name, parsed_args.out_dir)
