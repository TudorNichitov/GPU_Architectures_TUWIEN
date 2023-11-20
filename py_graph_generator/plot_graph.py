import argparse
import os.path

import matplotlib.pyplot as plt
import networkx as nx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default=10, dest='file_name', action="store",
                        help="Filename graph")
    return parser.parse_args()


def draw_graph(g, plot_file_name):
    """Plots graph in kamada kawai layout (for report) """
    pos = nx.kamada_kawai_layout(g)
    nx.draw_networkx(g, pos=pos, with_labels=True, font_color="white", font_size=8)
    plt.savefig(plot_file_name)
    plt.close()
    print(plot_file_name)


def load_graph(file_name) -> nx.DiGraph:
    """Loads graph from file"""
    g = nx.DiGraph()
    with open(file_name, 'r') as f:
        rows = f.readlines()
    for row in rows:
        line_type, u, v, weight = row.split(' ')
        if line_type == "E":
            g.add_edge(u, v)
    return g


def main(graph_file_name):
    # load graph from file
    graph_file_name = os.path.abspath(graph_file_name)
    g = load_graph(graph_file_name)

    # plot graph as pdf
    plot_file_name = f"{graph_file_name.replace('graph', 'plot')}.pdf"
    draw_graph(g, plot_file_name)


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args.file_name)
