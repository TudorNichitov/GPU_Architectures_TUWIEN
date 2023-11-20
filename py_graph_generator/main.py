import argparse
import os.path

import networkx as nx
import numpy as np

# list of graph generators to choose from
TYPES = ["erdos", "scale", "path", "ring", "complete", "tree", "empty", "custom"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n", type=int, default=10, dest='n', action="store",
                        help="Number of nodes")
    parser.add_argument("-t", "--type", type=str, default="erdos", dest='type', action="store",
                        help=f"Graph type; choose from {TYPES}")
    parser.add_argument("-s", "--seed", type=int, default=42, dest='seed', action="store",
                        help="Indicator of random number generation state.")
    parser.add_argument("-d", "--out", type=str, default="../data", dest='out', action="store",
                        help="Directory to write graph files")
    parser.add_argument("-x", "--samples", type=int, default=1, dest='samples', action="store",
                        help="Number X determines how many samples are to be exported. ")

    # arg only used for erdos renyi graph generator
    parser.add_argument("-p", "--prob", type=float, dest='prob', action="store", default=0.1,
                        help="[Erdos]: edge probability")

    # args only used for scale free graph generator
    # (default values model the WWW, where an edge represents a link from one page to another)
    parser.add_argument("-a", "--alpha", type=float, default=0.41, dest='alpha', action="store",
                        help=f"[Scale]: Probability for adding an edge from an new node to an existing node")
    parser.add_argument("-b", "--beta", type=float, default=0.54, dest='beta', action="store",
                        help=f"[Scale]: Probability for adding an edge between two existing nodes. ")
    parser.add_argument("-g", "--gamma", type=float, default=0.05, dest='gamma', action="store",
                        help=f"[Scale]: Probability for adding an edge from an existing node to new node")
    parser.add_argument("-din", "--delta_in", type=float, default=0.2, dest='delta_in', action="store",
                        help="[Scale]: Bias for choosing nodes from in-degree distribution.")
    parser.add_argument("-dout", "--delta_out", type=float, default=0, dest='delta_out', action="store",
                        help="[Scale]: Bias for choosing nodes from out-degree distribution.")

    # args only used for custom graph (as defined in assignment)
    parser.add_argument("-e", "--density", type=float, default=0.2, dest='density', action="store",
                        help="[Custom]: Targeted density of the graph.")
    return parser.parse_args()


def create_dirs(path: str):
    """ make dirs """
    try:
        out_dir = os.path.abspath(path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_dir = os.path.abspath(out_dir)
    except Exception as ex:
        print(f"Fatal Error: create_out_dirs(...) throws: {str(ex)} ")
        raise ex
    return out_dir


def write_row(f, row: tuple):
    assert not f.closed, f"File must be opened - can not write to file!"
    f.write(' '.join(str(val) for val in row))
    f.write('\n')


def export_graph(g: nx.Graph, filename):
    """writes graph to file"""
    with open(filename, 'w') as f:
        header_row = ("H", len(g), len(g.edges), 0)
        write_row(f, header_row)
        for u, v in g.edges():
            edge_row = ("E", u, v, 1)
            write_row(f, edge_row)
    print(filename)


def create_scale_free_graph(args, seed):
    """
    Returns a scale free random networkx DiGraph (in/out degree follow the power law)
    for details see:
        [1] https://networkx.org/documentation/stable/reference/generated/networkx.generators.directed.scale_free_graph.html
        [2] B. Bollobás, C. Borgs, J. Chayes, and O. Riordan, Directed scale-free graphs,
        Proceedings of the fourteenth annual ACM-SIAM Symposium on Discrete Algorithms, 132–139, 2003.
    """
    assert args.alpha + args.beta + args.gamma == 1.0, "alpha, beta and gamma must sum up to 1"
    g = nx.DiGraph(
        nx.scale_free_graph(args.n, alpha=args.alpha, beta=args.beta, gamma=args.gamma, delta_in=args.delta_in,
                            delta_out=args.delta_out, seed=seed))
    name = f"scale_n_{args.n}_a_{args.alpha}_b_{args.beta}_g_{args.gamma}" \
           f"_din_{args.delta_in}_dout_{args.delta_out}_s_{seed}"
    return g, name


def create_custom_rnd_graph(args, seed):
    # set seed
    np.random.seed(args.seed)

    # get random adj matrix with desired density
    required_edges = int(args.density * args.n ** 2)
    links = np.random.randint(0, args.n, (required_edges, 2), dtype=int)
    adj_matrix = np.zeros((args.n, args.n))
    for i, j in links:
        adj_matrix[i][j] = 1

    # create nx graph from adj
    g = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    name = f"custom_n_{args.n}_density_{args.density}_s_{seed}"
    return g, name


def main(args):
    """Creates the chosen type of graph and exports it to file."""
    name = f"{args.type}_n_{args.n}"  # name of graph used for filename
    datadir = create_dirs(args.out)
    for x in range(args.samples):
        seed = args.seed + x
        if args.type == "erdos":
            g = nx.erdos_renyi_graph(n=args.n, seed=seed, directed=True, p=args.prob)
            name = f"{args.type}_n_{args.n}_p_{args.prob}_s_{seed}"
        elif args.type == "scale":
            g, name = create_scale_free_graph(args, seed)
        elif args.type == "path":
            g = nx.path_graph(n=args.n, create_using=nx.DiGraph)
        elif args.type == "ring":
            g = nx.path_graph(n=args.n, create_using=nx.DiGraph)
            g.add_edge(args.n - 1, 0)
        elif args.type == "complete":
            g = nx.complete_graph(n=args.n, create_using=nx.DiGraph)
        elif args.type == "tree":
            g = nx.random_tree(n=args.n, seed=seed, create_using=nx.DiGraph)
            name = f"tree_n_{args.n}_s_{seed}"
        elif args.type == "empty":
            g = nx.empty_graph(n=args.n, create_using=nx.DiGraph)
        elif args.type == "custom":
            g, name = create_custom_rnd_graph(args, seed)
        else:
            raise NotImplementedError(f"Graph generator not implemented; Choose from {TYPES}.")

        # write graph to file
        graph_file_name = os.path.abspath(os.path.join(datadir, f"graph_{name.replace('.', '')}"))
        export_graph(g, graph_file_name)


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
