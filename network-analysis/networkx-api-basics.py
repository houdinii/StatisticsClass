import matplotlib.pyplot as plt  # Used to draw network plot
import networkx as nx  # NetworkX is typically imported as nx
from networkx import number_of_selfloops
import datetime
from datetime import date

# Data from DataCamp iPython
T_sub_nodes = {
    1: {'category': 'I', 'occupation': 'politician'},
    3: {'category': 'D', 'occupation': 'celebrity'},
    4: {'category': 'I', 'occupation': 'politician'},
    5: {'category': 'I', 'occupation': 'scientist'},
    6: {'category': 'D', 'occupation': 'politician'},
    7: {'category': 'I', 'occupation': 'politician'},
    8: {'category': 'I', 'occupation': 'celebrity'},
    9: {'category': 'D', 'occupation': 'scientist'},
    10: {'category': 'D', 'occupation': 'celebrity'},
    11: {'category': 'I', 'occupation': 'celebrity'},
    12: {'category': 'I', 'occupation': 'celebrity'},
    13: {'category': 'P', 'occupation': 'scientist'},
    14: {'category': 'D', 'occupation': 'celebrity'},
    15: {'category': 'P', 'occupation': 'scientist'},
    16: {'category': 'P', 'occupation': 'politician'},
    17: {'category': 'I', 'occupation': 'scientist'},
    18: {'category': 'I', 'occupation': 'celebrity'},
    19: {'category': 'I', 'occupation': 'scientist'},
    20: {'category': 'P', 'occupation': 'scientist'},
    21: {'category': 'I', 'occupation': 'celebrity'},
    22: {'category': 'D', 'occupation': 'scientist'},
    23: {'category': 'D', 'occupation': 'scientist'},
    24: {'category': 'P', 'occupation': 'politician'},
    25: {'category': 'I', 'occupation': 'celebrity'},
    26: {'category': 'P', 'occupation': 'celebrity'},
    27: {'category': 'D', 'occupation': 'scientist'},
    28: {'category': 'P', 'occupation': 'celebrity'},
    29: {'category': 'I', 'occupation': 'celebrity'},
    30: {'category': 'P', 'occupation': 'scientist'},
    31: {'category': 'D', 'occupation': 'scientist'},
    32: {'category': 'P', 'occupation': 'politician'},
    33: {'category': 'I', 'occupation': 'politician'},
    34: {'category': 'D', 'occupation': 'celebrity'},
    35: {'category': 'P', 'occupation': 'scientist'},
    36: {'category': 'D', 'occupation': 'scientist'},
    37: {'category': 'I', 'occupation': 'scientist'},
    38: {'category': 'P', 'occupation': 'celebrity'},
    39: {'category': 'D', 'occupation': 'celebrity'},
    40: {'category': 'I', 'occupation': 'celebrity'},
    41: {'category': 'I', 'occupation': 'celebrity'},
    42: {'category': 'P', 'occupation': 'scientist'},
    43: {'category': 'I', 'occupation': 'celebrity'},
    44: {'category': 'I', 'occupation': 'politician'},
    45: {'category': 'D', 'occupation': 'scientist'},
    46: {'category': 'I', 'occupation': 'politician'},
    47: {'category': 'I', 'occupation': 'celebrity'},
    48: {'category': 'P', 'occupation': 'celebrity'},
    49: {'category': 'P', 'occupation': 'politician'}}
T_sub_edges = [
    (1, 3, {'date': datetime.date(2012, 11, 16)}), 
    (1, 4, {'date': datetime.date(2013, 6, 7)}), 
    (1, 5, {'date': datetime.date(2009, 7, 27)}), 
    (1, 6, {'date': datetime.date(2014, 12, 18)}), 
    (1, 7, {'date': datetime.date(2010, 10, 18)}),
    (1, 8, {'date': datetime.date(2012, 4, 18)}),
    (1, 9, {'date': datetime.date(2007, 10, 14)}),
    (1, 10, {'date': datetime.date(2012, 9, 8)}),
    (1, 11, {'date': datetime.date(2010, 1, 6)}),
    (1, 12, {'date': datetime.date(2012, 12, 27)}),
    (1, 13, {'date': datetime.date(2008, 12, 18)}),
    (1, 14, {'date': datetime.date(2014, 5, 25)}),
    (1, 15, {'date': datetime.date(2009, 11, 12)}),
    (1, 16, {'date': datetime.date(2008, 8, 6)}),
    (1, 17, {'date': datetime.date(2007, 8, 11)}),
    (1, 18, {'date': datetime.date(2009, 10, 7)}),
    (1, 19, {'date': datetime.date(2008, 7, 24)}),
    (1, 20, {'date': datetime.date(2013, 11, 18)}),
    (1, 21, {'date': datetime.date(2011, 3, 28)}),
    (1, 22, {'date': datetime.date(2013, 3, 4)}),
    (1, 23, {'date': datetime.date(2012, 4, 20)}),
    (1, 24, {'date': datetime.date(2009, 6, 6)}),
    (1, 25, {'date': datetime.date(2013, 6, 18)}),
    (1, 26, {'date': datetime.date(2014, 11, 20)}),
    (1, 27, {'date': datetime.date(2007, 4, 28)}),
    (1, 28, {'date': datetime.date(2007, 2, 25)}),
    (1, 29, {'date': datetime.date(2014, 1, 23)}),
    (1, 30, {'date': datetime.date(2007, 10, 9)}),
    (1, 31, {'date': datetime.date(2009, 2, 17)}),
    (1, 32, {'date': datetime.date(2009, 10, 14)}),
    (1, 33, {'date': datetime.date(2010, 5, 19)}),
    (1, 34, {'date': datetime.date(2009, 12, 21)}),
    (1, 35, {'date': datetime.date(2014, 11, 16)}),
    (1, 36, {'date': datetime.date(2010, 2, 25)}),
    (1, 37, {'date': datetime.date(2010, 9, 23)}),
    (1, 38, {'date': datetime.date(2007, 4, 28)}),
    (1, 39, {'date': datetime.date(2008, 2, 26)}),
    (1, 40, {'date': datetime.date(2010, 5, 15)}),
    (1, 41, {'date': datetime.date(2009, 8, 12)}),
    (1, 42, {'date': datetime.date(2013, 1, 22)}),
    (1, 43, {'date': datetime.date(2011, 11, 14)}),
    (1, 44, {'date': datetime.date(2013, 4, 6)}),
    (1, 45, {'date': datetime.date(2008, 6, 22)}),
    (1, 46, {'date': datetime.date(2011, 8, 20)}),
    (1, 47, {'date': datetime.date(2014, 8, 3)}),
    (1, 48, {'date': datetime.date(2010, 3, 15)}),
    (1, 49, {'date': datetime.date(2007, 9, 2)}),
    (16, 18, {'date': datetime.date(2012, 1, 6)}),
    (16, 35, {'date': datetime.date(2014, 6, 4)}),
    (16, 36, {'date': datetime.date(2008, 10, 10)}),
    (16, 48, {'date': datetime.date(2014, 1, 27)}),
    (18, 16, {'date': datetime.date(2008, 8, 5)}),
    (18, 24, {'date': datetime.date(2009, 2, 4)}),
    (18, 35, {'date': datetime.date(2008, 12, 1)}),
    (18, 36, {'date': datetime.date(2013, 2, 6)}),
    (19, 5, {'date': datetime.date(2013, 6, 12)}),
    (19, 8, {'date': datetime.date(2010, 11, 5)}),
    (19, 11, {'date': datetime.date(2012, 4, 16)}),
    (19, 13, {'date': datetime.date(2012, 12, 13)}),
    (19, 15, {'date': datetime.date(2008, 12, 13)}),
    (19, 17, {'date': datetime.date(2007, 11, 11)}),
    (19, 20, {'date': datetime.date(2008, 11, 9)}),
    (19, 21, {'date': datetime.date(2007, 7, 23)}),
    (19, 24, {'date': datetime.date(2013, 12, 13)}),
    (19, 30, {'date': datetime.date(2012, 6, 6)}),
    (19, 31, {'date': datetime.date(2011, 1, 27)}),
    (19, 35, {'date': datetime.date(2014, 3, 3)}),
    (19, 36, {'date': datetime.date(2007, 10, 22)}),
    (19, 37, {'date': datetime.date(2008, 4, 20)}),
    (19, 48, {'date': datetime.date(2010, 12, 23)}),
    (28, 1, {'date': datetime.date(2014, 3, 28)}),
    (28, 5, {'date': datetime.date(2010, 12, 4)}),
    (28, 7, {'date': datetime.date(2011, 11, 21)}),
    (28, 8, {'date': datetime.date(2007, 6, 26)}),
    (28, 11, {'date': datetime.date(2011, 6, 21)}),
    (28, 14, {'date': datetime.date(2013, 12, 18)}),
    (28, 15, {'date': datetime.date(2014, 6, 3)}),
    (28, 17, {'date': datetime.date(2012, 10, 11)}),
    (28, 20, {'date': datetime.date(2012, 4, 15)}),
    (28, 21, {'date': datetime.date(2014, 4, 27)}),
    (28, 24, {'date': datetime.date(2013, 1, 27)}),
    (28, 25, {'date': datetime.date(2014, 5, 9)}),
    (28, 27, {'date': datetime.date(2007, 8, 9)}),
    (28, 29, {'date': datetime.date(2012, 4, 3)}),
    (28, 30, {'date': datetime.date(2007, 12, 2)}),
    (28, 31, {'date': datetime.date(2008, 6, 1)}),
    (28, 35, {'date': datetime.date(2012, 11, 16)}),
    (28, 36, {'date': datetime.date(2012, 9, 26)}),
    (28, 37, {'date': datetime.date(2014, 11, 12)}),
    (28, 44, {'date': datetime.date(2007, 11, 18)}),
    (28, 48, {'date': datetime.date(2008, 5, 25)}),
    (28, 49, {'date': datetime.date(2011, 12, 19)}),
    (36, 5, {'date': datetime.date(2013, 4, 7)}),
    (36, 24, {'date': datetime.date(2009, 4, 23)}),
    (36, 35, {'date': datetime.date(2008, 12, 1)}),
    (36, 37, {'date': datetime.date(2013, 4, 2)}),
    (37, 24, {'date': datetime.date(2008, 6, 27)}),
    (37, 35, {'date': datetime.date(2014, 5, 7)}),
    (37, 36, {'date': datetime.date(2014, 5, 13)}),
    (39, 1, {'date': datetime.date(2007, 4, 8)}),
    (39, 24, {'date': datetime.date(2007, 1, 27)}),
    (39, 33, {'date': datetime.date(2011, 9, 5)}),
    (39, 35, {'date': datetime.date(2007, 6, 17)}),
    (39, 36, {'date': datetime.date(2014, 12, 6)}),
    (39, 38, {'date': datetime.date(2009, 5, 15)}),
    (39, 40, {'date': datetime.date(2011, 6, 3)}),
    (39, 41, {'date': datetime.date(2009, 10, 5)}),
    (39, 45, {'date': datetime.date(2011, 4, 1)}),
    (42, 1, {'date': datetime.date(2013, 3, 9)}),
    (43, 24, {'date': datetime.date(2014, 2, 12)}),
    (43, 29, {'date': datetime.date(2014, 6, 4)}),
    (43, 35, {'date': datetime.date(2009, 6, 10)}),
    (43, 36, {'date': datetime.date(2013, 12, 17)}),
    (43, 37, {'date': datetime.date(2012, 1, 22)}),
    (43, 47, {'date': datetime.date(2014, 12, 21)}),
    (43, 48, {'date': datetime.date(2013, 1, 28)}),
    (45, 1, {'date': datetime.date(2010, 1, 18)}),
    (45, 39, {'date': datetime.date(2011, 1, 12)}),
    (45, 41, {'date': datetime.date(2009, 9, 7)})]
T_sub = nx.Graph()
T_sub.add_nodes_from(T_sub_nodes)
nx.set_node_attributes(T_sub, T_sub_nodes)
T_sub.add_edges_from(T_sub_edges)

# Data from data folder
T = nx.read_gpickle('./data/ego-twitter.p')

# Shared Globals


def first_exposure_to_networkx():
    G = nx.Graph()  # Using nx.Graph we can initialize empty graph which we can add nodes and edges
    G.add_nodes_from([1, 2, 3])  # Add ints 1, 2, and 3 as nodes using add_nodes_from with array arguement
    print(f"Nodes: {G.nodes()}")  # Print list of nodes

    G.add_edge(1, 2)  # Add an edge between nodes 1 and 2
    print(f"Edges: {G.edges()}")

    G.nodes[1]['label'] = 'blue'  # Add metadata key 'label' with data 'blue' to node 1
    print(f"Nodes list with data: {G.nodes(data=True)}")  # Print nodes with metadata attached

    nx.draw(G)  # Takes in a graph as an arguement and creates a plot to display
    plt.show()  # Show the node-link diagram of the graph


def basic_drawing_of_a_network_using_NetworkX():
    # Draw the graph to screen
    nx.draw(T_sub)
    plt.show()


def queries_on_a_graph():

    # Use a list comprehension to get the nodes of interest: noi
    noi = [n for n, d in T.nodes(data=True) if d['occupation'] == 'scientist']

    # Use a list comprehension to get the edges of interest: eoi
    eoi = [(u, v) for u, v, d in T.edges(data=True) if d['date'] < date(2010, 1, 1)]
    return noi, eoi


# Define find_selfloop_nodes()
def specifying_a_weight_on_edges():
    global T

    # Set the weight of the edge
    T.edges[1, 10]['weight'] = 2

    # Iterate over all the edges (with metadata)
    for u, v, d in T.edges(data=True):

        # Check if node 293 is involved
        if 293 in [u, v]:
            # Set the weight to 1.1
            T.edges[u, v]['weight'] = 1.1


def find_selfloop_nodes(G):
    """
    Finds all nodes that have self-loops in the graph G.
    """
    nodes_in_selfloops = []

    # Iterate over all the edges of G
    for u, v in G.edges:
        if u == v:  # Check if node u and node v are the same
            nodes_in_selfloops.append(u)  # Append node u to nodes_in_selfloops
    return nodes_in_selfloops


def checking_whether_there_are_self_loops_in_the_graph():
    if number_of_selfloops(T) == len(find_selfloop_nodes(T)):
        print(f"Numbers Match At {number_of_selfloops(T)} Self Loops")
    else:
        print("Numbers Do Not Match")


def main():
    # first_exposure_to_networkx()
    basic_drawing_of_a_network_using_NetworkX()
    noi, eoi = queries_on_a_graph()
    specifying_a_weight_on_edges()
    checking_whether_there_are_self_loops_in_the_graph()


if __name__ == '__main__':
    main()
