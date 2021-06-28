import matplotlib.pyplot as plt  # Used to draw network plot
import networkx as nx  # NetworkX is typically imported as nx
import datetime
import nxviz as nv  # Network Visualizations
from nxviz import CircosPlot, ArcPlot


T_nodes = {3584: {'category': 'D', 'occupation': 'celebrity'}, 6147: {'category': 'I', 'occupation': 'celebrity'},
           15369: {'category': 'I', 'occupation': 'celebrity'}, 17419: {'category': 'P', 'occupation': 'scientist'},
           8204: {'category': 'D', 'occupation': 'celebrity'}, 11281: {'category': 'I', 'occupation': 'politician'},
           18450: {'category': 'D', 'occupation': 'scientist'}, 1530: {'category': 'I', 'occupation': 'celebrity'},
           14357: {'category': 'P', 'occupation': 'scientist'}, 11286: {'category': 'D', 'occupation': 'politician'},
           17432: {'category': 'D', 'occupation': 'celebrity'}, 538: {'category': 'I', 'occupation': 'celebrity'},
           3100: {'category': 'D', 'occupation': 'scientist'}, 11804: {'category': 'D', 'occupation': 'scientist'},
           9262: {'category': 'I', 'occupation': 'scientist'}, 11824: {'category': 'P', 'occupation': 'celebrity'},
           8758: {'category': 'P', 'occupation': 'celebrity'}, 20026: {'category': 'I', 'occupation': 'celebrity'},
           2619: {'category': 'D', 'occupation': 'politician'}, 20030: {'category': 'D', 'occupation': 'scientist'},
           7233: {'category': 'P', 'occupation': 'celebrity'}, 7746: {'category': 'P', 'occupation': 'scientist'},
           3141: {'category': 'P', 'occupation': 'politician'}, 13893: {'category': 'I', 'occupation': 'politician'},
           16455: {'category': 'P', 'occupation': 'celebrity'}, 7238: {'category': 'D', 'occupation': 'politician'},
           72: {'category': 'P', 'occupation': 'politician'}, 1098: {'category': 'D', 'occupation': 'celebrity'},
           4171: {'category': 'P', 'occupation': 'celebrity'}, 1100: {'category': 'D', 'occupation': 'politician'},
           20044: {'category': 'I', 'occupation': 'politician'}, 2639: {'category': 'P', 'occupation': 'politician'},
           11859: {'category': 'I', 'occupation': 'politician'}, 11860: {'category': 'I', 'occupation': 'celebrity'},
           12372: {'category': 'P', 'occupation': 'politician'}, 17495: {'category': 'I', 'occupation': 'scientist'},
           19034: {'category': 'P', 'occupation': 'politician'}, 4187: {'category': 'D', 'occupation': 'celebrity'},
           6748: {'category': 'P', 'occupation': 'celebrity'}, 2141: {'category': 'P', 'occupation': 'scientist'},
           2654: {'category': 'D', 'occupation': 'celebrity'}, 19550: {'category': 'D', 'occupation': 'politician'},
           16998: {'category': 'D', 'occupation': 'celebrity'}, 10857: {'category': 'I', 'occupation': 'celebrity'},
           1129: {'category': 'D', 'occupation': 'scientist'}, 4718: {'category': 'I', 'occupation': 'politician'},
           6770: {'category': 'D', 'occupation': 'celebrity'}, 6260: {'category': 'P', 'occupation': 'politician'},
           2677: {'category': 'P', 'occupation': 'scientist'}, 15991: {'category': 'I', 'occupation': 'scientist'},
           9335: {'category': 'P', 'occupation': 'politician'}, 17017: {'category': 'P', 'occupation': 'celebrity'},
           13955: {'category': 'D', 'occupation': 'scientist'}, 134: {'category': 'P', 'occupation': 'politician'},
           8840: {'category': 'P', 'occupation': 'scientist'}, 17034: {'category': 'I', 'occupation': 'celebrity'},
           9359: {'category': 'P', 'occupation': 'celebrity'}, 19088: {'category': 'D', 'occupation': 'scientist'},
           144: {'category': 'D', 'occupation': 'scientist'}, 3220: {'category': 'P', 'occupation': 'scientist'},
           666: {'category': 'P', 'occupation': 'scientist'}, 2717: {'category': 'D', 'occupation': 'politician'},
           3230: {'category': 'I', 'occupation': 'scientist'}, 13471: {'category': 'I', 'occupation': 'celebrity'},
           2208: {'category': 'P', 'occupation': 'celebrity'}, 12459: {'category': 'P', 'occupation': 'politician'},
           179: {'category': 'P', 'occupation': 'celebrity'}, 6326: {'category': 'D', 'occupation': 'celebrity'},
           6841: {'category': 'P', 'occupation': 'celebrity'}, 186: {'category': 'I', 'occupation': 'scientist'},
           3772: {'category': 'D', 'occupation': 'celebrity'}, 6334: {'category': 'I', 'occupation': 'politician'},
           14022: {'category': 'I', 'occupation': 'politician'}, 2251: {'category': 'I', 'occupation': 'celebrity'},
           6862: {'category': 'I', 'occupation': 'politician'}, 6864: {'category': 'I', 'occupation': 'scientist'},
           3283: {'category': 'I', 'occupation': 'scientist'}, 3289: {'category': 'D', 'occupation': 'scientist'},
           11996: {'category': 'D', 'occupation': 'politician'}, 4317: {'category': 'I', 'occupation': 'scientist'},
           11488: {'category': 'P', 'occupation': 'politician'}, 19170: {'category': 'I', 'occupation': 'celebrity'},
           10981: {'category': 'I', 'occupation': 'celebrity'}, 12006: {'category': 'D', 'occupation': 'celebrity'},
           10982: {'category': 'I', 'occupation': 'politician'}, 1256: {'category': 'I', 'occupation': 'scientist'},
           11501: {'category': 'D', 'occupation': 'politician'}, 8942: {'category': 'I', 'occupation': 'scientist'},
           2802: {'category': 'P', 'occupation': 'politician'}, 18675: {'category': 'P', 'occupation': 'politician'},
           15092: {'category': 'I', 'occupation': 'scientist'}, 20726: {'category': 'I', 'occupation': 'celebrity'},
           5368: {'category': 'P', 'occupation': 'celebrity'}, 13561: {'category': 'P', 'occupation': 'politician'},
           20730: {'category': 'D', 'occupation': 'celebrity'}, 13564: {'category': 'P', 'occupation': 'scientist'},
           12039: {'category': 'I', 'occupation': 'politician'}, 2824: {'category': 'D', 'occupation': 'celebrity'},
           18697: {'category': 'I', 'occupation': 'politician'}, 23307: {'category': 'P', 'occupation': 'celebrity'},
           3342: {'category': 'I', 'occupation': 'politician'}, 22287: {'category': 'P', 'occupation': 'celebrity'},
           16656: {'category': 'D', 'occupation': 'celebrity'}, 15636: {'category': 'D', 'occupation': 'politician'},
           1307: {'category': 'P', 'occupation': 'celebrity'}, 12063: {'category': 'P', 'occupation': 'politician'},
           4384: {'category': 'P', 'occupation': 'politician'}, 4895: {'category': 'P', 'occupation': 'politician'},
           17702: {'category': 'D', 'occupation': 'celebrity'}, 808: {'category': 'P', 'occupation': 'politician'},
           20268: {'category': 'D', 'occupation': 'celebrity'}, 19244: {'category': 'I', 'occupation': 'celebrity'},
           17712: {'category': 'D', 'occupation': 'politician'}, 4400: {'category': 'I', 'occupation': 'scientist'},
           23348: {'category': 'P', 'occupation': 'politician'}, 15671: {'category': 'I', 'occupation': 'politician'},
           15672: {'category': 'D', 'occupation': 'politician'}, 6968: {'category': 'P', 'occupation': 'celebrity'},
           3898: {'category': 'P', 'occupation': 'politician'}, 825: {'category': 'I', 'occupation': 'scientist'},
           18745: {'category': 'D', 'occupation': 'scientist'}, 7488: {'category': 'P', 'occupation': 'celebrity'},
           834: {'category': 'P', 'occupation': 'politician'}, 844: {'category': 'D', 'occupation': 'celebrity'},
           3917: {'category': 'D', 'occupation': 'scientist'}, 2894: {'category': 'I', 'occupation': 'celebrity'},
           17740: {'category': 'I', 'occupation': 'politician'}, 2385: {'category': 'D', 'occupation': 'celebrity'},
           15190: {'category': 'D', 'occupation': 'celebrity'}, 1368: {'category': 'D', 'occupation': 'celebrity'},
           4441: {'category': 'I', 'occupation': 'politician'}, 15195: {'category': 'I', 'occupation': 'celebrity'},
           3420: {'category': 'P', 'occupation': 'politician'}, 6493: {'category': 'P', 'occupation': 'politician'},
           862: {'category': 'I', 'occupation': 'politician'}, 859: {'category': 'P', 'occupation': 'politician'},
           864: {'category': 'D', 'occupation': 'politician'}, 2405: {'category': 'D', 'occupation': 'celebrity'},
           4966: {'category': 'I', 'occupation': 'celebrity'}, 2922: {'category': 'I', 'occupation': 'celebrity'},
           16750: {'category': 'P', 'occupation': 'celebrity'}, 19316: {'category': 'I', 'occupation': 'politician'},
           2934: {'category': 'P', 'occupation': 'scientist'}, 12156: {'category': 'P', 'occupation': 'celebrity'},
           7045: {'category': 'I', 'occupation': 'politician'}, 6021: {'category': 'D', 'occupation': 'politician'},
           12679: {'category': 'I', 'occupation': 'politician'}, 19336: {'category': 'I', 'occupation': 'politician'},
           12168: {'category': 'P', 'occupation': 'scientist'}, 905: {'category': 'D', 'occupation': 'politician'},
           16268: {'category': 'D', 'occupation': 'politician'}, 17806: {'category': 'D', 'occupation': 'politician'},
           10134: {'category': 'P', 'occupation': 'scientist'}, 16279: {'category': 'I', 'occupation': 'celebrity'},
           22423: {'category': 'P', 'occupation': 'celebrity'}, 5529: {'category': 'P', 'occupation': 'politician'},
           16795: {'category': 'D', 'occupation': 'celebrity'}, 18844: {'category': 'P', 'occupation': 'scientist'},
           7069: {'category': 'D', 'occupation': 'scientist'}, 18845: {'category': 'D', 'occupation': 'celebrity'},
           15263: {'category': 'P', 'occupation': 'celebrity'}, 5542: {'category': 'P', 'occupation': 'politician'},
           6058: {'category': 'P', 'occupation': 'celebrity'}, 12720: {'category': 'P', 'occupation': 'celebrity'},
           3505: {'category': 'D', 'occupation': 'scientist'}, 10166: {'category': 'D', 'occupation': 'politician'},
           16823: {'category': 'P', 'occupation': 'politician'}, 18874: {'category': 'I', 'occupation': 'politician'},
           5565: {'category': 'P', 'occupation': 'politician'}, 7102: {'category': 'D', 'occupation': 'celebrity'},
           2503: {'category': 'P', 'occupation': 'politician'}, 969: {'category': 'P', 'occupation': 'celebrity'},
           16842: {'category': 'D', 'occupation': 'scientist'}, 1996: {'category': 'D', 'occupation': 'celebrity'},
           5586: {'category': 'P', 'occupation': 'scientist'}, 5076: {'category': 'P', 'occupation': 'celebrity'},
           16853: {'category': 'P', 'occupation': 'politician'}, 15834: {'category': 'P', 'occupation': 'celebrity'},
           17889: {'category': 'D', 'occupation': 'politician'}, 19937: {'category': 'P', 'occupation': 'celebrity'},
           6117: {'category': 'D', 'occupation': 'celebrity'}, 14833: {'category': 'D', 'occupation': 'scientist'},
           3571: {'category': 'D', 'occupation': 'politician'}, 1529: {'category': 'P', 'occupation': 'celebrity'},
           5626: {'category': 'P', 'occupation': 'politician'}, 14846: {'category': 'P', 'occupation': 'politician'}}
T_edges = [(15369, 2824, {'date': datetime.date(2009, 11, 10)}), (17419, 17432, {'date': datetime.date(2007, 1, 10)}),
           (17419, 15092, {'date': datetime.date(2012, 4, 13)}), (17419, 17495, {'date': datetime.date(2007, 9, 28)}),
           (11281, 11286, {'date': datetime.date(2011, 5, 20)}), (11804, 1529, {'date': datetime.date(2007, 2, 8)}),
           (11804, 1100, {'date': datetime.date(2013, 12, 11)}), (11804, 1129, {'date': datetime.date(2014, 1, 7)}),
           (9262, 9335, {'date': datetime.date(2009, 10, 4)}), (11824, 11859, {'date': datetime.date(2007, 10, 20)}),
           (11824, 11860, {'date': datetime.date(2007, 6, 22)}), (11824, 4718, {'date': datetime.date(2010, 8, 21)}),
           (11824, 2503, {'date': datetime.date(2011, 4, 20)}), (11824, 11996, {'date': datetime.date(2013, 11, 21)}),
           (11824, 12006, {'date': datetime.date(2014, 8, 15)}), (20026, 20030, {'date': datetime.date(2013, 9, 21)}),
           (7746, 4895, {'date': datetime.date(2009, 4, 8)}), (7746, 538, {'date': datetime.date(2008, 3, 27)}),
           (7746, 1098, {'date': datetime.date(2011, 2, 22)}), (7746, 1100, {'date': datetime.date(2008, 6, 1)}),
           (7746, 1129, {'date': datetime.date(2012, 10, 7)}), (16455, 1530, {'date': datetime.date(2013, 9, 11)}),
           (16455, 4171, {'date': datetime.date(2008, 7, 18)}), (4171, 4187, {'date': datetime.date(2014, 10, 27)}),
           (2639, 7488, {'date': datetime.date(2013, 6, 26)}), (2639, 2717, {'date': datetime.date(2011, 8, 10)}),
           (19034, 19088, {'date': datetime.date(2010, 11, 18)}), (2141, 72, {'date': datetime.date(2011, 12, 26)}),
           (2141, 2208, {'date': datetime.date(2011, 2, 20)}), (2141, 2251, {'date': datetime.date(2010, 12, 2)}),
           (2654, 8942, {'date': datetime.date(2011, 4, 16)}), (19550, 15263, {'date': datetime.date(2007, 12, 7)}),
           (19550, 3283, {'date': datetime.date(2011, 10, 1)}), (16998, 17017, {'date': datetime.date(2011, 2, 19)}),
           (16998, 17034, {'date': datetime.date(2009, 4, 3)}), (10857, 19244, {'date': datetime.date(2014, 3, 23)}),
           (6770, 2619, {'date': datetime.date(2010, 1, 3)}), (6770, 8942, {'date': datetime.date(2012, 4, 16)}),
           (6770, 2654, {'date': datetime.date(2007, 3, 26)}), (6260, 19170, {'date': datetime.date(2012, 2, 20)}),
           (2677, 13955, {'date': datetime.date(2013, 10, 5)}), (2677, 8942, {'date': datetime.date(2008, 6, 3)}),
           (2677, 2654, {'date': datetime.date(2010, 12, 13)}), (2677, 14022, {'date': datetime.date(2008, 3, 27)}),
           (2677, 6770, {'date': datetime.date(2013, 12, 14)}), (134, 144, {'date': datetime.date(2010, 2, 4)}),
           (8840, 1307, {'date': datetime.date(2007, 12, 5)}), (3220, 3141, {'date': datetime.date(2012, 11, 14)}),
           (2717, 2639, {'date': datetime.date(2012, 9, 8)}), (2717, 2654, {'date': datetime.date(2012, 5, 17)}),
           (2717, 6770, {'date': datetime.date(2008, 3, 7)}), (3230, 6326, {'date': datetime.date(2007, 6, 23)}),
           (3230, 6334, {'date': datetime.date(2014, 10, 26)}), (13471, 15991, {'date': datetime.date(2010, 9, 23)}),
           (12459, 18697, {'date': datetime.date(2014, 8, 16)}), (12459, 905, {'date': datetime.date(2010, 11, 25)}),
           (179, 186, {'date': datetime.date(2010, 10, 25)}), (6841, 6862, {'date': datetime.date(2011, 7, 6)}),
           (6841, 6864, {'date': datetime.date(2012, 4, 17)}), (3772, 8204, {'date': datetime.date(2011, 3, 2)}),
           (2251, 2141, {'date': datetime.date(2010, 2, 23)}), (2251, 72, {'date': datetime.date(2013, 5, 1)}),
           (2251, 13893, {'date': datetime.date(2009, 10, 1)}), (6862, 14357, {'date': datetime.date(2009, 1, 11)}),
           (3283, 18675, {'date': datetime.date(2007, 5, 15)}), (3289, 3342, {'date': datetime.date(2008, 9, 24)}),
           (4317, 4384, {'date': datetime.date(2011, 3, 6)}), (4317, 4400, {'date': datetime.date(2013, 7, 3)}),
           (4317, 4441, {'date': datetime.date(2014, 8, 19)}), (10981, 17712, {'date': datetime.date(2011, 7, 22)}),
           (10981, 17740, {'date': datetime.date(2009, 3, 7)}), (10982, 23348, {'date': datetime.date(2014, 2, 19)}),
           (1256, 11488, {'date': datetime.date(2013, 11, 17)}), (1256, 11501, {'date': datetime.date(2013, 4, 6)}),
           (2802, 8942, {'date': datetime.date(2010, 3, 13)}), (2802, 6770, {'date': datetime.date(2013, 5, 5)}),
           (15092, 17806, {'date': datetime.date(2011, 2, 16)}), (20726, 20730, {'date': datetime.date(2011, 6, 26)}),
           (5368, 13564, {'date': datetime.date(2010, 11, 19)}), (5368, 15834, {'date': datetime.date(2014, 7, 4)}),
           (13561, 13564, {'date': datetime.date(2014, 3, 27)}), (12039, 12063, {'date': datetime.date(2010, 4, 1)}),
           (23307, 666, {'date': datetime.date(2007, 5, 16)}), (15636, 15671, {'date': datetime.date(2008, 1, 28)}),
           (15636, 15672, {'date': datetime.date(2014, 9, 17)}), (4384, 4317, {'date': datetime.date(2014, 9, 6)}),
           (4384, 1307, {'date': datetime.date(2014, 3, 6)}), (4384, 8758, {'date': datetime.date(2014, 7, 12)}),
           (17702, 72, {'date': datetime.date(2008, 11, 4)}), (17702, 12168, {'date': datetime.date(2014, 7, 27)}),
           (808, 825, {'date': datetime.date(2010, 2, 24)}), (808, 834, {'date': datetime.date(2011, 10, 28)}),
           (808, 862, {'date': datetime.date(2013, 5, 8)}), (6968, 16268, {'date': datetime.date(2010, 8, 7)}),
           (6968, 16279, {'date': datetime.date(2013, 4, 19)}), (3898, 3917, {'date': datetime.date(2014, 4, 17)}),
           (825, 808, {'date': datetime.date(2012, 2, 13)}), (825, 834, {'date': datetime.date(2010, 2, 13)}),
           (825, 844, {'date': datetime.date(2011, 3, 28)}), (825, 859, {'date': datetime.date(2011, 5, 12)}),
           (825, 6493, {'date': datetime.date(2007, 1, 4)}), (825, 862, {'date': datetime.date(2012, 11, 19)}),
           (825, 864, {'date': datetime.date(2012, 11, 18)}), (18745, 18844, {'date': datetime.date(2007, 9, 25)}),
           (844, 825, {'date': datetime.date(2011, 9, 3)}), (844, 6493, {'date': datetime.date(2010, 11, 20)}),
           (844, 862, {'date': datetime.date(2013, 11, 1)}), (844, 864, {'date': datetime.date(2014, 4, 16)}),
           (2894, 2922, {'date': datetime.date(2008, 7, 21)}), (2385, 2405, {'date': datetime.date(2008, 9, 12)}),
           (15190, 15195, {'date': datetime.date(2012, 11, 17)}), (1368, 22287, {'date': datetime.date(2009, 3, 20)}),
           (1368, 18845, {'date': datetime.date(2008, 1, 17)}), (3420, 3505, {'date': datetime.date(2014, 1, 21)}),
           (859, 825, {'date': datetime.date(2014, 8, 16)}), (859, 834, {'date': datetime.date(2007, 11, 25)}),
           (859, 18450, {'date': datetime.date(2011, 2, 16)}), (859, 862, {'date': datetime.date(2009, 7, 28)}),
           (859, 864, {'date': datetime.date(2007, 3, 28)}), (864, 825, {'date': datetime.date(2008, 8, 11)}),
           (864, 844, {'date': datetime.date(2014, 1, 4)}), (864, 859, {'date': datetime.date(2008, 8, 16)}),
           (864, 19937, {'date': datetime.date(2014, 1, 15)}), (864, 862, {'date': datetime.date(2012, 9, 25)}),
           (2405, 2385, {'date': datetime.date(2014, 9, 23)}), (4966, 20268, {'date': datetime.date(2007, 2, 26)}),
           (16750, 538, {'date': datetime.date(2008, 3, 18)}), (16750, 1100, {'date': datetime.date(2013, 8, 13)}),
           (16750, 3283, {'date': datetime.date(2007, 7, 22)}), (19316, 19336, {'date': datetime.date(2009, 6, 9)}),
           (12156, 72, {'date': datetime.date(2007, 12, 4)}), (12156, 12168, {'date': datetime.date(2007, 3, 1)}),
           (7045, 905, {'date': datetime.date(2007, 1, 11)}), (7045, 7069, {'date': datetime.date(2009, 6, 17)}),
           (6021, 3100, {'date': datetime.date(2009, 8, 14)}), (6021, 5368, {'date': datetime.date(2011, 6, 18)}),
           (6021, 6058, {'date': datetime.date(2010, 9, 14)}), (12679, 12720, {'date': datetime.date(2011, 7, 23)}),
           (12679, 3283, {'date': datetime.date(2008, 11, 21)}), (905, 12372, {'date': datetime.date(2010, 9, 22)}),
           (905, 12459, {'date': datetime.date(2012, 10, 21)}), (5529, 5542, {'date': datetime.date(2011, 7, 14)}),
           (16795, 6748, {'date': datetime.date(2009, 10, 15)}), (18845, 18874, {'date': datetime.date(2008, 12, 7)}),
           (6058, 17889, {'date': datetime.date(2011, 4, 15)}), (10166, 2934, {'date': datetime.date(2011, 12, 20)}),
           (16823, 16842, {'date': datetime.date(2009, 12, 7)}), (5565, 5586, {'date': datetime.date(2008, 6, 18)}),
           (7102, 7233, {'date': datetime.date(2009, 6, 11)}), (7102, 7238, {'date': datetime.date(2013, 12, 15)}),
           (2503, 16279, {'date': datetime.date(2009, 2, 27)}), (2503, 11824, {'date': datetime.date(2009, 5, 6)}),
           (2503, 11996, {'date': datetime.date(2013, 9, 2)}), (969, 20044, {'date': datetime.date(2008, 12, 26)}),
           (1996, 9359, {'date': datetime.date(2007, 6, 21)}), (5076, 22423, {'date': datetime.date(2008, 11, 5)}),
           (16853, 134, {'date': datetime.date(2009, 1, 9)}), (16853, 10134, {'date': datetime.date(2008, 12, 16)}),
           (16853, 144, {'date': datetime.date(2013, 10, 27)}), (6117, 6147, {'date': datetime.date(2014, 7, 1)}),
           (14833, 14846, {'date': datetime.date(2013, 7, 12)}), (3571, 3584, {'date': datetime.date(2008, 11, 14)}),
           (1529, 538, {'date': datetime.date(2014, 12, 26)}), (1529, 1100, {'date': datetime.date(2007, 3, 1)}),
           (5626, 16656, {'date': datetime.date(2009, 1, 9)})]
T = nx.Graph()
T.add_nodes_from(T_nodes)
nx.set_node_attributes(T, T_nodes)
T.add_edges_from(T_edges)


def nxviz_quickstart():
    # G = nx.Graph()  # Using nx.Graph we can initialize empty graph which we can add nodes and edges
    # G.add_nodes_from([1, 2, 3])  # Add ints 1, 2, and 3 as nodes using add_nodes_from with array arguement
    # G.add_edge(1, 2)  # Add an edge between nodes 1 and 2

    ap = nv.ArcPlot(T)
    ap.draw()
    plt.show()


def visualizing_using_matrix_plots():
    """
    Visualizing using Matrix plots
    It is time to try your first "fancy" graph visualization method: a matrix plot. To do this, nxviz provides a
    MatrixPlot object.

    nxviz is a package for visualizing graphs in a rational fashion. Under the hood, the MatrixPlot utilizes
    nx.to_numpy_matrix(G), which returns the matrix form of the graph. Here, each node is one column and one row,
    and an edge between the two nodes is indicated by the value 1. In doing so, however, only the weight metadata is
    preserved; all other metadata is lost, as you'll verify using an assert statement.

    A corresponding nx.from_numpy_matrix(A) allows one to quickly create a graph from a NumPy matrix. The default
    graph type is Graph(); if you want to make it a DiGraph(), that has to be specified using the create_using keyword
    argument, e.g. (nx.from_numpy_matrix(A, create_using=nx.DiGraph)).

    One final note, matplotlib.pyplot and networkx have already been imported as plt and nx, respectively, and the
    graph T has been pre-loaded. For simplicity and speed, we have sub-sampled only 100 edges from the network.

    Instructions:
    + Import nxviz as nv.
    + Plot the graph T as a matrix plot. To do this:
    + Create the MatrixPlot object called m using the nv.MatrixPlot() function with T passed in as an argument.
    + Draw the m to the screen using the .draw() method.
    + Display the plot using plt.show().
    + Convert the graph to a matrix format, and then convert the graph to back to the NetworkX form from the matrix as a directed graph. This has been done for you.
    + Check that the category metadata field is lost from each node. This has also been done for you, so hit 'Submit Answer' to see the results!
    """

    # Create the MatrixPlot object: m
    m = nv.MatrixPlot(T)

    # Draw m to the screen
    m.draw()

    # Display the plot
    plt.show()

    # Convert T to a matrix format: A
    A = nx.to_numpy_matrix(T)

    # Convert A back to the NetworkX form as a directed graph: T_conv
    T_conv = nx.from_numpy_matrix(A, create_using=nx.DiGraph())

    # Check that the `category` metadata field is lost from each node
    for n, d in T_conv.nodes(data=True):
        assert 'category' not in d.keys()


def visualizing_using_circos_plots():
    # Create the CircosPlot object: c
    c = CircosPlot(T)

    # Draw c to the screen
    c.draw()

    # Display the plot
    plt.show()


def visualizing_using_arc_plots():
    # Create the un-customized ArcPlot object: a
    a = ArcPlot(T)

    # Draw a to the screen
    a.draw()

    # Display the plot
    plt.show()

    # Create the customized ArcPlot object: a2
    a2 = ArcPlot(T, node_order='category', node_color='category')

    # Draw a2 to the screen
    a2.draw()

    # Display the plot
    plt.show()

    """
    Excellent job! Notice the node coloring in the customized ArcPlot compared to the uncustomized version. In the 
    customized ArcPlot, the nodes in each of the categories - 'I', 'D', and 'P' - have their own color. If it's 
    difficult to see on your screen, you can expand the plot into a new window by clicking on the pop-out icon on the
    top-left next to 'Plots'.
    """


def v_number_of_neighbors():
    g_nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    g_edges = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9)]
    G = nx.Graph()
    G.add_nodes_from(g_nodes)
    G.add_edges_from(g_edges)
    print(f"Nodes: {G.nodes}")
    print(f"Edges: {G.edges}")

    print(f"G.neighbors(1): {list(G.neighbors(1))}")
    print(f"G.neighbors(8): {list(G.neighbors(8))}")
    # print(f"G.neighbors(10): {list(G.neighbors(10))}")  # Throws error doesn't exist

    print("nx.degree_centrality(G):")
    print(nx.degree_centrality(G))  # NOTE: SELF LOOPS ARE NOT CONSIDERED WITH THIS FUNCTION!


# Define nodes_with_m_nbrs()
def nodes_with_m_nbrs(G, m):
    """
    Returns all nodes in graph G that have m neighbors.
    """
    nodes = set()

    # Iterate over all nodes in G
    for n in G.nodes():

        # Check if the number of neighbors of n matches m
        if len(list(G.neighbors(n))) == m:

            # Add the node n to the set
            nodes.add(n)

    # Return the nodes with m neighbors
    return nodes


def compute_number_of_neighbors_for_each_node():
    # Compute and print all nodes in T that have 6 neighbors
    six_nbrs = nodes_with_m_nbrs(T, 6)
    print(f"All Nodes In T That Have 6 Neighbors: {six_nbrs}")
    print(f"Total Number: {len(six_nbrs)}")

    """
    Great work! The number of neighbors a node has is one way to identify important nodes. 
    It looks like 2 nodes in graph T have 6 neighbors.
    """


def compute_degree_distribution():
    """
    Compute degree distribution
    The number of neighbors that a node has is called its "degree", and it's possible to compute the degree distribution across the
    entire graph. In this exercise, your job is to compute the degree distribution across T.

    Instructions:
    + Use a list comprehension along with the .neighbors(n) method to get the degree of every node. The result should be a list of
      integers.
    + Use n as your iterator variable.
    + The output expression of your list comprehension should be the number of neighbors that node n has - that is, its degree. Use
      the len() and list() functions together with the .neighbors() method to compute this.
    + The iterable in your list comprehension is all the nodes in T, accessed using the .nodes() method.
    + Print the degrees.
    """
    # fruits = ["apple", "banana", "cherry", "kiwi", "mango"]
    # newlist = [x for x in fruits if "a" in x]

    # Compute the degree of every node: degrees
    degrees = [len(list(T.neighbors(n))) for n in T.nodes()]

    # Print the degrees
    print(f"The Degree Of Every Node In T: {degrees}")
    return degrees


def degree_centrality_distribution(degrees):
    """
    Degree centrality distribution
    The degree of a node is the number of neighbors that it has. The degree centrality is the number of neighbors divided by all
    possible neighbors that it could have. Depending on whether self-loops are allowed, the set of possible neighbors a node could
    have could also include the node itself.

    The nx.degree_centrality(G) function returns a dictionary, where the keys are the nodes and the values are their degree
    centrality values.

    The degree distribution degrees you computed in the previous exercise using the list comprehension has been pre-loaded.

    Instructions:
    + Compute the degree centrality of the Twitter network T.
    + Using plt.hist(), plot a histogram of the degree centrality distribution of T. This can be accessed using
      list(deg_cent.values()).
    + Plot a histogram of the degree distribution degrees of T. This is the same list you computed in the last exercise.
    + Create a scatter plot with degrees on the x-axis and the degree centrality distribution list(deg_cent.values()) on the y-axis.
    """

    # Compute the degree centrality of the Twitter network: deg_cent
    deg_cent = nx.degree_centrality(T)

    # Plot a histogram of the degree centrality distribution of the graph.
    plt.figure()
    plt.hist(list(deg_cent.values()))
    plt.show()

    # Plot a histogram of the degree distribution of the graph
    plt.figure()
    plt.hist(degrees)
    plt.show()

    # Plot a scatter plot of the centrality distribution and the degree distribution
    plt.figure()
    plt.scatter(degrees, list(deg_cent.values()))
    plt.show()

    """
    Great work! Click the 'Next Plot' and 'Previous Plot' buttons to cycle through your 3 plots. Given the similarities of their 
    histograms, it should not surprise you to see a perfect correlation between the centrality distribution and the degree 
    distribution.
    """


def shortest_path_i():
    """
    Shortest Path I
    You can leverage what you know about finding neighbors to try finding paths in a network. One algorithm for path-finding between two nodes is the "breadth-first search" (BFS) algorithm. In a BFS algorithm, you start from a particular node and iteratively search through its neighbors and neighbors' neighbors until you find the destination node.

    Pathfinding algorithms are important because they provide another way of assessing node importance; you'll see this in a later exercise.

    In this set of 3 exercises, you're going to build up slowly to get to the final BFS algorithm. The problem has been broken into 3 parts that, if you complete in succession, will get you to a first pass implementation of the BFS algorithm.

    Instructions:
    + Create a function called path_exists() that has 3 parameters - G, node1, and node2 - and returns whether or not a path exists
      between the two nodes.
    + Initialize the queue of nodes to visit with the first node, node1. queue should be a list.
    + Iterate over the nodes in queue.
    + Get the neighbors of the node using the .neighbors() method of the graph G.
    + Check to see if the destination node node2 is in the set of neighbors. If it is, return True.
    """
    # See path_exists() below


def path_exists(G, node1, node2):
    """
        This function checks whether a path exists between two nodes (node1, node2) in graph G.
        """
    visited_nodes = set()

    # Initialize the queue of nodes to visit with the first node: queue
    queue = [node1]

    # Iterate over the nodes in the queue
    for node in queue:

        # Get neighbors of the node
        neighbors = G.neighbors(node)

        # Check to see if the destination node is in the set of neighbors
        if node2 in neighbors:
            print('Path exists between nodes {0} and {1}'.format(node1, node2))
            return True
            break

    """
    Great! In the next exercise, you'll extend this function by including the condition where the destination node is not 
    present in the neighbors.
    """


def main():
    # nxviz_quickstart()
    # visualizing_using_matrix_plots()
    # visualizing_using_circos_plots()
    # visualizing_using_arc_plots()
    # v_number_of_neighbors()
    # compute_number_of_neighbors_for_each_node()
    degrees = compute_degree_distribution()
    degree_centrality_distribution(degrees)
    shortest_path_i()


if __name__ == '__main__':
    main()
