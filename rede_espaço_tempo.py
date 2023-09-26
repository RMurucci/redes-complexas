#!/usr/bin/python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns
import random as rd
from networkx_robustness import networkx_robustness as nx_r

### LEITURA DA MATRIZ DE ADJACÊNCIAS
#grafo direcionado com peso
graph_filepath = "C:\Romulo\Modelagem de Redes Complexas\matriz_custo_SF_SD.csv"
df = pd.read_csv(graph_filepath, sep=';', index_col=0)
df1 = df.replace({',': '.'}, regex=True)
df2 = df1.astype('float')

### CRIAÇÃO DO GRAFO
D = nx.from_pandas_adjacency(df2, create_using = nx.DiGraph())

### NÓS E ARCOS
n = len(list(D.nodes()))
m = len(list(D.edges()))

### GRAU
D_degree_dict = dict(nx.degree(D))
D_degree = list(D_degree_dict.values())

# Grau médio
D_array = np.array(D_degree)
D_degree_mean = np.mean(D_array)

# Distribuição de graus (histograma)
bin_edges = [2,4,6,8,10]
sns.set()
plt.hist(D_degree, bins = bin_edges)
plt.ylabel("Frequencie")
plt.title("Degree histogram (original network)")

# Distribuição de graus (densidade)
sns.set()
sns.kdeplot(data = D_degree, fill=True, cut = 0)
plt.title("Degree distribution")

### BUSCA EM LARGURA (BFS) E BUSCA EM PROFUNDIDADE (DFS)
root = 'SF'
# Arcos e nós gerados pela busca em largura
bfs_edges = list(nx.bfs_edges(D, root))
bfs_nodes = [root] + [v for u, v in bfs_edges]

# Árvore geradora da busca em largura
bfs_tree = nx.bfs_tree(D, root)
bfs_tree_edges = list(bfs_tree.edges())
bfs_tree_nodes = list(bfs_tree.nodes())

# Arcos e nós gerados pela busca em profundidade
dfs_edges = list(nx.dfs_edges(D, root))
dfs_nodes = [root] + [v for u, v in dfs_edges]

# Árvore geradora da busca em profundidade
dfs_tree = nx.dfs_tree(D, root)
dfs_tree_edges = list(dfs_tree.edges())
dfs_tree_nodes = list(dfs_tree.nodes())

### DESENHO DO GRAFO
#nx.draw(D, node_size = 10, with_labels=True, font_size=9)
#gv.graphviz_layout(D, prog='neato', root='SF')
#nx.draw(bfs_tree, node_size = 10, with_labels=True, font_size=9)
#nx.draw(dfs_tree, node_size = 10, with_labels=True, font_size=9)

### COEFICIENTE DE CLUSTERIZAÇÃO
transit = nx.transitivity(D) #metrica global
avg_clust = nx.average_clustering(D, weight='weight') #metrica local
#calcular clustering de cada nó e comparar original e aleatório

### CENTRALIDADE
#degree_cent = nx.degree_centrality(D)
degree_cent_in = nx.in_degree_centrality(D)
degree_cent_out = nx.out_degree_centrality(D)
#center = nx.center(D)
density = nx.density(D)

### BETWEENNESS, CLOSENESS AND PAGERANK CENTRALITY
between = nx.betweenness_centrality(D)
#close_in = nx.closeness_centrality(D) #não aplicável conforme teoria estudada
close_out = nx.closeness_centrality(D.reverse())
pr = nx.pagerank(D, weight='weight')

### DETECÇÃO DE COMUNIDADES - ALGORITMO DE LOUVAIN
# Modularidade calculada sem biblioteca (não considera pesos dos arcos)
def communities(D, c):
    nc = 0
    for k in range(0,len(c)):
        for l in list(c[k]):
            for m in list(D.keys()):
                if l == m:
                    D[m] = nc
        nc = nc + 1
    return D

def modularity(G, c, A):
    M = G.number_of_edges()
    Q = 0
    for i in list(G.nodes()):
        ki = len(list(G.neighbors(i)))
        for j in list(G.nodes()):
            if(c[i] == c[j]):
                kj = len(list(G.neighbors(j)))
                if A[i][j] > 0:
                    Q = Q + 1 - (ki*kj)/(2*M)
                else:
                    Q = Q - (ki*kj)/(2*M)
    Q = Q/(2*M)
    return Q

louvain_com = nx.community.louvain_communities(D) #lista de sets
D_nodes = dict(nx.nodes(D))
D_communities = communities(D_nodes, louvain_com)
modularity = modularity(D, D_communities, df2) #não considera peso dos arcos

# Modularidade calculada com biblioteca (Networkx)
#precisa mudar para lista de frozensets ou lista de tuples
louvain_c = [
                frozenset(community) 
                for community in louvain_com
            ]
modularity_nx = nx.community.modularity(D, louvain_c, weight='weight')

# Comunidades
D_nodes = dict(nx.nodes(D))
nc = 0
for i in range(0,len(louvain_c)):
    for j in list(louvain_c[i]):
        for k in list(D_nodes.keys()):
            if j == k:
                D_nodes[k] = nc
    nc = nc + 1
    print('Community',i+1,':', sorted(louvain_c[i]))

def communities_nodes_quant (louvain_communities):
#retorna uma lista onde o indice representa a quantidade de nós
#e o valor a quantidade de comunidades
    nc = []
    nodes_communities = []
    for community in louvain_communities:
        nc.append(len(community))
    for i in range(24): 
        nodes_communities.append(nc.count(i))
    df_nc = pd.DataFrame(nodes_communities, 
                         index=pd.Index(data=np.array(range(24)), name='Qtd. nós'), 
                         columns=['Qtd. comunidades'])
    return df_nc
nodes_communities = communities_nodes_quant (louvain_com)
#nodes_communities.to_excel('C:\Romulo\Modelagem de Redes Complexas\comunidades.xlsx')

### SIMULAÇÃO DE ATAQUE E FALHA

# #Remoção de 1% dos nós
# #Simulating random attacks
# initial_rand, frac_rand, apl_rand = nx_r.simulate_random_attack(G=D, attack_fraction=0.01, weight='weight')

# #Simulating degree centrality targeted attacks
# initial_degree, frac_degree, apl_degree = nx_r.simulate_degree_attack(G=D, attack_fraction=0.01, weight='weight')

# #Simulating betweenness centrality targeted attacks
# initial_between, frac_between, apl_between = nx_r.simulate_betweenness_attack(G=D, attack_fraction=0.01, weight='weight', normalized=True, k=None, seed=None, endpoints=False)

# #Simulating closeness centrality targeted attacks
# initial_close, frac_close, apl_close = nx_r.simulate_closeness_attack(G=D, attack_fraction=0.01, weight='weight', u=None, wf_improved=True)

# #Simulating closeness centrality targeted attacks
# apl_eigen = nx_r.simulate_eigenvector_attack(G=D, attack_fraction=0.01, weight='weight', tol=1e-06, max_iter=100, nstart=None)

#Remoção de 10% dos nós (retorna apenas o primeiro e último valor de apl)
def random_attack(G=None, attack_fraction=0.1, weight=None):
    """
    Simulate random attack on a network
    :param G: networkx graph
    :param attack_fraction: fraction of nodes to be attacked (default: 0.1)
    :param weight: weight of edges (default: None)
    :return: initial (float), frac (list), apl (list)
    """
    # copy the graph to avoid changing the original graph
    G = G.copy()
    # get the  number of nodes
    G_nodes = G.number_of_nodes()
    # get the largest connected component of G
    if G.is_directed():
        Gc = G.to_undirected().subgraph(sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)[0])
    else:
        Gc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    # initialize lists
    apl = []
    frac = []
    # get the first average path length of the largest connected component 
    apl.append(nx.average_shortest_path_length(Gc, weight=weight))
    # simulate random attack
    for i in range(0, int(G_nodes * attack_fraction)):
        G.remove_node(rd.choice(list(G.nodes())))
    # get the largest connected component of G
    if G.is_directed():
        Gc = G.to_undirected().subgraph(sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)[0])
    else:
        Gc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    # get the number of nodes in the largest connected component
    Gc_nodes = Gc.number_of_nodes()
    # get the fraction of nodes in the largest connected component
    frac.append(Gc_nodes / G_nodes)
    # get the last average path length of the largest connected component    
    apl.append(nx.average_shortest_path_length(Gc, weight=weight))
    return apl, frac
apl_rand_10, frac_10 = random_attack(G=D, attack_fraction=0.1, weight='weight')

def degree_attack(G=None, attack_fraction=0.1, weight=None):
    """
    Simulate degree attack on a network
    :param G: networkx graph
    :param attack_fraction: fraction of nodes to be attacked (default: 0.1)
    :param weight: weight of edges (default: None)
    :return: initial (float), frac (list), apl (list)
    """
    # copy the graph to avoid changing the original graph
    G = G.copy()
    # get the  number of nodes
    G_nodes = G.number_of_nodes()
    # get the largest connected component of G
    if G.is_directed():
        Gc = G.to_undirected().subgraph(sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)[0])
    else:
        Gc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    # initialize lists
    apl = []
    # get the first average path length of the largest connected component
    apl.append(nx.average_shortest_path_length(Gc, weight=weight))
    #get the degree of each node
    degree = nx.degree_centrality(G)
    # sort the nodes by degree
    degree = sorted(degree, key=degree.get, reverse=True)
    # simulate degree attack
    for i in range(0, int(G_nodes * attack_fraction)):
        # remove the node with the highest degree
        G.remove_node(degree[i])
    # get the largest connected component of G
    if G.is_directed():
        Gc = G.to_undirected().subgraph(sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)[0])
    else:
        Gc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    # get the last average path length of the largest connected component
    apl.append(nx.average_shortest_path_length(Gc, weight=weight))
    return apl
apl_degree_10 = degree_attack(G=D, attack_fraction=0.1, weight='weight')

def betweenness_attack(G=None, attack_fraction=0.1, weight=None, normalized=True, k=None, seed=None, endpoints=False):
    """
    Simulate betweenness attack on a network
    :param G: networkx graph
    :param attack_fraction: fraction of nodes to be attacked (default: 0.1)
    :param weight: weight of edges (default: None)
    :param normalized: if True, betweenness is normalized by 2/((n-1)(n-2)) for graphs, and 1/((n-1)(n-2)) for directed graphs where n is the number of nodes in G (default: True)
    :param k: use k node samples to estimate betweenness (default: None)
    :param seed: seed for random number generator (default: None)
    :param endpoints: If True include the endpoints in the shortest path counts (default: False)
    :return: initial (float), frac (list), apl (list)
    """
    # copy the graph to avoid changing the original graph
    G = G.copy()
    # get the  number of nodes
    G_nodes = G.number_of_nodes()
    # get the largest connected component of G
    if G.is_directed():
        Gc = G.to_undirected().subgraph(sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)[0])
    else:
        Gc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    # initialize lists
    apl = []
    # get the first average path length of the largest connected component
    apl.append(nx.average_shortest_path_length(Gc, weight=weight))
    # get the betweenness of each node
    betweenness = nx.betweenness_centrality(G, weight=weight, normalized=normalized, k=k, seed=seed, endpoints=endpoints)
    # sort the nodes by betweenness
    betweenness = sorted(betweenness, key=betweenness.get, reverse=True)
    # simulate betweenness attack
    for i in range(0, int(G_nodes * attack_fraction)):
        # remove the node with the highest betweenness
        G.remove_node(betweenness[i])
    # get the largest connected component of G
    if G.is_directed():
        Gc = G.to_undirected().subgraph(sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)[0])
    else:
        Gc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    # get the last average path length of the largest connected component
    apl.append(nx.average_shortest_path_length(Gc, weight=weight))
    return apl
apl_between_10 = betweenness_attack(G=D, attack_fraction=0.1, weight='weight', normalized=True, k=None, seed=None, endpoints=False)

def closeness_attack(G=None, attack_fraction=0.1, weight=None, u=None, wf_improved=True):
    """
    Simulate closeness attack on a network
    :param G: networkx graph
    :param attack_fraction: fraction of nodes to be attacked (default: 0.1)
    :param weight: weight of edges (default: None)
    :param u: node for which closeness is to be computed (default: None)
    :param wf_improved: use of the improved algorithm to scale by the fraction of nodes reachable (default: True)
    :return: initial (float), frac (list), apl (list)
    """
    # copy the graph to avoid changing the original graph
    G = G.copy()
    # get the  number of nodes
    G_nodes = G.number_of_nodes()
    # get the largest connected component of G
    if G.is_directed():
        Gc = G.to_undirected().subgraph(sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)[0])
    else:
        Gc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    # initialize lists
    apl = []
    # get the first average path length of the largest connected component
    apl.append(nx.average_shortest_path_length(Gc, weight=weight))
    # get the closeness of each node
    closeness = nx.closeness_centrality(G, distance=weight, u=u, wf_improved=wf_improved)
    # sort the nodes by closeness
    closeness = sorted(closeness, key=closeness.get, reverse=True)
    # simulate closeness attack
    for i in range(0, int(G_nodes * attack_fraction)):
        # remove the node with the highest closeness
        G.remove_node(closeness[i])
    # get the largest connected component of G
    if G.is_directed():
        Gc = G.to_undirected().subgraph(sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)[0])
    else:
        Gc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    # get the last average path length of the largest connected component
    apl.append(nx.average_shortest_path_length(Gc, weight=weight))
    return apl
apl_close_10 = closeness_attack(G=D, attack_fraction=0.1, weight='weight', u=None, wf_improved=True)

def eigenvector_attack(G=None, attack_fraction=0.1, weight=None, tol=1e-06, max_iter=100, nstart=None):
    """
    Simulate eigenvector attack on a network
    :param G: networkx graph
    :param attack_fraction: fraction of nodes to be attacked (default: 0.1)
    :param weight: weight of edges (default: None)
    :param tol: tolerance for the power iteration method (default: 1e-06)
    :param max_iter: maximum number of iterations for the power iteration method (default: 100)
    :param nstart: initial vector for the power iteration method (default: None)
    :return: initial (float), frac (list), apl (list)
    """
    # copy the graph to avoid changing the original graph
    G = G.copy()
    # get the  number of nodes
    G_nodes = G.number_of_nodes()
    # initialize lists
    apl = []
    # get the eigenvector of each node
    eigenvector = nx.eigenvector_centrality(G, weight=weight, tol=tol, max_iter=max_iter, nstart=nstart)
    # sort the nodes by eigenvector
    eigenvector = sorted(eigenvector, key=eigenvector.get, reverse=True)
    # simulate eigenvector attack
    for i in range(0, int(G_nodes * attack_fraction)):
        # remove the node with the highest eigenvector
        G.remove_node(eigenvector[i])
    # get the largest connected component of G
    if G.is_directed():
        Gc = G.to_undirected().subgraph(sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)[0])
    else:
        Gc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    # get the last average path length of the largest connected component
    apl.append(nx.average_shortest_path_length(Gc, weight=weight))
    return apl
apl_eigen_10 = eigenvector_attack(G=D, attack_fraction=0.1, weight='weight', tol=1e-06, max_iter=100, nstart=None)

def molloy_reed(G=None):
    """
    Compute the Molloy-Reed criterion for a network
    :param G: networkx graph
    :return: Molloy-Reed criterion
    """
    # get the average squared degree
    avg_sq_degree = sum([d ** 2 for n, d in G.degree()]) / G.number_of_nodes()
    # get the average degree
    avg_degree = sum([d for n, d in G.degree()]) / G.number_of_nodes()
    # compute the Molloy-Reed criterion
    molloy_reed = avg_sq_degree/avg_degree
    return molloy_reed
mr = molloy_reed(G=D)

def critical_threshold(G=None):
    """
    Compute the critical threshold for a network
    :param G: networkx graph
    :return: critical threshold
    """
    # get the average squared degree
    avg_sq_degree = sum([d ** 2 for n, d in G.degree()]) / G.number_of_nodes()
    # get the average degree
    avg_degree = sum([d for n, d in G.degree()]) / G.number_of_nodes()
    # compute the Molloy-Reed criterion
    molloy_reed = avg_sq_degree/avg_degree
    # compute the critical threshold
    critical_threshold = 1 - (1/(molloy_reed-1))
    return critical_threshold
ct = critical_threshold(G=D)



################################################################



### MODELO ALEATÓRIO
#p = m*2 / (n*(n-1)) #probabilidade de ligação entre os nós
R_weighted = nx.gnm_random_graph(n, m, directed=True)

# Determinação dos pesos (aleatório)
pesos_dict = nx.get_edge_attributes(D, 'weight')
pesos = list(pesos_dict.values())
rd.shuffle(pesos) #embaralha aleatoriamente a lista pesos
def random_weighted_network(rand_net, weights):
    i=0
    for (u, v) in rand_net.edges():
        while i < 4313:
            rand_net.edges()[u,v]['weight'] = weights[i]
            break
        i=i+1
    return rand_net
random_weighted_network(R_weighted, pesos)
R = nx.gnm_random_graph(n, m, directed=True)

# Grau
R_degree_dict = dict(nx.degree(R))
R_degree = list(R_degree_dict.values())

R_array = np.array(R_degree)
R_degree_mean = np.mean(R_array)

# Distribuição de graus
# sns.kdeplot(data = R_degree, fill = True, cut = 0)
# plt.title("Degree distribution (random)")

# Distribuição de graus (histograma)
bin_edges = [2,4,6,8,10,12,14]
sns.set()
plt.hist(D_degree, histtype='barstacked', bins = bin_edges)
plt.hist(R_degree, histtype='barstacked', bins = bin_edges)
plt.legend(labels=["Original", "Random"], title = "Network model")
plt.ylabel("Frequencie")
plt.title("Degree histogram")

sns.set()
sns.kdeplot(data = (D_degree, R_degree), fill=True, clip=(0,15))
plt.legend(labels=["Random","Original"], title = "Network model")
plt.title("Degree distribution")

# Coeficiente de clusterização
transit_R_weighted = nx.transitivity(R_weighted) #metrica global
transit_R = nx.transitivity(R)
avg_clust_R_weighted = nx.average_clustering(R_weighted, weight='weight') #metrica local
avg_clust_R = nx.average_clustering(R)
#calcular clustering de cada nó e comparar original e aleatório

# Centralidade
density_R_weighted = nx.density(R_weighted)

# Algoritmo de Louvain
# Modularidade calculada sem biblioteca
louvain_com_R = nx.community.louvain_communities(D) #lista de sets
R_nodes = dict(nx.nodes(R))
R_communities = communities(R_nodes, louvain_com_R)
# modularity_R = modularity(R, R_communities) #falta matriz de adjacência
#gerar matriz de adjacências para R
#não considera peso dos arcos

# Modularidade calculada com biblioteca (Networkx)
#precisa mudar para lista de frozensets ou lista de tuples
louvain_c_R = [
                frozenset(community) 
                for community in louvain_com_R
              ]
modularity_nx_R = nx.community.modularity(D, louvain_c_R, weight='weight')

# Comunidades
R_nodes = dict(nx.nodes(D))
nc = 0
for i in range(0,len(louvain_c_R)):
    for j in list(louvain_c_R[i]):
        for k in list(R_nodes.keys()):
            if j == k:
                R_nodes[k] = nc
    nc = nc + 1
    print('Community',i+1,':', sorted(louvain_c_R[i]))