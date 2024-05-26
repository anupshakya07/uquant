import dgl
import pickle
import numpy as np
import random
import scipy
from sklearn.cluster import SpectralBiclustering


SOFTNESS = 5
THRESHOLD = 0.5
MU = 0.5
SIGMA = 0.18


def load_dataset(dataset_name):
    if dataset_name == "cora":
        cora_full = dgl.data.CoraGraphDataset()
        graph = cora_full[0]
        return graph, cora_full

    elif dataset_name == "pubmed":
        pubmed_full = dgl.data.PubmedGraphDataset()
        graph = pubmed_full[0]
        return graph, pubmed_full

    elif dataset_name == "citeseer":
        citeseer_full = dgl.data.CiteseerGraphDataset()
        graph = citeseer_full[0]
        return graph, citeseer_full


def cluster(num_nodes, num_classes, num_clusters, distance_matrix_np, random_state=16):
    distance_matrix_trunc = distance_matrix_np[:num_nodes, :num_nodes]

    # Groupings for the Hybrid formulas
    clustering = SpectralBiclustering(n_clusters=num_clusters, random_state=random_state).fit(distance_matrix_trunc)

    supernode_dict = dict()
    node_to_cluster_idx = dict()
    total_num_links = 0
    for cluster_id in range(num_clusters):
        cluster_shape = clustering.get_shape(cluster_id)
        cluster_indices = clustering.get_indices(cluster_id)
        cluster_members = []
        for row in cluster_indices[0]:
            for column in cluster_indices[1]:
                mem = (row, column)
                cluster_members.append(mem)
                node_to_cluster_idx[str(mem)] = cluster_id
                total_num_links += 1
        supernode_dict[cluster_id] = cluster_members
    print("Total  = ", total_num_links)
    return supernode_dict, node_to_cluster_idx


def get_group_index(idx):
    if idx < 100:
        return 0
    elif 100 <= idx < 200:
        return 1
    elif 200 <= idx < 300:
        return 2
    elif 300 <= idx < 400:
        return 3
    elif 400 <= idx < 500:
        return 4
    elif 500 <= idx < 600:
        return 5
    elif 600 <= idx < 700:
        return 6
    elif 700 <= idx < 800:
        return 7
    elif 800 <= idx < 900:
        return 8
    elif 900 <= idx < 1000:
        return 9
    else:
        print("No Group Found for index ", idx)


def get_edges_tuple(graph, num_nodes):
    edges_tuple = []
    for i in range(graph.num_edges()):
        src = graph.edges()[0][i].item()
        dst = graph.edges()[1][i].item()
        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
            edges_tuple.append((src, dst))
    return edges_tuple


def cluster_fol(edges_tuple):
    supernode_fol_dict = dict()
    fol_node_to_cluster_idx = dict()

    for edge in edges_tuple:
        n1 = edge[0]
        n2 = edge[1]
        id1 = get_group_index(n1)
        id2 = get_group_index(n2)
        group_num = id1 * 10 + id2
        if group_num in supernode_fol_dict:
            supernode_fol_dict[group_num].append(edge)
        else:
            supernode_fol_dict[group_num] = [edge]
        fol_node_to_cluster_idx[str(edge)] = group_num
    return supernode_fol_dict, fol_node_to_cluster_idx


def calculate_accuracy(class_vars, graph, num_nodes=500):
    correct = 0

    for j in range(0, num_nodes):
        for i in range(7):
            if class_vars[j, i].X > 0:
                if i == graph.ndata["label"][j].item():
                    correct += 1
    print("Accuracy = ", (correct / num_nodes) * 100)


def cluster_edge_count(supernode_fol_dict, supernode_dict):
    N_fol = dict()
    N_hfol = dict()

    for cid, edge_list in supernode_fol_dict.items():
        N_fol[cid] = len(edge_list)

    for cid, edge_list in supernode_dict.items():
        N_hfol[cid] = len(edge_list)
    return N_fol, N_hfol


def initialize_weights(num_clusters, num_classes):
    supernode_fol_weights = np.zeros(shape=(100, num_classes))
    supernode_hybrid_weights = np.zeros(shape=(num_clusters, num_classes))

    # Adding auxiliary variables and weights for each grounding for an edge in the graph
    for i in range(num_classes):

        for j in range(num_clusters):
            weight_1 = round(random.uniform(0, 1), 2)
            supernode_fol_weights[j, i] = weight_1

        for j in range(num_clusters):
            weight_2 = round(random.uniform(0, 1), 2)
            supernode_hybrid_weights[j, i] = weight_2
    return supernode_fol_weights, supernode_hybrid_weights


def load_file(filepath):
    file = pickle.load(open(filepath, "rb"))
    return file


def compute_distance_matrix(emb_vectors, num_nodes):
    distance_matrix_np = np.zeros(shape=(num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i, num_nodes):
            dist = scipy.spatial.distance.cosine(emb_vectors[i], emb_vectors[j])
            distance_matrix_np[i, j] = round(dist, 3)
            distance_matrix_np[j, i] = round(dist, 3)
    return distance_matrix_np

def generate_query(supernode_dict):
    query_list = []
    for cluster_id, cluster_edges in supernode_dict.items():
        sampled_edge = random.sample(cluster_edges, 1)[0]
        query_list.append(sampled_edge)

    query_list_2 = []
    for cluster_id, cluster_edges in supernode_dict.items():
        sampled_edges = random.sample(cluster_edges, 2)
        sampled_edges = [x for x in sampled_edges if x not in query_list]
        query_list_2.append(sampled_edges[0])
    return query_list, query_list_2
