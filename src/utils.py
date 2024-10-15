import dgl
import pickle
import numpy as np
import random
import scipy
from sklearn.cluster import SpectralBiclustering
from collections import defaultdict
import math

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


# This method counts the n_i(x, y) for the FOL and Hybrid formulas
def count_ground_truths(edges_tuple, num_classes, num_clusters, node_to_cluster_idx, dist_soft_lt_inequality, labels):
    G_FOL = np.zeros(shape=(num_clusters, num_classes))
    G_HFOL = np.zeros(shape=(num_clusters, num_classes))
    for i, edge in enumerate(edges_tuple):
        if str(edge) in node_to_cluster_idx:
            x = edge[0]
            y = edge[1]
            cid = node_to_cluster_idx[str(edge)]
            for k in range(num_classes):
                Cx = labels[x] == k
                Cy = labels[y] == k
                ## Class_k(edge[0]) < = > CLass_k(edge[1])
                ##(¬Cy ∨ Cx) ∧ (¬Cx ∨ Cy)
                satisfies = (not Cy or Cx) and (not Cx or Cy)
                if satisfies:
                    G_FOL[cid, k] += 1
                    G_HFOL[cid, k] += dist_soft_lt_inequality[x][y][k]
    print("G_FOL sum = ", G_FOL.sum())
    print("G_HFOL sum = ", G_HFOL.sum())
    return G_FOL, G_HFOL


def estimate_ground_values(edges_tuple, num_classes, num_clusters, node_to_cluster_idx, auxiliary_vars,
                           auxiliary_lt_softineq_vars, dist_soft_lt_inequality):
    E_FOL = np.zeros(shape=(num_clusters, num_classes))
    E_HFOL = np.zeros(shape=(num_clusters, num_classes))

    for eid, edge in enumerate(edges_tuple):
        if str(edge) in node_to_cluster_idx:
            i = edge[0]
            j = edge[1]
            cid = node_to_cluster_idx[str(edge)]
            for k in range(num_classes):
                E_FOL[cid, k] += auxiliary_vars[eid, k].X
                E_HFOL[cid, k] += (auxiliary_lt_softineq_vars[eid, k].X * dist_soft_lt_inequality[i][j][k])
    print("E_FOL ground truth sum = ", E_FOL.sum())
    print("E_HFOL ground value sum = ", E_HFOL.sum())
    return E_FOL, E_HFOL


def cluster(num_nodes, num_classes, num_clusters, distance_matrix_np, random_state=16):
    distance_matrix_trunc = distance_matrix_np[:num_nodes, :num_nodes]

    # Groupings for the Hybrid formulas
    clustering = SpectralBiclustering(n_clusters=num_clusters, random_state=random_state).fit(distance_matrix_trunc)

    supernode_dict = dict()
    node_to_cluster_idx = dict()
    total_num_links = 0
    for cluster_id in range(num_clusters*num_clusters):
        cluster_shape = clustering.get_shape(cluster_id)
        cluster_indices = clustering.get_indices(cluster_id)
        cluster_members = []
        for row in cluster_indices[0]:
            for column in cluster_indices[1]:
                mem = (row,column)
                cluster_members.append(mem)
                node_to_cluster_idx[str(mem)] = cluster_id
                total_num_links += 1
        supernode_dict[cluster_id] = cluster_members
    print("Total  = ", total_num_links)
    return supernode_dict, node_to_cluster_idx, clustering


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


def calculate_accuracy(class_vars, graph, num_classes, num_nodes=500):
    correct = 0

    for j in range(0, num_nodes):
        for i in range(num_classes):
            if class_vars[j, i].X > 0:
                if i == graph.ndata["label"][j].item():
                    correct += 1
    print("Accuracy = ", (correct / num_nodes) * 100)


def cluster_edge_count(supernode_dict, edges_tuple):
    N_fol = defaultdict(int)
    N_hfol = defaultdict(int)

    for cid, edge_list in supernode_dict.items():
        for edge in edge_list:
            if edge in edges_tuple:
                N_fol[cid] += 1
                N_hfol[cid] += 1
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


def get_clusterwise_scale_factors(dist_soft_lt_inequality, clusters, num_classes=6):
    #
    # Input: the matrix of the soft inequality values for each formulas. Dimension: [n x n x n_f]
    #        where n is the number of nodes, n_f is the number of formulas
    # Returns: A dictionary where key is the cluster ID and value is the scale factors for each formula
    #

    clusterwise_predicate_connections = {}
    for cluster_id, cluster_pairs in clusters.items():
        result_dict_1 = {}
        result_dict_2 = {}
        for pair in cluster_pairs:
            first_pos, second_pos = pair
            if first_pos not in result_dict_1:
                result_dict_1[first_pos] = []
            if second_pos not in result_dict_2:
                result_dict_2[second_pos] = []
            result_dict_1[first_pos].append(second_pos)
            result_dict_2[second_pos].append(first_pos)
        clusterwise_predicate_connections[cluster_id] = [result_dict_1, result_dict_2]

    clusterwise_scale_factors = dict()
    for cluster_id in clusters:
        connection_vectors = list()
        for fmla in range(num_classes):
            conn_vector = list()
            for predicate in range(2):
                pred_val_connection = list()
                for node_1, pairs in clusterwise_predicate_connections[cluster_id][predicate].items():
                    summ = 0
                    for node_2 in pairs:
                        summ += dist_soft_lt_inequality[node_1][node_2][fmla]
                    pred_val_connection.append(round(summ, 3))
                conn_vector.append(max(pred_val_connection))
            connection_vectors.append(conn_vector)
        scale_factors = [max(v) for v in connection_vectors]
        clusterwise_scale_factors[cluster_id] = scale_factors

    return clusterwise_scale_factors

def initialize_distances(num_nodes, supernode_fol_weights, supernode_hybrid_weights,
                         node_to_cluster_idx, edges_tuple, num_classes):

    formula_weights = np.zeros(shape=(len(edges_tuple), num_classes))
    hybrid_formula_weights = np.zeros(shape=(len(edges_tuple), num_classes))

    for eid, edge in enumerate(edges_tuple):
        for k in range(num_classes):
            w1 = round(random.uniform(0, 1), 2)
            w2 = round(random.uniform(0, 1), 2)
            if str(edge) in node_to_cluster_idx:
                cid = node_to_cluster_idx[str(edge)]
                w1 = supernode_fol_weights[cid][k]
                w2 = supernode_hybrid_weights[cid][k]
            formula_weights[eid, k] = w1
            hybrid_formula_weights[eid, k] = w2

    return formula_weights, hybrid_formula_weights

def initialize_soft_inequalities(num_nodes, distance_matrix, num_classes):
    dist_soft_lt_inequality = []
    dist_soft_gt_inequality = []

    for i in range(num_nodes):
        temp_lt_row = []
        temp_gt_row = []

        for j in range(num_nodes):

            temp_lt_third_dim = []
            temp_gt_third_dim = []

            distance = distance_matrix[i][j]  # embedding distance between nodes i and j

            degree_lt = SOFTNESS * (distance - THRESHOLD)
            degree_gt = SOFTNESS * (THRESHOLD - distance)
            d_lt = -math.log(1 + math.exp(degree_lt))
            d_gt = -math.log(1 + math.exp(degree_gt))

            for k in range(num_classes):
                temp_lt_third_dim.append(round(d_lt, 3))
                temp_gt_third_dim.append(round(d_gt, 3))
            temp_lt_row.append(temp_lt_third_dim)
            temp_gt_row.append(temp_gt_third_dim)
        dist_soft_lt_inequality.append(temp_lt_row)
        dist_soft_gt_inequality.append(temp_gt_row)

    return dist_soft_lt_inequality, dist_soft_gt_inequality
