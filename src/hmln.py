from gurobipy import *
import gurobipy as gb
from gurobipy import GRB
import numpy as np
import random
import math
from time import time
from scipy.stats import norm
from src import utils


class CitationHMLN(object):
    def __init__(self, dataset, num_nodes, num_clusters, sub_symbolic_distances, pred_probabilities):
        self.t_model = gb.Model(name=dataset)
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.distance_matrix = sub_symbolic_distances
        self.pred_probabilities = pred_probabilities
        self.graph, self.dataset_full = utils.load_dataset(dataset)
        self.num_classes = self.dataset_full.num_classes
        self.edges_tuple = utils.get_edges_tuple(self.graph, self.num_nodes)
        self.supernode_dict, self.node_to_cluster_idx, self.clustering = utils.cluster(self.num_nodes, self.num_classes,
                                                                      self.num_clusters, sub_symbolic_distances)
        # self.supernode_fol_dict, self.fol_node_to_cluster_idx = utils.cluster_fol(self.edges_tuple)
        self.N_fol, self.N_hfol = utils.cluster_edge_count(self.supernode_dict, self.edges_tuple)
        self.supernode_fol_weights, self.supernode_hybrid_weights = utils.initialize_weights(self.num_clusters * self.num_clusters,
                                                                                             self.num_classes)

    def train(self, num_iters):
        time_1 = time()
        dist_soft_lt_inequality, dist_soft_gt_inequality = utils.initialize_soft_inequalities(self.num_nodes, self.distance_matrix, self.num_classes)
        G_FOL, G_HFOL = utils.count_ground_truths(self.edges_tuple, self.num_classes,
                                                  self.num_clusters * self.num_clusters, self.node_to_cluster_idx,
                                                  dist_soft_lt_inequality, self.graph.ndata["label"].numpy())

        for it_num in range(num_iters):
            formula_weights, hybrid_formula_weights = utils.initialize_distances(self.num_nodes, self.supernode_fol_weights,
                                                                                 self.supernode_hybrid_weights, self.node_to_cluster_idx,
                                                                                 self.edges_tuple, self.num_classes)

            self.construct_optimization_model(formula_weights, hybrid_formula_weights, dist_soft_lt_inequality)
            self.t_model.setParam('MIPGap', 0.1)
            self.t_model.Params.LogToConsole = 1
            # self.t_model.setParam("")
            self.t_model.optimize()

            E_FOL, E_HFOL = utils.estimate_ground_values(self.edges_tuple, self.num_classes, self.num_clusters * self.num_clusters,
                                                   self.node_to_cluster_idx, self.auxiliary_vars, self.auxiliary_lt_softineq_vars,
                                                   dist_soft_lt_inequality)

            delta = E_FOL - G_FOL
            delta_HFOL = E_HFOL - G_HFOL

            # Delta weights for FOL
            epsilon = 0.1  # Learning Rate

            dw_fol = np.zeros(shape=(self.num_clusters * self.num_clusters, self.num_classes))
            dw_hfol = np.zeros(shape=(self.num_clusters * self.num_clusters, self.num_classes))
            for i in range(self.num_clusters * self.num_clusters):
                for j in range(self.num_classes):
                    if delta_HFOL[i, j] != 0:
                        dw = (epsilon * delta_HFOL[i, j]) / self.N_hfol[i]
                        dw_hfol[i, j] += dw

            for i in range(self.num_clusters * self.num_clusters):
                for j in range(self.num_classes):
                    if delta[i, j] != 0:
                        dw = (epsilon * delta[i, j]) / self.N_fol[i]
                        dw_fol[i, j] += dw

            print("DW_FOL sum = ", np.abs(dw_fol).sum())
            print("DW_HFOL sum = ", np.abs(dw_hfol).sum())
            utils.calculate_accuracy(self.class_vars, self.graph, self.num_classes, self.num_nodes)

            if np.abs(dw_fol).sum() == 0 and np.abs(dw_hfol).sum() == 0:
                break

            self.supernode_fol_weights = self.supernode_fol_weights - dw_fol
            self.supernode_hybrid_weights = self.supernode_hybrid_weights - dw_hfol

        time_2 = time()
        print("Total time spent for training the specification HMLN == ", time_2 - time_1, " secs")

    def construct_optimization_model(self, formula_weights, hybrid_formula_weights, dist_softineq_vars):
        self.t_model.Params.LogToConsole = 0

        self.class_vars = self.t_model.addVars(self.num_nodes, self.num_classes, vtype=GRB.BINARY, name="class_vars")
        self.auxiliary_lt_softineq_vars = self.t_model.addVars(len(self.edges_tuple), self.num_classes, vtype=GRB.BINARY, name="auxiliary_lt_vars")
        self.auxiliary_vars = self.t_model.addVars(len(self.edges_tuple), self.num_classes, vtype=GRB.BINARY, name="auxiliary_vars")

        ##### Adding Constraints for each link  for        N * [(Cx <-> Cy) <-> K]
        for idx, link in enumerate(self.edges_tuple):
            node_1 = link[0]
            node_2 = link[1]

            for j in range(self.num_classes):
                self.t_model.addConstr(
                    self.class_vars[node_2, j] + self.auxiliary_vars[idx, j] - self.class_vars[node_1, j] <= 1)
                self.t_model.addConstr(
                    self.class_vars[node_1, j] + self.auxiliary_vars[idx, j] - self.class_vars[node_2, j] <= 1)
                self.t_model.addConstr(
                    self.class_vars[node_2, j] - self.auxiliary_vars[idx, j] + self.class_vars[node_1, j] <= 1)
                self.t_model.addConstr(
                    self.class_vars[node_2, j] + self.auxiliary_vars[idx, j] + self.class_vars[node_1, j] >= 1)

                self.t_model.addConstr(
                    self.class_vars[node_2, j] + self.auxiliary_lt_softineq_vars[idx, j] - self.class_vars[node_1, j] <= 1)
                self.t_model.addConstr(
                    self.class_vars[node_1, j] + self.auxiliary_lt_softineq_vars[idx, j] - self.class_vars[node_2, j] <= 1)
                self.t_model.addConstr(
                    self.class_vars[node_2, j] - self.auxiliary_lt_softineq_vars[idx, j] + self.class_vars[node_1, j] <= 1)
                self.t_model.addConstr(
                    self.class_vars[node_2, j] + self.auxiliary_lt_softineq_vars[idx, j] + self.class_vars[node_1, j] >= 1)

        for i in range(self.num_nodes):
            self.t_model.addConstr(gb.quicksum(self.class_vars[i, j] for j in range(self.num_classes)) == 1,
                                   name="node_class_%d" % i)

        self.t_model.ModelSense = GRB.MAXIMIZE
        self.t_model.setObjective(gb.quicksum([gb.quicksum([self.auxiliary_vars[i, j] * formula_weights[i, j]
                                                       for i, edge in enumerate(self.edges_tuple) for j in
                                                       range(self.num_classes)]),
                                          gb.quicksum([0.25 * self.class_vars[i, j] * self.pred_probabilities[i][j] for i in
                                                       range(self.num_nodes) for j in range(self.num_classes)]),
                                          gb.quicksum([self.auxiliary_lt_softineq_vars[eid, k] * hybrid_formula_weights[
                                              eid, k] * dist_softineq_vars[i][j][k]
                                                       for eid, (i, j) in enumerate(self.edges_tuple) for k in
                                                       range(self.num_classes)]),
                                          ]),
                             GRB.MAXIMIZE)

        self.t_model.update()

    def estimate_ground_truths(self):
        E_FOL = np.zeros(shape=(100, self.num_classes))
        for i, edge in enumerate(self.edges_tuple):
            if str(edge) in self.fol_node_to_cluster_idx:
                cid = self.fol_node_to_cluster_idx[str(edge)]
                for k in range(self.num_classes):
                    E_FOL[cid, k] += self.auxiliary_vars[k, i].X
        print("E_FOL sum = ", E_FOL.sum())

        E_HFOL = np.zeros(shape=(self.num_clusters, self.num_classes))
        for cid, edge_list in self.supernode_dict.items():
            for edge in edge_list:
                if str(edge) in self.node_to_cluster_idx:
                    i = edge[0]
                    j = edge[1]
                    for k in range(self.num_classes):
                        E_HFOL[cid, k] += self.auxiliary_lt_softineq_vars[i, j, k].X
        print("E_HFOL sum = ", E_HFOL.sum())
        return E_FOL, E_HFOL



