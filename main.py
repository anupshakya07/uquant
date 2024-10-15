import argparse
from src.hmln import CitationHMLN
import numpy as np
from src import utils


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-num_nodes", type=int, default=200, help="Number of nodes to take in training")
    parser.add_argument("-num_clusters", type=int, default=10, help="Number of weight-sharing clusters to form")
    parser.add_argument("-num_iters", type=int, default=100, help="Number of iterations to train the HMLN")
    parser.add_argument("-dataset", type=str, default="cora", choices=["cora", "citeseer"],
                        help="Choose the dataset name")

    opt = parser.parse_args()
    print(opt)
    num_nodes = opt.num_nodes
    num_clusters = opt.num_clusters
    num_iters = opt.num_iters
    dataset = opt.dataset


    embedding_filepath = f"representations/{dataset}/gcn_specification_embeddings.pkl"
    pred_filepath = f"representations/{dataset}/gcn_specification_pred_prob.pkl"
    distance_matrix_filepath = f"representations/{dataset}/gcn_specification_trunc_distance_matrix.pkl"

    distance_matrix = utils.load_file(distance_matrix_filepath)
    pred_probabilities = utils.load_file(pred_filepath)

    citation_hmln = CitationHMLN(dataset, num_nodes, num_clusters, np.array(distance_matrix), pred_probabilities)
    citation_hmln.train(num_iters)

    return


if __name__ == "__main__":
    main()
