import pandas as pd
import numpy as np


class Taxonomy:
    """
    A total iNat taxonomy. Not complete in the sense that it covers the
    whole tree of life, but complete in the sense that it represents the
    full taxonomy for a vision export.
    """

    def __init__(self, taxonomy_csv_file):
        self.load_taxonomy(taxonomy_csv_file)

    def load_taxonomy(self, taxonomy_csv_file):
        """
        load the taxonomy from a csv file. grafts each of the kingdoms
        onto a synthetic node called life. also creates some helper lookup
        dictionaries.
        """
        self.tax = pd.read_csv(taxonomy_csv_file)
        self.leaf_tax = self.tax.dropna(subset=["leaf_class_id"])
        self.cv_class_to_tax = [
            x[1]
            for x in self.leaf_tax.apply(
                lambda x: (x["leaf_class_id"], x["taxon_id"]), axis=1
            )
        ]

        self.nodes = []
        self.nodes_by_taxon_id = {}
        self.nodes_by_leaf_class_id = {}

        self.life = Node(
            name="Life",
            taxon_id=48460,
            leaf_class_id=None,
            parent=None,
            children=set(),
            rank_level=100,
        )
        self.nodes.append(self.life)
        self.nodes_by_taxon_id[48460] = self.life

        for i, t in self.tax.iterrows():
            n = Node(
                name=t["name"],
                taxon_id=t["taxon_id"],
                leaf_class_id=t["leaf_class_id"],
                parent=t["parent_taxon_id"],
                children=set(),
                rank_level=t["rank_level"],
            )
            self.nodes.append(n)
            self.nodes_by_taxon_id[t["taxon_id"]] = n

            if n.leaf_class_id is not None:
                self.nodes_by_leaf_class_id[n.leaf_class_id] = n

        for node in self.nodes:
            # graft kingdoms onto life
            if node.parent is None:
                if node.taxon_id != 48460:
                    node.parent = 48460
                    self.life.children.add(node.taxon_id)
            if node.taxon_id != 48460:
                # attach parents to children
                parent_node = self.nodes_by_taxon_id[node.parent]
                parent_node.children.add(node.taxon_id)

    def aggregate_scores(self, scores, node):
        """
        given a list of scores, where each index in the list corresponds
        to the leaf class id of a node in the taxonomy, aggregate the scores
        up the tree under the node argument. typically this node argument
        is set to the life node.
        for example, the scores for Genus Xyz will be the sum of the
        scores for all of the species beneath Genus Xyz. because the
        scores are typically normalized (due to softmax in vision, or
        explicit normalization when combining vision and geo scores),
        the life score will be 1.0.
        """
        if node.leaf_class_id is None:
            # calculate for descendents
            all_scores = {}
            for child_taxon_id in node.children:
                child_node = self.nodes_by_taxon_id[child_taxon_id]
                child_scores = self.aggregate_scores(scores, child_node)
                all_scores.update(child_scores)

            this_node_score = 0
            for child_taxon_id in node.children:
                this_node_score += all_scores[child_taxon_id]

            all_scores[node.taxon_id] = this_node_score

            return all_scores
        else:
            # calculate for this leaf node
            return {node.taxon_id: scores[node.leaf_class_id]}

    def best_branch_from_scores(self, scores):
        """
        once the scores are aggregated, so every node in the tree has a score,
        including inner nodes, then we can calculate the best branch.
        """
        best_branch = []

        life_node = self.nodes_by_taxon_id[48460]
        life_score = scores[life_node.taxon_id]

        best_branch.append((life_node, life_score))

        current_node = life_node
        while current_node.leaf_class_id is None:
            # find the best child of the current node
            best_child_taxon_id = -1
            best_child_score = -1
            for child_taxon_id in current_node.children:
                child_score = scores[child_taxon_id]
                if child_score > best_child_score:
                    best_child_taxon_id = child_taxon_id
                    best_child_score = child_score

            best_child_node = self.nodes_by_taxon_id[best_child_taxon_id]

            best_branch.append((best_child_node, best_child_score))
            current_node = best_child_node

        return best_branch


class Node:
    """
    A single node in the iNat taxonomy.
    """

    def __init__(self, name, taxon_id, leaf_class_id, parent, children, rank_level):
        self.taxon_id = taxon_id

        if leaf_class_id is None:
            self.leaf_class_id = None
        elif np.isnan(leaf_class_id):
            self.leaf_class_id = None
        else:
            self.leaf_class_id = int(leaf_class_id)

        if parent is None:
            self.parent = None
        elif np.isnan(parent):
            self.parent = None
        else:
            self.parent = int(parent)

        self.children = children
        self.name = name
        self.rank_level = rank_level

    def __repr__(self):
        return "Node <Name: {} {}, Taxon Id: {}, Leaf Class Id: {}, Parent Id: {}, Children: {}>".format(
            self.rank_level,
            self.name,
            self.taxon_id,
            self.leaf_class_id,
            self.parent,
            self.children,
        )
