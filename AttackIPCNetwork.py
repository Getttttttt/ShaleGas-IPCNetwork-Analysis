import networkx as nx
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

class IPCNetworkAttacker():
    def __init__(self, file_path, directed=True):
        self.file_path = file_path
        self.network_name = os.path.splitext(os.path.basename(file_path))[0] + 'Undirected'
        # undirected
        self.G = nx.read_pajek(self.file_path)

        simple_G = nx.Graph()

        for u, v, data in self.G.edges(data=True):
            weight = data.get('weight', 1)
            if simple_G.has_edge(u, v):
                # 如果边已存在，累加权值
                simple_G[u][v]['weight'] += weight
            else:
                # 否则，添加边，并设置权值
                simple_G.add_edge(u, v, weight=weight)

        self.G = simple_G
        
        # 检查G是否为多重图
        if self.G.is_multigraph():
            print("G is a multigraph.")
        else:
            print("G is not a multigraph.")

        # 检查G是否为有向图
        if self.G.is_directed():
            print("G is a directed graph.")
        else:
            print("G is not a directed graph.")
        
        self.results_dir = self.create_results_directory()
        
        self.directed = False    
        
        # Prepare dictionaries for results
        self.P_infinity_results = defaultdict(dict)
        self.second_largest_cc_results = defaultdict(dict)
    
    def create_results_directory(self):
        results_dir = os.path.join(os.getcwd(), self.network_name)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        return results_dir
    
    def calculate_centrality_measures(self):
        measures = {
            'degree_centrality': nx.degree_centrality(self.G),
            'betweenness_centrality': nx.betweenness_centrality(self.G),
            'eigenvector_centrality': nx.eigenvector_centrality(self.G, max_iter=100000, tol=1e-06),
            'pagerank': nx.pagerank(self.G),
            'core_number': nx.core_number(self.G),
            'closeness_centrality': nx.closeness_centrality(self.G),
        }
        return measures

    def simulate_attack(self, importance_measure, measure_name):
        # Reset G_copy for each simulation
        G_copy = self.G.copy()
        # Initialize results storage for P_infinity and second largest component
        results = defaultdict(list)
        second_largest_cc_results = defaultdict(list)
        for f in np.arange(0.01, 1.01, 0.01):  # Adjusted to simulate a wider range of f
            G_copy = self.G.copy()
            num_to_remove = int(np.ceil(f * len(G_copy.nodes())))
            nodes_sorted_by_importance = sorted(importance_measure.items(), key=lambda x: x[1], reverse=True)
            nodes_to_remove = [n for n, _ in nodes_sorted_by_importance[:num_to_remove]]
            G_copy.remove_nodes_from(nodes_to_remove)
            
            # Calculate the largest and second largest strongly connected components
            ccs = sorted([cc for cc in nx.connected_components(G_copy)], key=len, reverse=True)
            largest_cc = ccs[0] if ccs else []
            second_largest_cc = ccs[1] if len(ccs) > 1 else []
            P_infinity = len(largest_cc) / len(self.G.nodes()) if self.G.nodes() else 0
            P_second = len(second_largest_cc) / len(self.G.nodes()) if self.G.nodes() else 0
            
            # Store the results
            results[f].append(P_infinity)
            second_largest_cc_results[f].append(P_second)
        
        # Save results to class variables for later use
        self.P_infinity_results[measure_name] = results
        self.second_largest_cc_results[measure_name] = second_largest_cc_results
    
    def plot_results(self):
        plt.figure(figsize=(14, 7))
        
        # Subplot for the largest connected component
        plt.subplot(1, 2, 1)
        for measure_name, results in self.P_infinity_results.items():
            fs = list(results.keys())
            P_infinities = [results[f][0] for f in fs]
            plt.plot(fs, P_infinities, label=measure_name)

        plt.xlabel('Fraction of Nodes Removed (f)')
        plt.ylabel('Largest Connected Component (P_infinity)')
        plt.title('Network Robustness: Largest CC'+self.network_name)
        plt.legend()

        # Subplot for the second largest connected component
        plt.subplot(1, 2, 2)
        for measure_name, results in self.second_largest_cc_results.items():
            fs = list(results.keys())
            P_second_largest = [results[f][0] for f in fs]
            plt.plot(fs, P_second_largest, label=measure_name, linestyle='--')

        plt.xlabel('Fraction of Nodes Removed (f)')
        plt.ylabel('Second Largest Connected Component')
        plt.title('Network Robustness: Second Largest CC'+self.network_name)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'Combined_CC_Results.png'))
        plt.close()
        
    def main_experiment(self):
        centrality_measures = self.calculate_centrality_measures()
        fc_values = {}

        for measure_name, measure in centrality_measures.items():
            self.simulate_attack(measure, measure_name)
        
        self.plot_results()

        # Prepare DataFrame for results
        df_P_infinity = pd.DataFrame()
        df_second_largest = pd.DataFrame()

        for measure_name in centrality_measures.keys():
            df_P_infinity[measure_name] = pd.Series([val[0] for val in self.P_infinity_results[measure_name].values()])
            df_second_largest[measure_name] = pd.Series([val[0] for val in self.second_largest_cc_results[measure_name].values()])

        # Write results to CSV
        df_P_infinity.to_csv(os.path.join(self.results_dir, 'P_infinity_results.csv'), index_label='f')
        df_second_largest.to_csv(os.path.join(self.results_dir, 'second_largest_cc_results.csv'), index_label='f')
        
        # Calculate fc values and save to CSV
        for measure_name, results in self.P_infinity_results.items():
            fc = min([f for f, P_infinity_list in results.items() if P_infinity_list[0] < 0.8], default=None)
            if fc is not None:
                fc_values[measure_name] = fc

        pd.DataFrame.from_dict(fc_values, orient='index', columns=['fc']).to_csv(os.path.join(self.results_dir, 'fc_values.csv'))

        return fc_values

if __name__ == '__main__':
    file_path = './ModifiedIPCNetwork.net'
    IPCna = IPCNetworkAttacker(file_path=file_path)
    fc_values = IPCna.main_experiment()
    print("Critical fraction values (f_c) for each centrality measure:")
    for measure, f_c in fc_values.items():
        print(f"{measure}: {f_c}")

    