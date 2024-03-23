import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_network(country):
    df = pd.read_csv(f'./MulitipleCountry/IPC{country}.csv')
    df['IPC'] = df['IPC'].apply(lambda x: [i.strip() for i in x.split('|')])
    G = nx.Graph()
    for ipc_list in df['IPC']:
        for node in ipc_list:
            if node not in G:
                G.add_node(node, weight=0)
            G.nodes[node]['weight'] += 1
        for i in range(len(ipc_list)):
            for j in range(i+1, len(ipc_list)):
                if G.has_edge(ipc_list[i], ipc_list[j]):
                    G[ipc_list[i]][ipc_list[j]]['weight'] += 1
                else:
                    G.add_edge(ipc_list[i], ipc_list[j], weight=1)

    if not os.path.exists(f'./{country}IPCNetwork'):
        os.makedirs(f'./{country}IPCNetwork')
    nx.write_pajek(G, f'./{country}IPCNetwork/{country}IPCNetwork.net')

    # 输出网络的节点数和边数
    print("节点数:", G.number_of_nodes())
    print("边数:", G.number_of_edges())
    
    return G

if __name__ == '__main__':
    country_list = ['US','JP','CN','GB','EP']
    for country in country_list:
        G = load_network(country) 
        # 计算重要性指标
        centrality_measures = {
            'degree_centrality': nx.degree_centrality(G),
            'betweenness_centrality': nx.betweenness_centrality(G),
            'eigenvector_centrality': nx.eigenvector_centrality(G, max_iter=100000, tol=1e-06),
            'pagerank': nx.pagerank(G),
            'core_number': nx.core_number(G),
            'closeness_centrality': nx.closeness_centrality(G),
        }

        # 将重要性指标的字典转换为DataFrame
        df_centrality = pd.DataFrame(centrality_measures)

        # 将索引（节点名称）添加为一列
        df_centrality['node'] = df_centrality.index

        # 重新排列列的顺序
        cols = ['node'] + [col for col in df_centrality.columns if col != 'node']
        df_centrality = df_centrality[cols]

        # 保存为CSV文件
        df_centrality.to_csv(f'./{country}IPCNetwork/node_centrality_measures.csv', index=False)

        print("已成功计算重要性指标并保存为CSV文件。")
        
        # degree_centrality = np.array(list(nx.degree_centrality(G).values()))

        # # 计算CDF
        # degree_centrality_sorted = np.sort(degree_centrality)
        # cdf = np.arange(1, len(degree_centrality_sorted) + 1) / len(degree_centrality_sorted)

        # # 绘制CDF图
        # plt.figure(figsize=(8, 5))
        # plt.plot(degree_centrality_sorted, cdf, marker='.', linestyle='none')
        # plt.xlabel('Degree Centrality')
        # plt.ylabel('CDF')
        # plt.title('CDF of Degree Centrality in Network')
        # plt.grid(True)
        # plt.show()
        
        # # 使用同样的degree_centrality数据
        # plt.figure(figsize=(8, 5))
        # plt.hist(degree_centrality, bins=50, alpha=0.75)
        # plt.xlabel('Degree Centrality')
        # plt.ylabel('Number of Nodes')
        # plt.title('Histogram of Degree Centrality in Network')
        # plt.grid(True)
        # plt.show()
        
        # # 计算指标间的相关性矩阵
        # corr = df_centrality.drop('node', axis=1).corr()

        # # 绘制热图
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
        # plt.title('Correlation Between Different Centrality Measures')
        # plt.xticks(rotation=45, ha='right')
        # plt.yticks(rotation=45)
        # plt.tight_layout()  # 调整布局以避免剪切
        # plt.show()