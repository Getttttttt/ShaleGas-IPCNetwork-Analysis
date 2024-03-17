import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def load_network():
    # 读取CSV文件
    df = pd.read_csv('DatasetIPC.csv')
    # 处理IPC列，去除空格，分割成列表
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

    nx.write_pajek(G, 'IPCNetwork.net')

    # 输出网络的节点数和边数
    print("节点数:", G.number_of_nodes())
    print("边数:", G.number_of_edges())
    
    return G

if __name__ == '__main__':
    G = load_network() 
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
    df_centrality.to_csv('node_centrality_measures.csv', index=False)

    print("已成功计算重要性指标并保存为CSV文件。")
    
    pos = nx.spring_layout(G)  # 为网络中的节点设置布局

    # 使用度中心性作为节点大小的例子
    degree_centrality = nx.degree_centrality(G)
    node_size = [v * 10000 for v in degree_centrality.values()]  # 调整大小比例因子以适合你的图形

    nx.draw_networkx(G, pos, node_size=node_size, with_labels=True, font_weight='bold')
    plt.show()
    
    degree_centrality = nx.degree_centrality(G)

    # 绘制条形图
    plt.figure(figsize=(10, 8))
    plt.bar(range(len(degree_centrality)), list(degree_centrality.values()), align='center')
    plt.xticks(range(len(degree_centrality)), list(degree_centrality.keys()), rotation='vertical')
    plt.title('Degree Centrality of Nodes')
    plt.show()