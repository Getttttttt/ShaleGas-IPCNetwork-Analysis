import networkx as nx

# 步骤1: 读取.net文件
input_file_path = 'IPCNetwork.net'  # 你的输入文件路径
G = nx.read_pajek(input_file_path)

# 检查图是否为有向图
if G.is_directed():
    print("G is a directed graph.")
else:
    print("G is an undirected graph.")

# 检查图是否为多重图
if G.is_multigraph():
    print("G is a multigraph.")
else:
    print("G is not a multigraph.")

# 转换为无向图，此时如果原图为多重图，则转换结果为MultiGraph
G = G.to_undirected()

# 创建一个新的无向简单图用于合并多重边
simple_G = nx.Graph()

# 步骤2: 合并多重边
for u, v, data in G.edges(data=True):
    weight = data.get('weight', 1)
    if simple_G.has_edge(u, v):
        # 如果边已存在，累加权值
        simple_G[u][v]['weight'] += weight
    else:
        # 否则，添加边，并设置权值
        simple_G.add_edge(u, v, weight=weight)

# 步骤3: 保存修改后的图为一个新的.net文件
output_file_path = 'ModifiedIPCNetwork.net'  # 你的输出文件路径
nx.write_pajek(simple_G, output_file_path)

# 检查图是否为多重图
if simple_G.is_multigraph():
    print("simple_G is a multigraph.")
else:
    print("simple_G is not a multigraph.")

print(f'Modified network has been saved to {output_file_path}')
