import copy
import random
from collections import defaultdict
from collections import deque
import time
import matplotlib.pyplot as plt
import networkx as nx

class TradingGraph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v, n):
        self.graph[u].append((v, n))

    def get_neighbors(self, u):
        return self.graph[u]
    def update_edge_capacity(self, u, v, new_capacity):
        if u in self.graph:
            for i, (neighbor, capacity) in enumerate(self.graph[u]):
                if neighbor == v:
                    self.graph[u][i] = (v, new_capacity)
                    break
    def within_capacity(self, path, graph):
        edge_count = defaultdict(int)
        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            edge_count[edge] += 1
        
        for u in graph:
            for v, n in graph[u]:
                if edge_count[(u, v)] > n:
                    return False

        return True
    def dfs(self, graph, current_node, path, all_walks):
        for neighbor, capacity in list(graph[current_node]):

            if self.within_capacity(path+[neighbor],graph):  
                if len(path) > 2 and path[0] == neighbor:
                    all_walks.append(path+[neighbor])
                    return
                self.dfs(graph, neighbor, path+[neighbor], all_walks)
        if path:
            path.pop()
    def all_possible_walks(self, graph, players):
        all_walks = []
        for player in players:
            self.dfs(graph.graph, player, [player], all_walks)
        if all_walks:
            return max(all_walks, key=len)
        return all_walks

    # Generate a random walk in the graph that starts and ends at the same node
    def random_walk(self, graph, start):
        walk = [start]
        current = start
        visited_edges = defaultdict(int)
        while True:
            neighbors = [(neighbor, capacity) for neighbor, capacity in graph.get_neighbors(current) if visited_edges[(current, neighbor)] < capacity]
            if not neighbors:
                break
            next_node, _ = random.choice(neighbors)
            walk.append(next_node)
            visited_edges[(current, next_node)] += 1
            current = next_node
            if current == start:
                break
        return walk

    # Initialization
    def initialize_population(self, graph, population_size, players):
        population = []
        for _ in range(population_size):
            start = random.choice(players)
            walk = self.random_walk(graph, start)
            population.append(walk)
        return population

    # Fitness Evaluation
    def fitness(self, walk, graph):
        if len(walk) < 2 or walk[0] != walk[-1]:
            return 0
        visited_edges = defaultdict(int)
        for i in range(len(walk) - 1):
            edge = (walk[i], walk[i + 1])
            edge_capacity = next((capacity for neighbor, capacity in graph.get_neighbors(walk[i]) if neighbor == walk[i + 1]), 0)
            if edge_capacity == 0 or visited_edges[edge] >= edge_capacity:
                return 0
            visited_edges[edge] += 1
        return len(walk)

    # Selection
    def select(self, population, fitnesses, num_selected):
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return random.choices(population, k=num_selected)  # Select randomly if all fitnesses are zero
        selected = random.choices(population, weights=fitnesses, k=num_selected)
        return selected

    # Crossover (Splicing)
    def crossover(self, parent1, parent2):
        common = set(parent1) & set(parent2)
        if not common:
            return parent1  # No crossover point
        crossover_point = random.choice(list(common))
        index1 = parent1.index(crossover_point)
        index2 = parent2.index(crossover_point)
        child = parent1[:index1 + 1] + parent2[index2 + 1:]
        return child

    # Mutation
    def mutate(self, walk, graph, mutation_rate):
        if random.random() < mutation_rate:
            mutate_point = random.randint(0, len(walk) - 1)
            neighbors = graph.get_neighbors(walk[mutate_point])
            if neighbors:
                new_node = random.choice(neighbors)[0]
                walk[mutate_point] = new_node
        return walk

    # Genetic Algorithm
    def genetic_algorithm(self, graph, players, population_size=1000, generations=100, mutation_rate=0.15):
        population = self.initialize_population(graph, population_size, players)
        for _ in range(generations):
            fitnesses = [self.fitness(walk, graph) for walk in population]
            new_population = self.select(population, fitnesses, population_size // 2)
            next_generation = []
            while len(next_generation) < population_size:
                parent1, parent2 = random.sample(new_population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, graph, mutation_rate)
                next_generation.append(child)
            population = next_generation
        best_walk = max(population, key=lambda walk: self.fitness(walk, graph))
        if self.fitness(best_walk, graph) == 0:
            return []
        return best_walk
    
    def find_cycle_bfs(self, start):
        queue = deque([(start, None)])  # (current node, parent node)
        visited = {}
        count = 0
        while queue:
            count += 1
            current, parent = queue.popleft()            
            if current == start and count > 1:
                # Cycle detected, reconstruct the cycle
                cycle = []
                cycle.append(current)
                while parent is not None:
                    cycle.append(parent)
                    parent = visited[parent]
                cycle.append(current)  # to complete the cycle
                cycle.reverse()
                return cycle[1:]
            if current in visited:
                continue
            visited[current] = parent
            
            for neighbor, capacity in self.get_neighbors(current):
                if neighbor != parent and capacity > 0 :  # Avoid adding the edge to the parent node
                    queue.append((neighbor, current))
        
        return None  # No cycle found
    
    def find_min_weight_in_cycle(self, cycle):
        if not cycle:
            return None

        min_weight = float('inf')

        for i in range(len(cycle) - 1):
            u = cycle[i]
            v = cycle[i + 1]
            # Find the weight of the edge (u, v)
            for neighbor, weight in self.graph[u]:
                if neighbor == v:
                    if weight < min_weight:
                        min_weight = weight
                    break
        
        return min_weight
    
    def update_graph_reduce_weights(self, cycle, min_weight):
        if cycle is None or min_weight is None:
            return
        
        for i in range(len(cycle) - 1):
            u = cycle[i]
            v = cycle[i + 1]
            # Update the weight of the edge (u, v)
            for j in range(len(self.graph[u])):
                if self.graph[u][j][0] == v:
                    self.graph[u][j] = (v, self.graph[u][j][1] - min_weight)
                    break
    
    def max_flow(self, graph, players):
        cycles = []
        for p in players:
            while graph.find_cycle_bfs(p):
                cycle = graph.find_cycle_bfs(p)                          
                min_weight = graph.find_min_weight_in_cycle(cycle)
                cycles.append((cycle, min_weight))
                graph.update_graph_reduce_weights(cycle, min_weight)                
        return cycles        
    def show_all_graphs(self, graph_data_list):
        num_graphs = len(graph_data_list)
        num_cols = 5  # Number of columns for subplots
        num_rows = (num_graphs + num_cols - 1) // num_cols  # Number of rows for subplots

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12 * num_cols, 8 * num_rows))
        axes = axes.flatten()  # Flatten in case we have more than 1 row/column
        
        for i, (graph, players, title, brute_score, genetic_score, flow_score, genetic_time, flow_time) in enumerate(graph_data_list):
            ax = axes[i]
            G = nx.DiGraph()
            
            for u in graph.graph:
                for v, weight in graph.graph[u]:
                    G.add_edge(u, v, weight=weight)
            
            # Determine bipartite node sets
            top_nodes = players
            
            # Create bipartite layout
            pos = nx.bipartite_layout(G, top_nodes)
            edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
            
            nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, 
                edge_color='black', linewidths=0.5, font_size=10, font_weight='bold', arrows=True, ax=ax)        
            edge_labels = nx.get_edge_attributes(G, 'weight')

            # Adjust positions for edge labels
            edge_label_pos = 0.5
            # Draw edge labels
            nx.draw_networkx_edge_labels(G, edge_label_pos, edge_labels=edge_labels, font_color='blue', font_size=8, ax=ax)

            # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue', font_size=8, ax=ax)
            # ax.set_title(title, fontsize=12)

            # Display additional information if provided
            
            # if genetic_score is not None:
            #     info_text += f"Genetic Score: {genetic_score}\n"
            # if flow_score is not None:
            #     info_text += f"Flow Score: {flow_score}\n"
            # if genetic_time is not None:
            #     info_text += f"Genetic Time: {genetic_time:.4f}\n"
            # if flow_time is not None:
            #     info_text += f"Flow Time: {flow_time:.4f}\n"
            
        # Hide any remaining empty subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
    def show_graph(self, graph, players, title, genetic_score=None, flow_score=None, genetic_time=None, flow_time=None):
        G = nx.DiGraph()
        
        for u in graph.graph:
            for v, weight in graph.graph[u]:
                G.add_edge(u, v, weight=weight)
        
         # Determine bipartite node sets
        top_nodes = players
        
        # Create bipartite layout
        pos = nx.bipartite_layout(G, top_nodes)
        edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
        # edge_labels = nx.get_edge_attributes(G, 'weight')
        label_pos = 0.6
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, 
            edge_color='black', linewidths=1, font_size=15, font_weight='bold', arrows=True)        
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=label_pos, font_color='blue', font_size=12)
        plt.title(title)
        
        # Display additional information if provided
        if genetic_score is not None:
            plt.text(0.5, 0.95, f"Genetic Score: {genetic_score}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        if flow_score is not None:
            plt.text(0.5, 0.90, f"Flow Score: {flow_score}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        # if genetic_time is not None:
        #     plt.text(0.5, 0.85, f"Genetic Time: {genetic_time:.4f}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        # if flow_time is not None:
        #     plt.text(0.5, 0.80, f"Flow Time: {flow_time:.4f}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)        
        plt.show()


    def find_repeated_edges(self, graph):
        edges_count = {}
        repeated_edges = []

        for u in graph.graph:
            for v, n in  graph.graph[u]:
                if "".join(sorted((u, v))) in edges_count:
                    edges_count["".join(sorted((u, v)))] += 1
                else:
                    edges_count["".join(sorted((u, v)))] = 1
        for edge in edges_count:
            if edges_count[edge] > 1:
                repeated_edges.append(edge)

        return repeated_edges
    # Function to run a test case
    def run_test_case(self, graph, players, description, show_graph=False):   
        repeated_edges = self.find_repeated_edges(graph)
        if repeated_edges:
            raise ValueError(description, "Graph should not contain multiple edges between two vertices", repeated_edges)
        
        start_graph = copy.deepcopy(graph)
        # Measure time and find the best walk using the genetic algorithm
        start_time = time.time()
        best_walk_genetic = self.genetic_algorithm(graph, players)
        genetic_time = time.time() - start_time
        
        start_time = time.time()
        best_flows = self.max_flow(graph, players)        
        flow_time = time.time() - start_time

        # Measure time and find the best walk using the brute force algorithm
        start_time = time.time()
        best_walk_brute = self.all_possible_walks(start_graph, players)
        brute_time = time.time() - start_time
        
        # Calculate the optimal rate
        genetic_length = len(best_walk_genetic)
        genetic_score = genetic_length // 2
        flow_score = sum([len(b[0]) // 2 * b[1] for b in best_flows])
        brute_score = len(best_walk_brute) // 2
        
        # flow_optimal_rate = (flow_length / brute_length) * 100 if brute_length > 0 else 0
        # genetic_optimal_rate = (genetic_length / brute_length) * 100 if brute_length > 0 else 0
        
        # Print the results
        print(f"{description}:")
        
        print(f"  Brute Force - Best walk: {best_walk_brute}")
        print(f"  Score: {brute_score}, Time: {brute_time:.4f} seconds\n")
        print(f"  Genetic Algorithm - Best walk: {best_walk_genetic}")
        print(f"  Score: {genetic_score}, Time: {genetic_time:.4f} seconds\n")
        # print(f"  Optimal Rate: {genetic_optimal_rate:.2f}%\n")
        print(f"  St Network - Max Flow: {best_flows}")
        print(f"  Score: {flow_score}, Time: {flow_time:.4f} seconds")
        # print(f"  Optimal Rate: {flow_optimal_rate:.2f}%\n")
        print()
        if show_graph:
            self.show_graph(start_graph, players, description, genetic_score, flow_score, genetic_time,flow_time)
        return(brute_score, genetic_score, flow_score, genetic_time,flow_time,start_graph, players, description,)
    
    def test_case_0(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 1)
        graph.add_edge('P1', 'C2', 1)
        graph.add_edge('P2', 'C3', 3)
        graph.add_edge('P3', 'C2', 2)
        graph.add_edge('C1', 'P2', 4)
        graph.add_edge('C1', 'P3', 2)
        graph.add_edge('C2', 'P2', 2)
        graph.add_edge('C3', 'P1', 2)
        graph.add_edge('C3', 'P3', 1)
        players = ['P1', 'P2', 'P3']
        return self.run_test_case(graph, players,"Test Case 0 - Document")

    def test_case_1(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 1)
        graph.add_edge('P2', 'C2', 1)
        graph.add_edge('P3', 'C3', 1)
        graph.add_edge('C1', 'P2', 1)
        graph.add_edge('C2', 'P3', 1)
        graph.add_edge('C3', 'P1', 1)
        graph.add_edge('P1', 'C2', 1)
        players = ['P1', 'P2', 'P3']
        return self.run_test_case(graph, players, "Test Case 1 - Small graph with few trades")

    def test_case_2(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 1)
        graph.add_edge('P1', 'C2', 1)
        graph.add_edge('P2', 'C1', 1)
        graph.add_edge('P2', 'C3', 1)
        graph.add_edge('P3', 'C2', 1)
        graph.add_edge('C1', 'P3', 1)
        graph.add_edge('C2', 'P2', 1)
        players = ['P1', 'P2', 'P3']
        return self.run_test_case(graph, players, "Test Case 2 - Small graph with few trades")

    def test_case_3(self):
        graph = TradingGraph()
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 1)
        graph.add_edge('P1', 'C2', 1)
        graph.add_edge('P2', 'C1', 1)
        graph.add_edge('P2', 'C3', 1)
        graph.add_edge('P3', 'C2', 1)
        graph.add_edge('P4', 'C2', 1)
        graph.add_edge('C1', 'P3', 1)
        graph.add_edge('C2', 'P2', 1)
        graph.add_edge('C1', 'P4', 1)
        graph.add_edge('C3', 'P3', 1)
        players = ['P1', 'P2', 'P3', 'P4']
        return self.run_test_case(graph, players, "Test Case 3 - More players than cards")

    def test_case_4(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 1)
        graph.add_edge('P1', 'C2', 1)
        graph.add_edge('P1', 'C4', 1)
        graph.add_edge('P2', 'C3', 1)
        graph.add_edge('P2', 'C4', 1)
        graph.add_edge('C1', 'P2', 1)
        graph.add_edge('C3', 'P1', 1)
        players = ['P1', 'P2']
        return self.run_test_case(graph, players, "Test Case 4 - More cards than players")

    def test_case_5(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 1)
        graph.add_edge('C1', 'P2', 1)
        graph.add_edge('P2', 'C2', 1)
        graph.add_edge('C2', 'P3', 1)
        graph.add_edge('P3', 'C3', 1)
        graph.add_edge('C3', 'P1', 1)
        players = ['P1', 'P2', 'P3']
        return self.run_test_case(graph, players, "Test Case 5 - Simple cycle")

    def test_case_6(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 1)
        graph.add_edge('P1', 'C2', 1)
        graph.add_edge('P2', 'C3', 1)
        graph.add_edge('P2', 'C4', 1)
        graph.add_edge('C1', 'P3', 1)
        players = ['P1', 'P2', 'P3']
        return self.run_test_case(graph, players, "Test Case 6 - Disconnected graph")

    def test_case_7(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 1)
        graph.add_edge('P1', 'C2', 1)
        graph.add_edge('P1', 'C3', 1)
        graph.add_edge('P2', 'C4', 1)
        graph.add_edge('P2', 'C5', 1)
        graph.add_edge('P3', 'C6', 1)
        graph.add_edge('P4', 'C1', 1)
        graph.add_edge('P4', 'C4', 1)
        graph.add_edge('P5', 'C2', 1)
        graph.add_edge('C1', 'P3', 1)
        graph.add_edge('C2', 'P4', 1)
        graph.add_edge('C3', 'P5', 1)
        graph.add_edge('C4', 'P1', 1)
        players = ['P1', 'P2', 'P3', 'P4', 'P5']
        return self.run_test_case(graph, players, "Test Case 7 - More players and cards")

    def test_case_8(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 1)
        graph.add_edge('P1', 'C2', 1)
        graph.add_edge('P2', 'C3', 1)
        graph.add_edge('P2', 'C4', 1)
        graph.add_edge('P3', 'C5', 1)
        graph.add_edge('P3', 'C6', 1)
        graph.add_edge('P4', 'C7', 1)
        graph.add_edge('P4', 'C8', 1)
        graph.add_edge('P5', 'C9', 1)
        graph.add_edge('P5', 'C10', 1)
        graph.add_edge('P6', 'C1', 1)
        graph.add_edge('P6', 'C4', 1)
        graph.add_edge('C1', 'P2', 1)
        graph.add_edge('C2', 'P3', 1)
        graph.add_edge('C3', 'P4', 1)
        graph.add_edge('C4', 'P5', 1)
        graph.add_edge('C5', 'P6', 1)
        graph.add_edge('C6', 'P1', 1)
        graph.add_edge('C7', 'P2', 1)
        graph.add_edge('C8', 'P3', 1)
        graph.add_edge('C9', 'P4', 1)
        players = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
        return self.run_test_case(graph, players, "Test Case 8 - Even more players and cards")

    def test_case_9(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 1)
        graph.add_edge('P1', 'C2', 1)
        graph.add_edge('P1', 'C3', 1)
        graph.add_edge('P2', 'C4', 1)
        graph.add_edge('P2', 'C5', 1)
        graph.add_edge('P3', 'C6', 1)
        graph.add_edge('P3', 'C7', 1)
        graph.add_edge('P4', 'C8', 1)
        graph.add_edge('P4', 'C9', 1)
        graph.add_edge('P5', 'C10', 1)
        graph.add_edge('P6', 'C1', 1)
        graph.add_edge('P6', 'C2', 1)
        graph.add_edge('P7', 'C3', 1)
        graph.add_edge('P7', 'C4', 1)
        graph.add_edge('P8', 'C5', 1)
        graph.add_edge('P8', 'C6', 1)
        graph.add_edge('P9', 'C7', 1)
        graph.add_edge('P9', 'C8', 1)
        graph.add_edge('P10', 'C9', 1)
        graph.add_edge('P10', 'C10', 1)
        graph.add_edge('C1', 'P4', 1)
        graph.add_edge('C2', 'P5', 1)
        graph.add_edge('C3', 'P6', 1)
        graph.add_edge('C6', 'P9', 1)
        graph.add_edge('C7', 'P10', 1)
        graph.add_edge('C8', 'P1', 1)
        graph.add_edge('C9', 'P2', 1)
        graph.add_edge('C10', 'P3', 1)
        players = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
        return self.run_test_case(graph, players, "Test Case 9 - Large graph with many players and cards")

    def test_case_10(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 1)
        graph.add_edge('P1', 'C2', 1)
        graph.add_edge('P1', 'C3', 1)
        graph.add_edge('P2', 'C4', 1)
        graph.add_edge('P2', 'C5', 1)
        graph.add_edge('P2', 'C6', 1)
        graph.add_edge('P3', 'C7', 1)
        graph.add_edge('P3', 'C8', 1)
        graph.add_edge('P4', 'C9', 1)
        graph.add_edge('P4', 'C10', 1)
        graph.add_edge('P4', 'C1', 1)
        graph.add_edge('P5', 'C2', 1)
        graph.add_edge('P5', 'C3', 1)
        graph.add_edge('P5', 'C4', 1)
        graph.add_edge('P6', 'C5', 1)
        graph.add_edge('P6', 'C6', 1)
        graph.add_edge('P7', 'C7', 1)
        graph.add_edge('P7', 'C8', 1)
        graph.add_edge('P8', 'C9', 1)
        graph.add_edge('P8', 'C10', 1)
        graph.add_edge('P9', 'C1', 1)
        graph.add_edge('P9', 'C2', 1)
        graph.add_edge('P10', 'C3', 1)
        graph.add_edge('P10', 'C4', 1)
        graph.add_edge('C1', 'P5', 1)
        graph.add_edge('C2', 'P6', 1)
        graph.add_edge('C3', 'P7', 1)
        graph.add_edge('C4', 'P8', 1)
        graph.add_edge('C5', 'P9', 1)
        graph.add_edge('C6', 'P10', 1)
        graph.add_edge('C7', 'P1', 1)
        graph.add_edge('C8', 'P2', 1)
        graph.add_edge('C9', 'P3', 1)
        players = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
        return self.run_test_case(graph, players, "Test Case 10 - Very large graph with dense connections")

    def test_case_11(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 1)
        graph.add_edge('P2', 'C2', 3)
        graph.add_edge('P3', 'C3', 2)
        graph.add_edge('C1', 'P2', 4)
        graph.add_edge('C2', 'P3', 1)
        graph.add_edge('C3', 'P1', 5)
        graph.add_edge('P1', 'C2', 2)
        players = ['P1', 'P2', 'P3']
        return self.run_test_case(graph, players, "Test Case 11 - Small graph with varying weights")

    def test_case_12(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 3)
        graph.add_edge('P1', 'C2', 5)
        graph.add_edge('P2', 'C1', 1)
        graph.add_edge('P2', 'C3', 1)
        graph.add_edge('P3', 'C2', 3)
        graph.add_edge('C1', 'P3', 6)
        graph.add_edge('C2', 'P2', 2)        
        players = ['P1', 'P2', 'P3']
        return self.run_test_case(graph, players, "Test Case 12 - Dense graph with varying weights")

    def test_case_13(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 3)
        graph.add_edge('P1', 'C2', 1)
        graph.add_edge('P2', 'C1', 5)
        graph.add_edge('P2', 'C3', 6)
        graph.add_edge('P3', 'C2', 2)
        graph.add_edge('P4', 'C2', 7)
        graph.add_edge('C1', 'P3', 4)
        graph.add_edge('C2', 'P2', 3)
        graph.add_edge('C1', 'P4', 2)
        graph.add_edge('C3', 'P3', 1)
        players = ['P1', 'P2', 'P3', 'P4']
        return self.run_test_case(graph, players, "Test Case 13 - More players than cards with varying weights")

    def test_case_14(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 1)
        graph.add_edge('P1', 'C2', 5)
        graph.add_edge('P1', 'C4', 6)
        graph.add_edge('P2', 'C3', 3)
        graph.add_edge('P2', 'C4', 4)
        graph.add_edge('C1', 'P2', 6)
        graph.add_edge('C3', 'P1', 3)
        players = ['P1', 'P2']
        return self.run_test_case(graph, players, "Test Case 14 - More cards than players with varying weights")

    def test_case_15(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 3)
        graph.add_edge('C1', 'P2', 1)
        graph.add_edge('P2', 'C2', 4)
        graph.add_edge('C2', 'P3', 2)
        graph.add_edge('P3', 'C3', 5)
        graph.add_edge('C3', 'P1', 1)
        players = ['P1', 'P2', 'P3']
        return self.run_test_case(graph, players, "Test Case 15 - Simple cycle with varying weights")

    def test_case_16(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 2)
        graph.add_edge('P1', 'C2', 3)
        graph.add_edge('P2', 'C3', 1)
        graph.add_edge('P2', 'C4', 4)
        graph.add_edge('C1', 'P3', 5)
        players = ['P1', 'P2', 'P3']
        return self.run_test_case(graph, players, "Test Case 16 - Disconnected graph with varying weights")

    def test_case_17(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 1)
        graph.add_edge('P1', 'C2', 5)
        graph.add_edge('P1', 'C3', 3)
        graph.add_edge('P2', 'C4', 2)
        graph.add_edge('P2', 'C5', 4)
        graph.add_edge('P3', 'C6', 2)
        graph.add_edge('P4', 'C1', 3)
        graph.add_edge('P4', 'C4', 1)
        graph.add_edge('P5', 'C2', 4)
        graph.add_edge('C1', 'P3', 1)
        graph.add_edge('C2', 'P4', 5)
        graph.add_edge('C3', 'P5', 3)
        graph.add_edge('C4', 'P1', 2)
        players = ['P1', 'P2', 'P3', 'P4', 'P5']
        return self.run_test_case(graph, players, "Test Case 17 - More players and cards with varying weights", True)

    def test_case_18(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 3)
        graph.add_edge('P1', 'C2', 4)
        graph.add_edge('P2', 'C3', 2)
        graph.add_edge('P2', 'C4', 1)
        graph.add_edge('P3', 'C5', 5)
        graph.add_edge('P3', 'C6', 3)
        graph.add_edge('P4', 'C7', 1)
        graph.add_edge('P4', 'C8', 4)
        graph.add_edge('P5', 'C9', 2)
        graph.add_edge('P5', 'C10', 5)
        graph.add_edge('P6', 'C1', 3)
        graph.add_edge('P6', 'C4', 2)
        graph.add_edge('C1', 'P2', 1)
        graph.add_edge('C2', 'P3', 4)
        graph.add_edge('C3', 'P4', 5)
        graph.add_edge('C4', 'P5', 2)
        graph.add_edge('C5', 'P6', 3)
        graph.add_edge('C6', 'P1', 4)
        graph.add_edge('C7', 'P2', 1)
        graph.add_edge('C8', 'P3', 3)
        graph.add_edge('C9', 'P4', 2)
        players = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
        return self.run_test_case(graph, players, "Test Case 18 - Even more players and cards with varying weights", True)

    def test_case_19(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 2)
        graph.add_edge('P1', 'C2', 4)
        graph.add_edge('P1', 'C3', 5)
        graph.add_edge('P2', 'C4', 1)
        graph.add_edge('P2', 'C5', 3)
        graph.add_edge('P3', 'C6', 2)
        graph.add_edge('P3', 'C7', 4)
        graph.add_edge('P4', 'C8', 1)
        graph.add_edge('P4', 'C9', 3)
        graph.add_edge('P5', 'C10', 5)
        graph.add_edge('P6', 'C1', 2)
        graph.add_edge('P6', 'C2', 4)
        graph.add_edge('P7', 'C3', 1)
        graph.add_edge('P7', 'C4', 5)
        graph.add_edge('P8', 'C5', 3)
        graph.add_edge('P8', 'C6', 2)
        graph.add_edge('P9', 'C7', 4)
        graph.add_edge('P9', 'C8', 5)
        graph.add_edge('P10', 'C9', 1)
        graph.add_edge('P10', 'C10', 3)
        graph.add_edge('C1', 'P4', 2)
        graph.add_edge('C2', 'P5', 4)
        graph.add_edge('C3', 'P6', 3)
        graph.add_edge('C6', 'P9', 4)
        graph.add_edge('C7', 'P10', 2)
        graph.add_edge('C8', 'P1', 3)
        graph.add_edge('C9', 'P2', 5)
        graph.add_edge('C10', 'P3', 1)
        players = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
        return self.run_test_case(graph, players, "Test Case 19 - Large graph with varying weights")

    def test_case_20(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 3)
        graph.add_edge('P1', 'C2', 2)
        graph.add_edge('P1', 'C3', 4)
        graph.add_edge('P2', 'C4', 1)
        graph.add_edge('P2', 'C5', 3)
        graph.add_edge('P2', 'C6', 5)
        graph.add_edge('P3', 'C7', 2)
        graph.add_edge('P3', 'C8', 4)
        graph.add_edge('P4', 'C9', 1)
        graph.add_edge('P4', 'C10', 3)
        graph.add_edge('P4', 'C1', 5)
        graph.add_edge('P5', 'C2', 2)
        graph.add_edge('P5', 'C3', 4)
        graph.add_edge('P5', 'C4', 1)
        graph.add_edge('P6', 'C5', 3)
        graph.add_edge('P6', 'C6', 5)
        graph.add_edge('P7', 'C7', 2)
        graph.add_edge('P7', 'C8', 4)
        graph.add_edge('P8', 'C9', 1)
        graph.add_edge('P8', 'C10', 3)
        graph.add_edge('P9', 'C1', 5)
        graph.add_edge('P9', 'C2', 2)
        graph.add_edge('P10', 'C3', 4)
        graph.add_edge('P10', 'C4', 1)
        graph.add_edge('C1', 'P5', 3)
        graph.add_edge('C2', 'P6', 4)
        graph.add_edge('C3', 'P7', 5)
        graph.add_edge('C4', 'P8', 2)
        graph.add_edge('C5', 'P9', 3)
        graph.add_edge('C6', 'P10', 1)
        graph.add_edge('C7', 'P1', 2)
        graph.add_edge('C8', 'P2', 3)
        graph.add_edge('C9', 'P3', 4)
        players = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
        return self.run_test_case(graph, players, "Test Case 20 - Very large graph with dense and varying weights")
    def test_case_21(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 5)
        graph.add_edge('P1', 'C2', 3)
        graph.add_edge('P2', 'C3', 6)
        graph.add_edge('P2', 'C4', 2)
        graph.add_edge('P3', 'C5', 4)
        graph.add_edge('P3', 'C6', 7)
        graph.add_edge('P4', 'C7', 1)
        graph.add_edge('P4', 'C8', 5)
        graph.add_edge('P5', 'C9', 3)
        graph.add_edge('P5', 'C10', 6)
        graph.add_edge('P6', 'C1', 8)
        graph.add_edge('P6', 'C2', 2)
        graph.add_edge('P7', 'C3', 5)
        graph.add_edge('P7', 'C4', 1)
        graph.add_edge('P8', 'C5', 4)
        graph.add_edge('P8', 'C6', 7)
        graph.add_edge('P9', 'C7', 2)
        graph.add_edge('P9', 'C8', 3)
        graph.add_edge('P10', 'C9', 5)
        graph.add_edge('P10', 'C10', 1)
        graph.add_edge('C1', 'P3', 4)
        graph.add_edge('C2', 'P4', 6)
        graph.add_edge('C3', 'P5', 3)
        graph.add_edge('C4', 'P6', 5)
        graph.add_edge('C5', 'P7', 7)
        graph.add_edge('C8', 'P10', 4)
        graph.add_edge('C9', 'P1', 6)
        graph.add_edge('C10', 'P2', 3)
        players = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
        return self.run_test_case(graph, players, "Test Case 21 - Large graph with varying weights and connections")

    def test_case_22(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 1)
        graph.add_edge('P1', 'C2', 4)
        graph.add_edge('P1', 'C3', 7)
        graph.add_edge('P2', 'C4', 2)
        graph.add_edge('P2', 'C5', 5)
        graph.add_edge('P2', 'C6', 8)
        graph.add_edge('P3', 'C7', 3)
        graph.add_edge('P3', 'C8', 6)
        graph.add_edge('P3', 'C9', 9)
        graph.add_edge('P4', 'C10', 1)
        graph.add_edge('P5', 'C1', 4)
        graph.add_edge('P5', 'C2', 7)
        graph.add_edge('P6', 'C3', 2)
        graph.add_edge('P6', 'C4', 5)
        graph.add_edge('P7', 'C5', 8)
        graph.add_edge('P7', 'C6', 3)
        graph.add_edge('P8', 'C7', 6)
        graph.add_edge('P8', 'C8', 9)
        graph.add_edge('P9', 'C9', 1)
        graph.add_edge('P9', 'C10', 4)
        graph.add_edge('P10', 'C1', 7)
        graph.add_edge('P10', 'C2', 2)
        graph.add_edge('C1', 'P6', 5)
        graph.add_edge('C2', 'P7', 8)
        graph.add_edge('C3', 'P8', 3)
        graph.add_edge('C4', 'P9', 6)
        graph.add_edge('C5', 'P10', 9)
        graph.add_edge('C6', 'P1', 1)
        graph.add_edge('C7', 'P2', 4)
        graph.add_edge('C9', 'P4', 2)
        graph.add_edge('C10', 'P5', 5)
        players = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
        return self.run_test_case(graph, players, "Test Case 22 - Graph with incremental weights and multiple cycles")

    def test_case_23(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 3)
        graph.add_edge('P1', 'C2', 3)
        graph.add_edge('P1', 'C3', 3)
        graph.add_edge('P2', 'C4', 3)
        graph.add_edge('P2', 'C5', 3)
        graph.add_edge('P2', 'C6', 3)
        graph.add_edge('P3', 'C7', 3)
        graph.add_edge('P3', 'C8', 3)
        graph.add_edge('P3', 'C9', 3)
        graph.add_edge('P4', 'C10', 3)
        graph.add_edge('P4', 'C1', 3)
        graph.add_edge('P5', 'C2', 3)
        graph.add_edge('P5', 'C3', 3)
        graph.add_edge('P6', 'C4', 3)
        graph.add_edge('P6', 'C5', 3)
        graph.add_edge('P7', 'C6', 3)
        graph.add_edge('P7', 'C7', 3)
        graph.add_edge('P8', 'C8', 3)
        graph.add_edge('P8', 'C9', 3)
        graph.add_edge('P9', 'C10', 3)
        graph.add_edge('P10', 'C1', 3)
        graph.add_edge('C1', 'P2', 3)
        graph.add_edge('C2', 'P3', 3)
        graph.add_edge('C3', 'P4', 3)
        graph.add_edge('C4', 'P5', 3)
        graph.add_edge('C7', 'P8', 3)
        graph.add_edge('C8', 'P9', 3)
        graph.add_edge('C9', 'P10', 3)
        graph.add_edge('C10', 'P1', 3)
        players = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
        return self.run_test_case(graph, players, "Test Case 23 - Uniform weight graph")

    def test_case_24(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 4)
        graph.add_edge('P2', 'C2', 4)
        graph.add_edge('P3', 'C3', 4)
        graph.add_edge('P4', 'C4', 4)
        graph.add_edge('P5', 'C5', 4)
        graph.add_edge('P6', 'C6', 4)
        graph.add_edge('P7', 'C7', 4)
        graph.add_edge('P8', 'C8', 4)
        graph.add_edge('P9', 'C9', 4)
        graph.add_edge('P10', 'C10', 4)
        graph.add_edge('C1', 'P2', 4)
        graph.add_edge('C2', 'P3', 4)
        graph.add_edge('C3', 'P4', 4)
        graph.add_edge('C4', 'P5', 4)
        graph.add_edge('C5', 'P6', 4)
        graph.add_edge('C6', 'P7', 4)
        graph.add_edge('C7', 'P8', 4)
        graph.add_edge('C8', 'P9', 4)
        graph.add_edge('C9', 'P10', 4)
        graph.add_edge('C10', 'P1', 4)
        players = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
        return self.run_test_case(graph, players, "Test Case 24 - Circular graph with uniform weights")

    def test_case_25(self):
        graph = TradingGraph()
        graph.add_edge('P1', 'C1', 2)
        graph.add_edge('P1', 'C2', 2)
        graph.add_edge('P1', 'C3', 2)
        graph.add_edge('P1', 'C4', 2)
        graph.add_edge('P2', 'C5', 2)
        graph.add_edge('P2', 'C6', 2)
        graph.add_edge('P2', 'C7', 2)
        graph.add_edge('P2', 'C8', 2)
        graph.add_edge('P3', 'C9', 2)
        graph.add_edge('P3', 'C10', 2)
        graph.add_edge('P4', 'C1', 2)
        graph.add_edge('P4', 'C2', 2)
        graph.add_edge('P5', 'C3', 2)
        graph.add_edge('P5', 'C4', 2)
        graph.add_edge('P6', 'C5', 2)
        graph.add_edge('P6', 'C6', 2)
        graph.add_edge('P7', 'C7', 2)
        graph.add_edge('P7', 'C8', 2)
        graph.add_edge('P8', 'C9', 2)
        graph.add_edge('P8', 'C10', 2)
        graph.add_edge('P9', 'C1', 2)
        graph.add_edge('P9', 'C2', 2)
        graph.add_edge('P10', 'C3', 2)
        graph.add_edge('P10', 'C4', 2)
        graph.add_edge('C1', 'P5', 2)
        graph.add_edge('C2', 'P6', 2)
        graph.add_edge('C3', 'P7', 2)
        graph.add_edge('C4', 'P8', 2)
        graph.add_edge('C5', 'P9', 2)
        graph.add_edge('C6', 'P10', 2)
        graph.add_edge('C7', 'P1', 2)
        graph.add_edge('C10', 'P4', 2)
        players = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
        return self.run_test_case(graph, players, "Test Case 25 - Uniform weights with multiple edges")
    def test_case_26(self):
        graph = TradingGraph()
        players = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
        edges = [
            ('P1', 'C2', 5), ('P1', 'C3', 2), ('P1', 'C4', 4), ('P1', 'C5', 1), ('P1', 'C6', 7), ('P1', 'C7', 2), ('P1', 'C8', 6),
            ('P2', 'C9', 4), ('P2', 'C10', 3), ('P2', 'C11', 5), ('P2', 'C12', 6), ('P2', 'C13', 2), ('P2', 'C14', 7), ('P2', 'C15', 4), ('P2', 'C16', 3),
            ('P3', 'C17', 5), ('P3', 'C18', 6), ('P3', 'C19', 2), ('P3', 'C20', 4), ('P3', 'C1', 7), ('P3', 'C2', 3), ('P3', 'C4', 6),
            ('P4', 'C5', 2), ('P4', 'C6', 7), ('P4', 'C7', 4), ('P4', 'C8', 3), ('P4', 'C9', 5), ('P4', 'C10', 6), ('P4', 'C11', 2), ('P4', 'C12', 7),
            ('P5', 'C13', 4), ('P5', 'C14', 3), ('P5', 'C15', 5), ('P5', 'C16', 6), ('P5', 'C17', 2), ('P5', 'C18', 7), ('P5', 'C19', 4), ('P5', 'C20', 3),
            ('P6', 'C1', 5), ('P6', 'C2', 6), ('P6', 'C3', 2), ('P6', 'C4', 4), ('P6', 'C5', 7), ('P6', 'C6', 3), ('P6', 'C7', 5), ('P6', 'C8', 6),
            ('P7', 'C9', 2), ('P7', 'C10', 7), ('P7', 'C11', 4), ('P7', 'C12', 3), ('P7', 'C13', 5), ('P7', 'C14', 6), ('P7', 'C15', 2), ('P7', 'C16', 7),
            ('P8', 'C17', 4), ('P8', 'C18', 3), ('P8', 'C19', 5), ('P8', 'C20', 6), ('P8', 'C1', 2), ('P8', 'C2', 7), ('P8', 'C3', 4), ('P8', 'C4', 3),
            ('P9', 'C5', 5), ('P9', 'C6', 6), ('P9', 'C7', 2), ('P9', 'C8', 4), ('P9', 'C10', 3), ('P9', 'C11', 5), ('P9', 'C12', 6),
            ('P10', 'C13', 2), ('P10', 'C14', 7), ('P10', 'C15', 4), ('P10', 'C16', 3), ('P10', 'C17', 5), ('P10', 'C18', 6), ('P10', 'C19', 2), ('P10', 'C20', 7),
            ('C1', 'P1', 3), ('C2', 'P2', 5), ('C3', 'P3', 2), ('C4', 'P4', 4), ('C5', 'P5', 1), ('C7', 'P7', 2), ('C8', 'P8', 6),
            ('C9', 'P9', 4), ('C10', 'P10', 3), ('C11', 'P1', 5), ('C13', 'P3', 2), ('C14', 'P4', 7), ('C16', 'P6', 3),
            ('C17', 'P7', 5), ('C19', 'P9', 2)
        ]

        for edge in edges:
            graph.add_edge(*edge)

        return self.run_test_case(graph, players, "Test Case 26 - Super dense graph with each player and card having at least 8 connections")


    def main(self):
        # Execute all test cases
        results = []
        others = []
        for i in range(0, 21):
            method_name = f"test_case_{i}"
            method = getattr(self, method_name)
            brute_score, genetic_score, flow_score, genetic_time,flow_time, start_graph, players, description = method()
            results.append((brute_score, genetic_score, flow_score, genetic_time,flow_time))
            others.append((start_graph, players, description, brute_score, genetic_score, flow_score, genetic_time,flow_time))

        average_results = [sum(x) / len(x) for x in zip(*results)]
        start_graph, players, description, genetic_score, flow_score, genetic_time,flow_time
        # self.show_all_graphs(others)
        print("Average Results:")
        print(f"Average Brute-walk Score: {average_results[0]:.2f}")
        print(f"Average Genetic Score: {average_results[1]:.2f}")
        print(f"Average Flow Score: {average_results[2]:.2f}")
        print(f"Average Genetic Time: {average_results[3]:.2f}")
        print(f"Average Flow Time: {average_results[4]:.2f}")
if __name__ == "__main__":
    t = TradingGraph()
    t.main()