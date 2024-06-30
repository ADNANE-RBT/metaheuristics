import math
import heapq

class Node:
    def __init__(self, city, visited, cost, heuristic, parent=None):
        self.city = city
        self.visited = visited
        self.cost = cost
        self.heuristic = heuristic
        self.total_cost = cost + heuristic
        self.parent = parent

    def __lt__(self, other):
        return self.total_cost < other.total_cost

def euclidean_distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def calculate_mst_heuristic(current_city, unvisited_cities, cities):
    if not unvisited_cities:
        return euclidean_distance(cities[current_city], cities[0])
    
    unvisited = list(unvisited_cities) + [current_city]
    edges = [(i, j, euclidean_distance(cities[i], cities[j])) 
             for i in unvisited for j in unvisited if i < j]
    edges.sort(key=lambda x: x[2])
    
    parent = list(range(len(cities)))
    rank = [0] * len(cities)
    
    def find(item):
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]
    
    def union(x, y):
        xroot = find(x)
        yroot = find(y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1
    
    mst_weight = 0
    for u, v, weight in edges:
        if find(u) != find(v):
            union(u, v)
            mst_weight += weight
    
    return mst_weight

def astar_tsp(cities):
    start_city = 0
    num_cities = len(cities)
    all_cities = (1 << num_cities) - 1  # Bit mask representing all cities visited
    
    initial_node = Node(start_city, 1 << start_city, 0, calculate_mst_heuristic(start_city, set(range(1, num_cities)), cities))
    priority_queue = [initial_node]
    closed_set = set()
    
    while priority_queue:
        current_node = heapq.heappop(priority_queue)
        
        state = (current_node.city, current_node.visited)
        if state in closed_set:
            continue
        closed_set.add(state)
        
        if current_node.visited == all_cities:
            final_cost = current_node.cost + euclidean_distance(cities[current_node.city], cities[start_city])
            return reconstruct_path(current_node), final_cost
        
        for next_city in range(num_cities):
            if current_node.visited & (1 << next_city):
                continue
            
            new_visited = current_node.visited | (1 << next_city)
            new_cost = current_node.cost + euclidean_distance(cities[current_node.city], cities[next_city])
            new_heuristic = calculate_mst_heuristic(next_city, set(i for i in range(num_cities) if not new_visited & (1 << i)), cities)
            
            new_node = Node(next_city, new_visited, new_cost, new_heuristic, current_node)
            heapq.heappush(priority_queue, new_node)
    
    return None, float('inf')

def reconstruct_path(node):
    path = []
    while node is not None:
        path.append(node.city)
        node = node.parent
    return path[::-1] + [path[0]]  # Return to the starting city

def parse_tsplib_file(file_path):
    cities = {}
    with open(file_path, 'r') as file:
        dimension = 0
        coordinates_section = False
        for line in file:
            if line.startswith("DIMENSION"):
                dimension = int(line.split(":")[1].strip())
            elif line.startswith("NODE_COORD_SECTION"):
                coordinates_section = True
            elif coordinates_section and len(cities) < dimension:
                parts = line.strip().split()
                if len(parts) == 3:
                    city_id, x, y = map(float, parts)
                    cities[int(city_id) - 1] = (x, y)
    return cities

def main():
    file_path = "./tsplib-master/a280.tsp"  # Replace with the path to your TSPLIB file
    cities = parse_tsplib_file(file_path)
    
    print("Starting A* TSP solver...")
    optimal_path, optimal_cost = astar_tsp(cities)
    
    if optimal_path:
        print("\nFinal Result:")
        print("Optimal Path:", optimal_path)
        print(f"Optimal Cost: {optimal_cost:.2f}")
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()
