"""
BITS WILP ACI Assignment 1 - PS07
Group: G087

Emergency Route Planning using Greedy Best First Search.
Input: graph edges, heuristics, start and goal nodes
Output: visited nodes, path, and total cost
"""

import heapq
import sys
import time
import tracemalloc


# NOTE: using heapq for priority queue implementation
class Node:
    """Represents a node in the frontier with heuristic value and path"""
    def __init__(self, h_val, node, path, cost, order):
        self.h_val = h_val
        self.node = node
        self.path = path
        self.cost = cost
        self.order = order

    def __lt__(self, other):
        # Compare by heuristic first, then order for tie-breaking
        if self.h_val != other.h_val:
            return self.h_val < other.h_val
        return self.order < other.order


class PriorityQueue:
    """Min-priority queue for GBFS with capacity checking"""

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("Priority queue capacity must be greater than zero.")
        self.capacity = capacity
        self._items = []

    def push(self, item):
        if len(self._items) >= self.capacity:
            raise OverflowError("Priority queue is full; cannot insert a new route.")
        heapq.heappush(self._items, item)

    def pop(self):
        if not self._items:
            raise IndexError("Priority queue is empty; cannot delete a route.")
        return heapq.heappop(self._items)

    def is_empty(self):
        return len(self._items) == 0


def clean_str(token):
    """Remove punctuation from input tokens"""
    return token.strip().strip(",").strip()


def get_neighbor(token):
    """Parse neighbor token like 'B:4' or 'B' into (node, cost)"""
    token = clean_str(token)
    if ":" in token:
        node, cost_text = token.split(":", 1)
        if not node:
            raise ValueError(f"Invalid edge entry: {token}")
        return node, float(cost_text)
    return token, 1.0


def read_node(line, keyword):
    """Read START/GOAL lines from input"""
    normalized = line.replace(":", " ").replace("=", " ")
    parts = normalized.split()
    if not parts or parts[0].upper() != keyword:
        return None
    if len(parts) >= 3 and parts[1].upper() == "NODE":
        return parts[2]
    if len(parts) == 2:
        return parts[1]
    raise ValueError(f"{keyword.title()} must be followed by one node.")


def read_input(path):
    """Parse input file and return graph, heuristics, start, goal"""
    graph = {}
    h_values = {}
    start = None
    goal = None
    section = None

    # Read input file line by line
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    line_num = 0
    for raw_line in lines:
        line_num += 1
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        upper_line = line.upper()
        if upper_line == "EDGES":
            section = "EDGES"
            continue
        if upper_line == "HEURISTICS":
            section = "HEURISTICS"
            continue

        parsed_start = read_node(line, "START")
        if parsed_start is not None:
            start = parsed_start
            continue
        parsed_goal = read_node(line, "GOAL")
        if parsed_goal is not None:
            goal = parsed_goal
            continue

        parts = [clean_str(part) for part in line.replace(",", " ").split() if clean_str(part)]

        if section == "EDGES":
            source = parts[0].rstrip(":")
            graph.setdefault(source, [])
            for token in parts[1:]:
                neighbor, cost = get_neighbor(token)
                if cost < 0:
                    raise ValueError(f"Line {line_num}: edge cost cannot be negative.")
                graph[source].append((neighbor, cost))
                graph.setdefault(neighbor, [])
        elif section == "HEURISTICS":
            if len(parts) == 1 and ":" in parts[0]:
                parts = [part.strip() for part in parts[0].split(":", 1)]
            if len(parts) != 2:
                raise ValueError(f"Line {line_num}: heuristic entry must be '<node> <value>'.")
            h_values[parts[0].rstrip(":")] = float(parts[1])
        else:
            raise ValueError(f"Line {line_num}: content must appear under EDGES or HEURISTICS.")

    if goal is None:
        raise ValueError("Goal node is missing. Add a line like 'GOAL G'.")

    check_graph(graph, h_values, start, goal)
    return graph, h_values, start, goal


def check_graph(graph, h_values, start, goal):
    """Validate graph nodes and heuristics"""
    # Build set of all nodes in graph
    nodes = set(graph.keys())
    for edges in graph.values():
        for neighbor, _ in edges:
            nodes.add(neighbor)

    # Check for missing heuristics
    missing = []
    for node in sorted(nodes):
        if node not in h_values:
            missing.append(node)

    if missing:
        raise ValueError("Heuristic value missing for node(s): " + ", ".join(missing))
    if start is not None and start not in nodes:
        raise ValueError(f"Start node '{start}' is not present in the graph.")
    if goal not in nodes:
        raise ValueError(f"Goal node '{goal}' is not present in the graph.")


def greedy_best_first_search(graph, h_values, start, goal):
    """
    Perform Greedy Best First Search using f(n) = h(n).
    Returns: visited_order, path, total_cost
    """
    if start not in graph:
        raise ValueError(f"Start node '{start}' is not present in the graph.")

    # TODO: maybe add better capacity estimation for frontier
    frontier = PriorityQueue(capacity=max(1, len(graph) * len(graph)))
    visited = set()
    order = 0
    frontier.push(Node(h_values[start], start, [start], 0.0, order))
    visited_order = []

    while not frontier.is_empty():
        current = frontier.pop()

        if current.node in visited:
            continue

        visited.add(current.node)
        visited_order.append(current.node)

        if current.node == goal:
            return visited_order, current.path, current.cost

        # Add neighbors to frontier
        for neighbor, step_cost in graph[current.node]:
            if neighbor not in visited:
                order += 1
                frontier.push(
                    Node(
                        h_values[neighbor],
                        neighbor,
                        current.path + [neighbor],
                        current.cost + step_cost,
                        order
                    )
                )

    raise ValueError(f"No route found from '{start}' to '{goal}'.")


def path_str(nodes):
    """Format path as A -> B -> C"""
    return " -> ".join(nodes)


def solve(input_path, output_path=None):
    """Run GBFS and return results"""
    graph, h_values, start, goal = read_input(input_path)

    if start is None:
        start = input("Enter Start Node: ").strip()
        check_graph(graph, h_values, start, goal)

    # Measure complexity
    tracemalloc.start()
    start_time = time.perf_counter()
    visited_order, path, total_cost = greedy_best_first_search(graph, h_values, start, goal)
    time_taken = time.perf_counter() - start_time
    _, mem_used = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    output = "\n".join([
        "Greedy Best First Search - Emergency Route Planning",
        f"Start Node: {start}",
        f"Goal Node: {goal}",
        f"Visited Nodes / Expansion Order: {path_str(visited_order)}",
        f"Optimal Path: {path_str(path)}",
        f"Total Cost: {total_cost:g}",
        f"Measured Time Complexity: {time_taken:.8f} seconds for this input",
        f"Measured Space Complexity: {mem_used / 1024:.2f} KB peak memory for this input",
    ])

    if output_path is not None:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output + "\n")

    return output


def main():
    """Main entry point"""
    input_path = sys.argv[1] if len(sys.argv) > 1 else "inputPS07.txt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        print(solve(input_path, output_path))
        return 0
    except (OSError, ValueError, IndexError, OverflowError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
 