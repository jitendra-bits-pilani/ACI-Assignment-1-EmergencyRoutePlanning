"""
BITS WILP ACI Assignment 1 - PS07
Group Id: G087

Emergency Route Planning using Greedy Best First Search.

This program models a city map as a graph and uses Greedy Best First Search
to guide an emergency rescue team from a user-provided start location to the
goal location. The search priority is based only on the heuristic value h(n),
as required in the problem statement.

Input format:
    START A
    GOAL G
    EDGES
    A B:4 C:2
    B D:5 E:1
    ...
    HEURISTICS
    A 10
    B 8
    ...

Edge costs are optional. If omitted, the cost is treated as 1.
Run:
    python gbfs_emergency_route.py inputPS07.txt outputPS07.txt
"""

from __future__ import annotations

import heapq
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


Graph = Dict[str, List[Tuple[str, float]]]
Heuristics = Dict[str, float]


@dataclass(order=True)
class QueueItem:
    """One route candidate stored in the GBFS frontier priority queue."""

    heuristic: float
    order: int
    node: str = field(compare=False)
    path: List[str] = field(compare=False)
    cost: float = field(compare=False)


class BoundedPriorityQueue:
    """
    Min-priority queue used by GBFS.

    The assignment asks data-structure insert/delete operations to report
    appropriate messages when the structure is full or empty. This wrapper
    provides those checks around Python's heapq implementation.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Priority queue capacity must be greater than zero.")
        self.capacity = capacity
        self._items: List[QueueItem] = []

    def push(self, item: QueueItem) -> None:
        """Insert a new route candidate, or raise an error if capacity is full."""
        if len(self._items) >= self.capacity:
            raise OverflowError("Priority queue is full; cannot insert a new route.")
        heapq.heappush(self._items, item)

    def pop(self) -> QueueItem:
        """Delete and return the candidate with the lowest heuristic value."""
        if not self._items:
            raise IndexError("Priority queue is empty; cannot delete a route.")
        return heapq.heappop(self._items)

    def is_empty(self) -> bool:
        """Return True when there are no more nodes left for expansion."""
        return len(self._items) == 0


def clean_token(token: str) -> str:
    """Remove common punctuation around node tokens read from the input file."""
    return token.strip().strip(",").strip()


def parse_neighbor(token: str) -> Tuple[str, float]:
    """Convert a neighbor token such as 'B:4' or 'B' into (node, cost)."""
    token = clean_token(token)
    if ":" in token:
        node, cost_text = token.split(":", 1)
        if not node:
            raise ValueError(f"Invalid edge entry: {token}")
        return node, float(cost_text)
    return token, 1.0


def parse_named_node(line: str, keyword: str) -> Optional[str]:
    """Read START/GOAL lines while accepting minor syntax variants."""
    normalized = line.replace(":", " ").replace("=", " ")
    parts = normalized.split()
    if not parts or parts[0].upper() != keyword:
        return None
    if len(parts) >= 3 and parts[1].upper() == "NODE":
        return parts[2]
    if len(parts) == 2:
        return parts[1]
    raise ValueError(f"{keyword.title()} must be followed by one node.")


def parse_input_file(path: Path) -> Tuple[Graph, Heuristics, Optional[str], str]:
    """
    Parse the assignment input file into graph, heuristic, start, and goal data.

    Supported sections are EDGES and HEURISTICS. START is optional in the file:
    if it is absent, the program asks the user to enter it at runtime, matching
    the assignment requirement that the starting point can be obtained as input.
    """
    graph: Graph = {}
    heuristics: Heuristics = {}
    start: Optional[str] = None
    goal: Optional[str] = None
    section: Optional[str] = None

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
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
        parsed_start = parse_named_node(line, "START")
        if parsed_start is not None:
            start = parsed_start
            continue
        parsed_goal = parse_named_node(line, "GOAL")
        if parsed_goal is not None:
            goal = parsed_goal
            continue

        parts = [clean_token(part) for part in line.replace(",", " ").split() if clean_token(part)]
        if section == "EDGES":
            source = parts[0].rstrip(":")
            graph.setdefault(source, [])
            for token in parts[1:]:
                neighbor, cost = parse_neighbor(token)
                if cost < 0:
                    raise ValueError(f"Line {line_number}: edge cost cannot be negative.")
                graph[source].append((neighbor, cost))
                graph.setdefault(neighbor, [])
        elif section == "HEURISTICS":
            if len(parts) == 1 and ":" in parts[0]:
                parts = [part.strip() for part in parts[0].split(":", 1)]
            if len(parts) != 2:
                raise ValueError(f"Line {line_number}: heuristic entry must be '<node> <value>'.")
            heuristics[parts[0].rstrip(":")] = float(parts[1])
        else:
            raise ValueError(f"Line {line_number}: content must appear under EDGES or HEURISTICS.")

    if goal is None:
        raise ValueError("Goal node is missing. Add a line like 'GOAL G'.")
    validate_problem(graph, heuristics, start, goal)
    return graph, heuristics, start, goal


def validate_problem(graph: Graph, heuristics: Heuristics, start: Optional[str], goal: str) -> None:
    """Validate that all graph nodes have heuristics and start/goal are valid."""
    nodes = set(graph) | {neighbor for edges in graph.values() for neighbor, _ in edges}
    missing_heuristics = sorted(nodes - set(heuristics))
    if missing_heuristics:
        raise ValueError("Heuristic value missing for node(s): " + ", ".join(missing_heuristics))
    if start is not None and start not in nodes:
        raise ValueError(f"Start node '{start}' is not present in the graph.")
    if goal not in nodes:
        raise ValueError(f"Goal node '{goal}' is not present in the graph.")


def greedy_best_first_search(
    graph: Graph, heuristics: Heuristics, start: str, goal: str
) -> Tuple[List[str], List[str], float]:
    """
    Perform Greedy Best First Search using f(n) = h(n).

    Returns:
        expansion_order: nodes in the order they were visited.
        path: route returned from start to goal.
        total_cost: sum of edge costs on the returned route.
    """
    if start not in graph:
        raise ValueError(f"Start node '{start}' is not present in the graph.")

    # A bounded frontier demonstrates explicit full/empty handling while keeping
    # enough capacity for all candidates generated by typical adjacency-list inputs.
    frontier = BoundedPriorityQueue(capacity=max(1, len(graph) * len(graph)))
    visited: Set[str] = set()
    insertion_order = 0
    frontier.push(QueueItem(heuristics[start], insertion_order, start, [start], 0.0))
    expansion_order: List[str] = []

    while not frontier.is_empty():
        current = frontier.pop()
        if current.node in visited:
            continue

        # GBFS commits to expanding the lowest-h(n) frontier node next.
        visited.add(current.node)
        expansion_order.append(current.node)

        if current.node == goal:
            return expansion_order, current.path, current.cost

        for neighbor, step_cost in graph[current.node]:
            if neighbor not in visited:
                insertion_order += 1
                frontier.push(
                    QueueItem(
                        heuristics[neighbor],
                        insertion_order,
                        neighbor,
                        current.path + [neighbor],
                        current.cost + step_cost,
                    )
                )

    raise ValueError(f"No route found from '{start}' to '{goal}'.")


def format_path(nodes: Iterable[str]) -> str:
    """Format a list of nodes in the readable A -> B -> C style."""
    return " -> ".join(nodes)


def solve(input_path: Path, output_path: Optional[Path] = None) -> str:
    """Run the complete workflow and optionally write the output text file."""
    graph, heuristics, start, goal = parse_input_file(input_path)
    if start is None:
        start = input("Enter Start Node: ").strip()
        validate_problem(graph, heuristics, start, goal)

    # The assignment asks for measured complexity values in the implementation.
    tracemalloc.start()
    start_time = time.perf_counter()
    expansion_order, path, total_cost = greedy_best_first_search(graph, heuristics, start, goal)
    elapsed = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    output = "\n".join(
        [
            "Greedy Best First Search - Emergency Route Planning",
            f"Start Node: {start}",
            f"Goal Node: {goal}",
            f"Visited Nodes / Expansion Order: {format_path(expansion_order)}",
            f"Optimal Path: {format_path(path)}",
            f"Total Cost: {total_cost:g}",
            f"Measured Time Complexity: {elapsed:.8f} seconds for this input",
            f"Measured Space Complexity: {peak_memory / 1024:.2f} KB peak memory for this input",
        ]
    )
    if output_path is not None:
        output_path.write_text(output + "\n", encoding="utf-8")
    return output


def main() -> int:
    """Command-line entry point with user-friendly error reporting."""
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("inputPS07.txt")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    try:
        print(solve(input_path, output_path))
        return 0
    except (OSError, ValueError, IndexError, OverflowError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
