from __future__ import annotations

import numpy as np



class QuantizationTree:
    _elements: list[tuple[int, str | QuantizationTree]]

    def __init__(self, level_count: int, elements: list[tuple[int, str | QuantizationTree]]):
        assert isinstance(elements, list)
        assert isinstance(level_count, int)
        assert level_count >= 2
        assert all(isinstance(e, tuple) for e in elements)
        assert all(len(e) == 2 for e in elements)
        assert all(isinstance(e[0], int) for e in elements)
        assert all(isinstance(e[1], str) or isinstance(e[1], QuantizationTree) for e in elements)
        assert sum(e[0] for e in elements) == level_count
        self._elements = elements

    def __len__(self):
        return len(self._elements)

    def __getitem__(self, index):
        return self._elements[index]

    def __iter__(self):
        return iter(self._elements)

    def repr(self, nest_level: int) -> str:
        assert isinstance(nest_level, int)
        assert nest_level >= 0
        if len(self._elements) == 1 and isinstance(self._elements[0][1], str):
            return self._elements[0][1]

        ans = []

        for e in self._elements:
            if isinstance(e[1], str):
                ans += f"\n{'  ' * nest_level}{e[0]}x'{e[1]}"
            else:
                ans += f"\n{'  ' * nest_level}{e[0]}x"
                ans += e[1].repr(nest_level + 1)
        return "".join(ans)

    def __repr__(self) -> str:
        """Draws a tree using ASCII art"""
        return self.repr(0)

    def _add_self_to_tree(self, tree, self_name: str): # tree: nx.classes.digraph.DiGraph
        """Returns a directed graph that represents the tree"""
        tree.add_node(self_name)
        for i, e in enumerate(self._elements):
            name = f"{e[0]}x{e[1]}"
            if isinstance(e[1], int):
                tree.add_node(name)
            else:
                assert isinstance(e[1], QuantizationTree)
                e[1]._add_self_to_tree(tree, name)
            tree.add_edge(self_name, name)

    def make_tree(self): # -> nx.classes.digraph.DiGraph:
        """Returns a directed graph that represents the tree"""
        import networkx as nx
        tree = nx.DiGraph()
        self._add_self_to_tree(tree, "root")
        return tree

    def plot(self, filename: str = "plot.html"): # -> go.FigureWidget:
        """Returns a directed graph that represents the tree"""
        from pyvis.network import Network as net
        tree = self.make_tree()
        nt = net(notebook=True, directed=True, cdn_resources='in_line')
        nt.from_nx(tree)
        return nt.show(filename)

    def __lt__(self, other: QuantizationTree) -> bool:
        """Compares two trees by their representation"""
        return repr(self) < repr(other)

    @property
    def children_size(self) -> int:
        """Returns the number of children. Usually a constant equal to the level_count, but it may be specific to the node"""
        return sum(e[0] for e in self._elements)

    def find_node_size(self, tree: QuantizationTree, node: str) -> float:
        """Returns the relative size of the node"""
        for count, child in self._elements:
            if isinstance(child, str):
                if child == node:
                    return count / self.children_size
            else:
                ans = child.find_node_size(tree, node)
                if ans is not np.nan:
                    return count * ans / self.children_size

        # Return NaN if not found
        return np.nan

