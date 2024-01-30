import heapq

import numpy as np

from .tree import QuantizationTree


def simplify_simplex(simplex: np.ndarray, level_count: int) -> np.ndarray:
    """Simplifies the simplex by using a best tree representation of the simplex in a similar fashion to how the Huffman coding works.
    level_count - number of children for each tree node. Chose 2 for Huffman coding."""

    indices_of_non_zero = np.nonzero(simplex)[0]
    labels = [f"{i}" for i in indices_of_non_zero]
    tree = rational_simplification_of_weights(simplex[indices_of_non_zero], level_count, labels=labels)
    output = np.zeros(len(simplex), dtype=float)
    for i, index in enumerate(indices_of_non_zero):
        output[index] = tree.find_node_size(tree, labels[i])

    return output


def rational_simplification_of_weights(simplex: np.ndarray, level_count: int, labels: list = None) -> QuantizationTree:
    """
    Simplifies the simplex by grouping the least frequent elements together. If level_count == 2, the simplification
    is the same as the frequency implied by Huffman coding. If level_count > 2, the simplification is different (but not necessarily better).
    """
    assert isinstance(simplex, np.ndarray)
    assert simplex.ndim == 1
    assert isinstance(level_count, int)
    assert level_count >= 2
    assert np.all(simplex > 0)
    if labels is None:
        labels = [str(i) for i in range(len(simplex))]
    assert isinstance(labels, list)
    assert len(labels) == len(simplex)

    q = [(float(simplex[i]), QuantizationTree(level_count, [(level_count, labels[i])])) for i in range(len(simplex))]
    heapq.heapify(q)

    while len(q) > 1:
        # Peek at the next `level_count` elements without removing them from the queue.

        cur_level_count = min(level_count, len(q))

        elements = heapq.nsmallest(cur_level_count, q)

        element_sizes = np.asarray([e[0] for e in elements])  # Elements sorted by size in ascending order

        # Running sum of elements:
        running_elements = np.cumsum(element_sizes)

        scores = np.zeros(cur_level_count - 1, dtype=float)
        min_score = 0
        proposed_partitionings: list[QuantizationTree] = []

        for count in range(2, cur_level_count + 1):  # We are guaranteed at least one iteration
            # Distribute the first `count` elements of the `elements` array into the `level_count` bins, proportional to their value.
            score = 0
            partition: list[tuple[int, QuantizationTree]] = []
            whole_pie = float(level_count)  # The 100% of the total weight that we are distributing
            running_sum_residuum = running_elements[count - 1]
            for i in range(count - 1):
                # The first `i` elements go to the first bin:
                element_count_float = element_sizes[i] / running_sum_residuum * whole_pie
                element_count_int = int(np.round(element_count_float))  # We may want to make sure, the min is 1, not 0.
                element_count_float = element_sizes[i] / running_elements[count - 1] * whole_pie
                score += np.abs((element_count_float - element_count_int))
                running_sum_residuum -= element_sizes[i]  # Remove the element from the running sum
                whole_pie -= element_count_int
                partition.append((int(element_count_int), elements[i][1]))

            assert np.isclose(whole_pie + sum(e[0] for e in partition), level_count)
            assert np.isclose(running_sum_residuum, element_sizes[count - 1])
            element_count_float = element_sizes[count - 1] / running_sum_residuum * whole_pie
            element_count_int = int(np.round(element_count_float))  # We may want to make sure, the min is 1, not 0.
            element_count_float = element_sizes[count - 1] / running_elements[count - 1] * whole_pie
            score += np.abs((element_count_float - element_count_int))
            partition.append((int(element_count_int), elements[count - 1][1]))

            scores[count - 2] = score
            if score > min_score:
                min_score = score
            proposed_partitionings.append(QuantizationTree(level_count, partition))

        # Choose the partitioning with the smallest score:
        index_of_best_partitioning = np.argmin(scores)
        best_partitioning = proposed_partitionings[index_of_best_partitioning]

        # Remove the elements from the queue:
        for e in range(len(best_partitioning)):
            heapq.heappop(q)

        # Add the partition to the queue
        # noinspection PyTypeChecker
        heapq.heappush(q, (running_elements[index_of_best_partitioning + 1], best_partitioning))

    ans = q[0]

    assert len(q) == 1
    return ans[1]
