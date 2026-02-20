from __future__ import annotations


class TrieNode:
    __slots__ = ("children", "is_word")

    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.is_word: bool = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_word = True


def load_trie(path: str, min_length: int = 3) -> Trie:
    trie = Trie()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().upper()
            if len(word) >= min_length and word.isalpha():
                trie.insert(word)
    return trie


def solve(board: list[list[str]], grid_size: int, trie: Trie, max_results: int = 50) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """Solve the Boggle board using DFS with Trie prefix pruning and bitmask visited tracking.

    Returns (words, positions) where positions maps each word to its
    topmost-leftmost starting cell (row, col).
    """
    found: set[str] = set()
    word_starts: dict[str, tuple[int, int]] = {}
    total_cells = grid_size * grid_size

    # Pre-expand cell values: most are single chars, "QU" is two chars
    cell_chars: list[str] = []
    for r in range(grid_size):
        for c in range(grid_size):
            cell_chars.append(board[r][c].upper())

    # Precompute adjacency lists
    neighbors: list[list[int]] = []
    for idx in range(total_cells):
        r, c = divmod(idx, grid_size)
        adj = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    adj.append(nr * grid_size + nc)
        neighbors.append(adj)

    def dfs(idx: int, node: TrieNode, path: list[str], visited: int, start_idx: int):
        chars = cell_chars[idx]
        if chars == "?":
            return

        # Walk trie through all characters in this cell (handles "QU")
        current = node
        for ch in chars:
            if ch not in current.children:
                return
            current = current.children[ch]

        path.append(chars)
        if current.is_word:
            word = "".join(path)
            found.add(word)
            sr, sc = divmod(start_idx, grid_size)
            # Keep the topmost-leftmost starting position
            if word not in word_starts or (sr, sc) < word_starts[word]:
                word_starts[word] = (sr, sc)

        if current.children:  # prune if no further prefixes
            for nidx in neighbors[idx]:
                if not (visited & (1 << nidx)):
                    dfs(nidx, current, path, visited | (1 << nidx), start_idx)

        path.pop()

    for start in range(total_cells):
        dfs(start, trie.root, [], 1 << start, start)

    # Sort: longest first, then alphabetical
    result = sorted(found, key=lambda w: (-len(w), w))
    result = result[:max_results] if max_results > 0 else result
    return result, word_starts
