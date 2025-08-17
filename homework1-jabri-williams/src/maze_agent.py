from typing import *
from environment import Environment
from maze_clause import MazeClause
from maze_knowledge_base import MazeKnowledgeBase
from constants import Constants
from itertools import combinations
import heapq

class MazeAgent:
    """
    MazeAgent uses logical inference and efficient path planning to navigate the Pitsweeper maze.
    It deduces safe tiles from perceptions and uses an A* search over confirmed safe tiles.
    When no complete safe path exists, it chooses among immediate neighbor moves the one that
    minimizes Manhattan distance to the goal (unless it’s confirmed to be a pit), so that it moves
    aggressively toward the goal and only stops if every immediate move is unsafe.
    """
    def __init__(self, env: Environment, perception: dict) -> None:
        self.env = env
        self.goal = env.get_goal_loc()
        self.maze = env.get_agent_maze()
        self.kb = MazeKnowledgeBase()
        self.possible_pits: set[tuple[int, int]] = set()
        self.safe_tiles: set[tuple[int, int]] = set()
        self.pit_tiles: set[tuple[int, int]] = set()

        # Mark starting position and goal as safe.
        start_loc = perception["loc"]
        self.safe_tiles.add(start_loc)
        self.kb.tell(MazeClause([(("P", start_loc), False)]))
        self.safe_tiles.add(self.goal)
        self.kb.tell(MazeClause([(("P", self.goal), False)]))

        # Ensure at least one neighbor of the goal is safe (game rule).
        goal_adj = [n for n in self.env.get_cardinal_locs(self.goal, 1)
                    if n in self.env.get_playable_locs()]
        if goal_adj:
            self.kb.tell(MazeClause([(("P", n), False) for n in goal_adj]))

        # Process the initial perception.
        self.think(perception)

    def think(self, perception: dict) -> tuple[int, int]:
        """
        Process the current perception to update internal knowledge and decide on the next move.
        Returns the chosen next location.
        """
        self._update_maze(perception)
        self._process_tile(perception)
        self._deduce_frontier()

        curr_loc = perception["loc"]

        # First try: compute a safe path using A* over confirmed safe tiles.
        path = self.a_star_path(curr_loc, self.goal)
        if path and len(path) > 1:
            return path[1]

        # Fallback: consider immediate cardinal moves from current location.
        return self._fallback_move(curr_loc)

    def _update_maze(self, perception: dict) -> None:
        """Update the agent's maze view with the current tile perception."""
        curr_loc = perception["loc"]
        tile = perception["tile"]
        self.maze[curr_loc[1]][curr_loc[0]] = tile

    def _process_tile(self, perception: dict) -> None:
        """
        Process the current tile to update the knowledge base.
        Safe tiles mark themselves as safe; warning tiles generate clauses about nearby pits.
        """
        curr_loc = perception["loc"]
        tile = perception["tile"]

        if tile == Constants.PIT_BLOCK:
            self.pit_tiles.add(curr_loc)
            self.kb.tell(MazeClause([(("P", curr_loc), True)]))
        else:
            self.safe_tiles.add(curr_loc)
            self.kb.tell(MazeClause([(("P", curr_loc), False)]))
            neighbors = list(self.env.get_cardinal_locs(curr_loc, 1))
            if tile in {Constants.SAFE_BLOCK, "0"}:
                # No pits adjacent: mark all neighbors safe.
                for n in neighbors:
                    self.kb.tell(MazeClause([(("P", n), False)]))
                    self.safe_tiles.add(n)
            elif tile in Constants.WRN_BLOCKS:
                # Warning tile: use the numeric hint to add exactly-k clauses.
                num_pits = int(tile)
                known_pits = [n for n in neighbors if n in self.pit_tiles]
                unknown = [n for n in neighbors if n not in self.safe_tiles and n not in self.pit_tiles]
                k_eff = num_pits - len(known_pits)
                if 0 <= k_eff <= len(unknown):
                    self._add_exactly_k_clauses(unknown, k_eff)

    def _deduce_frontier(self) -> None:
        """
        Query the knowledge base for frontier tiles (adjacent to explored areas) and update
        safe_tiles and pit_tiles accordingly.
        """
        frontier = self.env.get_frontier_locs()
        for loc in frontier:
            safety = self.is_safe_tile(loc)
            if safety is True:
                self.safe_tiles.add(loc)
            elif safety is False:
                self.pit_tiles.add(loc)
            else:
                self.possible_pits.add(loc)

    def _fallback_move(self, curr_loc: tuple[int, int]) -> tuple[int, int]:
        """
        Fallback move: from the immediate neighbors of the current location (that are playable),
        choose the one that minimizes Manhattan distance to the goal. If a candidate is confirmed as a pit,
        it is discarded. Only if every immediate move is unsafe will the agent remain in place.
        """
        # Get immediate cardinal neighbors that are in playable areas.
        neighbors = [n for n in self.env.get_cardinal_locs(curr_loc, 1)
                     if n in self.env.get_playable_locs()]
        # Filter out moves that are confirmed pits.
        valid = [n for n in neighbors if self.is_safe_tile(n) is not False]
        if valid:
            return min(valid, key=lambda loc: self._manhattan(loc, self.goal))
        return curr_loc

    def _add_exactly_k_clauses(self, locations: list[tuple[int, int]], k: int) -> None:
        """
        Add logical clauses to the knowledge base enforcing exactly k pits among the given locations.
        """
        if not locations or k < 0 or k > len(locations):
            return
        if k == 0:
            for loc in locations:
                self.kb.tell(MazeClause([(("P", loc), False)]))
                self.safe_tiles.add(loc)
        elif k == len(locations):
            for loc in locations:
                self.kb.tell(MazeClause([(("P", loc), True)]))
                self.pit_tiles.add(loc)
        else:
            for combo in combinations(locations, len(locations) - k + 1):
                self.kb.tell(MazeClause([(("P", loc), True) for loc in combo]))
            for combo in combinations(locations, k + 1):
                self.kb.tell(MazeClause([(("P", loc), False) for loc in combo]))

    def is_safe_tile(self, loc: tuple[int, int]) -> Optional[bool]:
        """
        Use the knowledge base to determine if a tile is safe (True), a pit (False), or undetermined (None).
        """
        if loc in self.safe_tiles:
            return True
        if loc in self.pit_tiles:
            return False
        no_pit_query = MazeClause([(("P", loc), False)])
        if self.kb.ask(no_pit_query):
            self.safe_tiles.add(loc)
            return True
        pit_query = MazeClause([(("P", loc), True)])
        if self.kb.ask(pit_query):
            self.pit_tiles.add(loc)
            return False
        return None

    def a_star_path(self, start: tuple[int, int], goal: tuple[int, int]) -> Optional[List[tuple[int, int]]]:
        """
        Compute a safe path from start to goal using the A* algorithm over confirmed safe tiles.
        """
        open_set = [(self._manhattan(start, goal), start)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], int] = {start: 0}
        f_score: dict[tuple[int, int], int] = {start: self._manhattan(start, goal)}
        closed_set: set[tuple[int, int]] = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in closed_set:
                continue
            closed_set.add(current)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            for neighbor in self.env.get_cardinal_locs(current, 1):
                if neighbor not in self.safe_tiles:
                    continue
                tentative_g = g_score[current] + 1
                if neighbor in closed_set and tentative_g >= g_score.get(neighbor, float('inf')):
                    continue
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._manhattan(neighbor, goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))
        return None

    def _manhattan(self, pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
        """Calculate the Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])




# import time
# import random
# import math
# import itertools
# from queue import Queue
# from constants import *
# import heapq
# from maze_clause import *
# from maze_knowledge_base import *
# from typing import List, Optional, Set, Dict, Tuple
#
#
# class MazeAgent:
#     def __init__(self, env: "Environment", perception: dict) -> None:
#         self.env: "Environment" = env
#         self.goal: tuple[int, int] = env.get_goal_loc()
#         self.maze: list = env.get_agent_maze()
#
#         # Knowledge tracking
#         self.kb: "MazeKnowledgeBase" = MazeKnowledgeBase()
#         self.possible_pits: set[tuple[int, int]] = set()
#         self.safe_tiles: set[tuple[int, int]] = set()
#         self.pit_tiles: set[tuple[int, int]] = set()
#         self.visited_tiles: set[tuple[int, int]] = set()
#
#         # Initialize with starting location
#         start_loc = perception["loc"]
#         self.kb.tell(MazeClause([(("P", start_loc), False)]))
#         self.safe_tiles.add(start_loc)
#         self.visited_tiles.add(start_loc)
#
#         # Mark goal and surroundings
#         self.kb.tell(MazeClause([(("P", self.goal), False)]))
#         self.safe_tiles.add(self.goal)
#
#         # Initial perception processing
#         if perception["tile"] in {".", "0"}:
#             for n in self.env.get_cardinal_locs(start_loc, 1):
#                 self.kb.tell(MazeClause([(("P", n), False)]))
#                 self.safe_tiles.add(n)
#
#     def process_warning_tile(self, loc: tuple[int, int], warning_count: int) -> None:
#         """Improved warning tile processing"""
#         neighbors = list(self.env.get_cardinal_locs(loc, 1))
#         known_pits = [n for n in neighbors if n in self.pit_tiles]
#         safe_neighbors = [n for n in neighbors if n in self.safe_tiles]
#         unknown = [n for n in neighbors if n not in self.safe_tiles and n not in self.pit_tiles]
#
#         # Adjust count based on known pits
#         count = int(warning_count) - len(known_pits)
#
#         # If warning is 0, all unknowns are safe
#         if count == 0:
#             for n in unknown:
#                 self.kb.tell(MazeClause([(("P", n), False)]))
#                 self.safe_tiles.add(n)
#             return
#
#         # If count equals remaining unknowns, all are pits
#         if count == len(unknown):
#             for n in unknown:
#                 self.kb.tell(MazeClause([(("P", n), True)]))
#                 self.pit_tiles.add(n)
#             return
#
#         # For warning tile = 1, no two adjacent unknowns can be pits
#         if count == 1:
#             # Add clause for each pair of unknowns
#             for u1, u2 in itertools.combinations(unknown, 2):
#                 self.kb.tell(MazeClause([(("P", u1), False), (("P", u2), False)]))
#             # At least one must be a pit
#             if unknown:
#                 self.kb.tell(MazeClause([(("P", u), True) for u in unknown]))
#
#         # For warning tile = 2 with 3 unknowns
#         elif count == 2 and len(unknown) == 3:
#             for u in unknown:
#                 others = [n for n in unknown if n != u]
#                 # Each unknown must be with at least one other
#                 self.kb.tell(MazeClause([(("P", u), True)] + [(("P", o), True) for o in others]))
#             # Not all three can be pits
#             self.kb.tell(MazeClause([(("P", u), False) for u in unknown]))
#
#     def get_move_cost(self, current: tuple[int, int], next_pos: tuple[int, int]) -> float:
#         """Better move cost calculation"""
#         manhattan = abs(current[0] - self.goal[0]) + abs(current[1] - self.goal[1])
#         safety = self.is_safe_tile(next_pos)
#
#         if safety is True:
#             return manhattan
#         if safety is False:
#             return float('inf')
#
#         # For unknown tiles, check warning counts
#         warning_count = 0
#         safe_neighbors = 0
#         for n in self.env.get_cardinal_locs(next_pos, 1):
#             if n in self.visited_tiles:
#                 tile = self.maze[n[1]][n[0]]
#                 if tile.isdigit():
#                     warning_count += int(tile)
#                 elif tile in {'.', '0'}:
#                     safe_neighbors += 1
#
#         risk = (warning_count * 3) - (safe_neighbors * 2)
#         return manhattan + max(0, risk)
#
#     def find_safe_path(self, start: tuple[int, int], goal: tuple[int, int]) -> Optional[List[tuple[int, int]]]:
#         """Find safest path to goal"""
#         if start == goal:
#             return [start]
#
#         queue = [(0, 0, start, [start])]  # (f_score, g_score, pos, path)
#         seen = {start: 0}
#
#         while queue:
#             _, g_score, pos, path = heapq.heappop(queue)
#
#             if pos == goal:
#                 return path
#
#             if pos in seen and g_score > seen[pos]:
#                 continue
#
#             for next_pos in self.env.get_cardinal_locs(pos, 1):
#                 if next_pos in self.pit_tiles:
#                     continue
#
#                 new_cost = g_score + self.get_move_cost(pos, next_pos)
#                 if new_cost == float('inf'):
#                     continue
#
#                 if next_pos not in seen or new_cost < seen[next_pos]:
#                     seen[next_pos] = new_cost
#                     h_cost = abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])
#                     f_score = new_cost + h_cost
#                     heapq.heappush(queue, (f_score, new_cost, next_pos, path + [next_pos]))
#
#         return None
#
#     def choose_move(self, frontier: set[tuple[int, int]], current_loc: tuple[int, int]) -> tuple[int, int]:
#         """Improved move selection"""
#         scored_moves = []
#
#         for pos in frontier:
#             if pos in self.pit_tiles:
#                 continue
#
#             score = 0
#             safety = self.is_safe_tile(pos)
#
#             if safety is True:
#                 score += 10000
#             elif safety is None:
#                 # Check warning density
#                 warnings = 0
#                 safe_count = 0
#                 for n in self.env.get_cardinal_locs(pos, 1):
#                     if n in self.visited_tiles:
#                         tile = self.maze[n[1]][n[0]]
#                         if tile.isdigit():
#                             warnings += int(tile)
#                         elif tile in {'.', '0'}:
#                             safe_count += 1
#
#                 if warnings == 0:
#                     score += 5000
#                 else:
#                     score += 1000 - (warnings * 200)
#                 score += safe_count * 500
#
#             # Goal proximity bonus
#             goal_dist = abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])
#             score -= goal_dist * 100
#
#             # Movement cost penalty
#             move_cost = self.get_move_cost(current_loc, pos)
#             score -= move_cost * 50
#
#             # Exploration bonus
#             if pos not in self.visited_tiles:
#                 score += 2000
#                 # Extra bonus for moves that open new areas
#                 unvisited = sum(1 for n in self.env.get_cardinal_locs(pos, 1)
#                                 if n not in self.visited_tiles)
#                 score += unvisited * 300
#
#             heapq.heappush(scored_moves, (-score, pos))
#
#         return scored_moves[0][1] if scored_moves else list(frontier)[0]
#
#     def think(self, perception: dict) -> tuple[int, int]:
#         """Process new information and choose next move"""
#         current_loc = perception["loc"]
#         current_tile = perception["tile"]
#
#         # Update knowledge
#         self.visited_tiles.add(current_loc)
#         self.safe_tiles.add(current_loc)
#         self.kb.tell(MazeClause([(("P", current_loc), False)]))
#
#         # Process tile information
#         if current_tile in {".", "0"}:
#             for n in self.env.get_cardinal_locs(current_loc, 1):
#                 self.kb.tell(MazeClause([(("P", n), False)]))
#                 self.safe_tiles.add(n)
#         elif current_tile.isdigit():
#             self.process_warning_tile(current_loc, int(current_tile))
#
#         # Optimize KB and update knowledge
#         self.kb.simplify_self(self.pit_tiles, self.safe_tiles)
#
#         # Check frontier locations
#         frontier = self.env.get_frontier_locs()
#         for pos in frontier:
#             if pos not in self.safe_tiles and pos not in self.pit_tiles:
#                 self.is_safe_tile(pos)
#
#         # Try safe path to goal
#         path = self.find_safe_path(current_loc, self.goal)
#         if path and len(path) > 1:
#             next_pos = path[1]
#             if next_pos not in self.pit_tiles:
#                 return next_pos
#
#         # Choose best frontier move
#         return self.choose_move(frontier, current_loc)
#
#     def is_safe_tile(self, loc: tuple[int, int]) -> Optional[bool]:
#         """Determine if a tile is safe, unsafe, or unknown"""
#         if loc in self.safe_tiles:
#             return True
#         if loc in self.pit_tiles:
#             return False
#
#         # Query KB for safety
#         no_pit = MazeClause([(("P", loc), False)])
#         if self.kb.ask(no_pit):
#             self.safe_tiles.add(loc)
#             return True
#
#         # Query KB for danger
#         has_pit = MazeClause([(("P", loc), True)])
#         if self.kb.ask(has_pit):
#             self.pit_tiles.add(loc)
#             return False
#
#         return None
#
#
# from environment import Environment




# import time
# import random
# import math
# from queue import Queue
# from constants import *
# import heapq
# from maze_clause import *
# from maze_knowledge_base import *
#
# class MazeAgent:
#     '''
#     BlindBot MazeAgent meant to employ Propositional Logic,
#     Planning, and Active Learning to navigate the Pitsweeper
#     Problem. Have fun!
#     '''
#
#     def __init__ (self, env: "Environment", perception: dict) -> None:
#         """
#         Initializes the MazeAgent with any attributes it will need to
#         navigate the maze.
#         [!] Add as many attributes as you see fit!
#
#         Parameters:
#             env (Environment):
#                 The Environment in which the agent is operating; make sure
#                 to see the spec / Environment class for public methods that
#                 your agent will use to solve the maze!
#             perception (dict):
#                 The starting perception of the agent, which is a
#                 small dictionary with keys:
#                   - loc:  the location of the agent as a (c,r) tuple
#                   - tile: the type of tile the agent is currently standing upon
#         """
#         self.env: "Environment" = env
#         self.goal: tuple[int, int] = env.get_goal_loc()
#
#         # The agent's maze can be manipulated as a tracking mechanic
#         # for what it has learned; changes to this maze will be drawn
#         # by the environment and is simply for visuals / debugging
#         # [!] Feel free to change self.maze at will
#         self.maze: list = env.get_agent_maze()
#
#         # Standard set of attributes you'll want to maintain
#         self.kb: "MazeKnowledgeBase" = MazeKnowledgeBase()
#         self.possible_pits: set[tuple[int, int]] = set()
#         self.safe_tiles: set[tuple[int, int]] = set()
#         self.pit_tiles: set[tuple[int, int]] = set()
#
#         # [!] TODO: Initialize any other knowledge-related attributes for
#         # agent here, or any other record-keeping attributes you'd like
#         # Mark the starting location as safe.
#         start_loc = perception["loc"]
#         self.kb.tell(MazeClause([(("P", start_loc), False)]))
#         self.safe_tiles.add(start_loc)
#         # Also, mark the goal as safe (by assumption).
#         self.kb.tell(MazeClause([(("P", self.goal), False)]))
#         self.safe_tiles.add(self.goal)
#
#     ##################################################################
#     # Methods
#     ##################################################################
#
#     def think(self, perception: dict) -> tuple[int, int]:
#         """
#         The main workhorse method of how your agent will process new information
#         and use that to make deductions and decisions. In gist, it should follow
#         this outline of steps:
#         1. Process the given perception, i.e., the new location it is in and the
#            type of tile on which it's currently standing (e.g., a safe tile, or
#            warning tile like "1" or "2")
#         2. Update the knowledge base and record-keeping of where known pits and
#            safe tiles are located, as well as locations of possible pits.
#         3. Query the knowledge base to see if any locations that possibly contain
#            pits can be deduced as safe or not.
#         4. Use all of the above to prioritize the next location along the frontier
#            to move to next.
#
#         Parameters:
#             perception (dict):
#                 A dictionary providing the agent's current location
#                 and current tile type being stood upon, of the format:
#                 {"loc": (x, y), "tile": tile_type}
#
#         Returns:
#             tuple[int, int]:
#                 The maze location along the frontier that your agent will try to
#                 move into next.
#         """
#         loc = perception["loc"]
#         tile = perception["tile"]
#         # Update internal maze representation (note: maze is row-major)
#         self.maze[loc[1]][loc[0]] = tile
#         self.safe_tiles.add(loc)
#         self.kb.tell(MazeClause([(("P", loc), False)]))  # current location is safe
#
#         # If the tile is safe (no adjacent pits): treat both "." and "0" as safe.
#         if tile in {".", "0"}:
#             for neighbor in self.env.get_cardinal_locs(loc, 1):
#                 self.kb.tell(MazeClause([(("P", neighbor), False)]))
#                 self.safe_tiles.add(neighbor)
#         # Else, if the tile is a warning tile ("1", "2", "3", or "4")
#         elif tile in {"1", "2", "3", "4"}:
#             count = int(tile)
#             neighbors = list(self.env.get_cardinal_locs(loc, 1))
#             # Only consider candidates not already known safe.
#             candidates = [n for n in neighbors if n not in self.safe_tiles]
#
#             if len(candidates) == 3:
#                 X, Y, Z = candidates
#                 if count == 1:
#                     # Exactly 1 pit among 3 candidates.
#                     # CNF: (¬X∨¬Y) ∧ (¬X∨¬Z) ∧ (¬Y∨¬Z) ∧ (X∨Y∨Z)
#                     self.kb.tell(MazeClause([(("P", X), False), (("P", Y), False)]))
#                     self.kb.tell(MazeClause([(("P", X), False), (("P", Z), False)]))
#                     self.kb.tell(MazeClause([(("P", Y), False), (("P", Z), False)]))
#                     self.kb.tell(MazeClause([(("P", X), True), (("P", Y), True), (("P", Z), True)]))
#                 elif count == 2:
#                     # Exactly 2 pits among 3 candidates.
#                     # CNF: (X∨Y) ∧ (X∨Z) ∧ (Y∨Z) ∧ (¬X∨¬Y∨¬Z)
#                     self.kb.tell(MazeClause([(("P", X), True), (("P", Y), True)]))
#                     self.kb.tell(MazeClause([(("P", X), True), (("P", Z), True)]))
#                     self.kb.tell(MazeClause([(("P", Y), True), (("P", Z), True)]))
#                     self.kb.tell(MazeClause([(("P", X), False), (("P", Y), False), (("P", Z), False)]))
#                 elif count == 3:
#                     # All three candidates must be pits.
#                     for candidate in candidates:
#                         self.kb.tell(MazeClause([(("P", candidate), True)]))
#                         self.pit_tiles.add(candidate)
#                 # (count == 4 is not possible with 3 candidates)
#             else:
#                 # If candidates count is not exactly 3, use simple rules:
#                 if count == 0:
#                     for candidate in candidates:
#                         self.kb.tell(MazeClause([(("P", candidate), False)]))
#                         self.safe_tiles.add(candidate)
#                 elif count == len(candidates):
#                     for candidate in candidates:
#                         self.kb.tell(MazeClause([(("P", candidate), True)]))
#                         self.pit_tiles.add(candidate)
#                 # Else, ambiguous cases: do not add clauses.
#         # Optimize KB by simplifying clauses given known safe and pit locations.
#         self.kb.simplify_self(self.pit_tiles, self.safe_tiles)
#
#         # Plan a path from current location to goal using A* on safe tiles.
#         path = self.a_star_path(loc, self.goal)
#         if path and len(path) > 1:
#             # Next move is the immediate step in the planned path.
#             return path[1]
#
#         # Fallback: choose a frontier cell that is deduced safe.
#         frontier = self.env.get_frontier_locs()
#         safe_frontier = [pos for pos in frontier if self.is_safe_tile(pos) is True]
#         if safe_frontier:
#             return min(safe_frontier, key=lambda pos: abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1]))
#         # [!] TODO! Agent is currently just making a random choice from the
#         # frontier -- use logic and your own strategy to improve this!
#         return random.choice(list(frontier))
#
#     def is_safe_tile (self, loc: tuple[int, int]) -> Optional[bool]:
#         """
#         Determines whether or not the given maze location can be concluded as
#         safe (i.e., not containing a pit), following the steps:
#         1. Check to see if the location is already a known pit or safe tile,
#            responding accordingly
#         2. If not, performs the necessary queries on the knowledge base in an
#            attempt to deduce its safety
#
#         Parameters:
#             loc (tuple[int, int]):
#                 The maze location in question
#
#         Returns:
#             One of three return values:
#             1. True if the location is certainly safe (i.e., not pit)
#             2. False if the location is certainly dangerous (i.e., pit)
#             3. None if the safety of the location cannot be currently determined
#         """
#         # [!] TODO! Agent is currently dumb; this method should perform queries
#         # on the agent's knowledge base from its gathered perceptions
#         if loc in self.safe_tiles:
#             return True
#         if loc in self.pit_tiles:
#             return False
#
#         # Query KB for "no pit at loc".
#         no_pit_clause = MazeClause([(("P", loc), False)])
#         if self.kb.ask(no_pit_clause):
#             self.safe_tiles.add(loc)
#             return True
#
#         # Query KB for "pit at loc".
#         pit_clause = MazeClause([(("P", loc), True)])
#         if self.kb.ask(pit_clause):
#             self.pit_tiles.add(loc)
#             return False
#
#         return None
#     def a_star_path(self, start: tuple[int, int], goal: tuple[int, int]) -> Optional[List[tuple[int, int]]]:
#         """
#         Performs A* search over safe tiles to find a path from start to goal.
#         Uses Manhattan distance as the heuristic and assumes uniform step cost (1 per move).
#         """
#         open_set = []
#         heapq.heappush(open_set, (0, start))
#         came_from = {}
#         g_score = {start: 0}
#         def manhattan(a: tuple[int,int], b: tuple[int,int]) -> int:
#             return abs(a[0]-b[0]) + abs(a[1]-b[1])
#         f_score = {start: manhattan(start, goal)}
#         closed_set = set()
#         while open_set:
#             current = heapq.heappop(open_set)[1]
#             if current == goal:
#                 path = [current]
#                 while current in came_from:
#                     current = came_from[current]
#                     path.append(current)
#                 path.reverse()
#                 return path
#             closed_set.add(current)
#             for neighbor in self.env.get_cardinal_locs(current, 1):
#                 if neighbor not in self.safe_tiles:
#                     continue
#                 tentative_g = g_score[current] + 1
#                 if neighbor in closed_set and tentative_g >= g_score.get(neighbor, float('inf')):
#                     continue
#                 if tentative_g < g_score.get(neighbor, float('inf')):
#                     came_from[neighbor] = current
#                     g_score[neighbor] = tentative_g
#                     f = tentative_g + manhattan(neighbor, goal)
#                     f_score[neighbor] = f
#                     heapq.heappush(open_set, (f, neighbor))
#         return None
#
# # Declared here to avoid circular dependency
# from environment import Environment