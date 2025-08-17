from maze_clause import MazeClause
import itertools
from copy import deepcopy

class MazeKnowledgeBase:
    '''
    A Conjunctive Normal Form (CNF) propositional logic knowledge base for
    tracking pit and safe tile locations in the Pitsweeper maze problem.
    Stores clauses as a set and supports adding facts and querying entailment.
    '''

    def __init__(self) -> None:
        # Initialize an empty KB to store MazeClauses
        self.clauses: set["MazeClause"] = set()  # Set of clauses in CNF (AND of ORs)

    def tell(self, clause: "MazeClause") -> None:
        # Add a new clause to the KB
        self.clauses.add(clause)  # Assumes clause doesn’t make KB inconsistent (for efficiency)

    def ask(self, query: "MazeClause") -> bool:
        # Check if the KB entails the query using proof by contradiction
        clauses = set(deepcopy(self.clauses))  # Copy current clauses to avoid modifying KB
        negated_query = MazeClause([(p, not v) for p, v in query.props.items()])  # Negate query (e.g., P becomes ~P)
        clauses.add(negated_query)  # Add ~query to KB copy
        new_resolvents = set()  # Track new clauses from resolution
        while True:
            clause_combo = itertools.combinations(clauses, 2)  # All pairs of clauses
            for c1, c2 in clause_combo:
                resolvent = MazeClause.resolve(c1, c2)  # Resolve pair to find new clauses
                if any(r.is_empty() for r in resolvent):  # Empty clause means contradiction
                    return True  # KB ∧ ~query is inconsistent, so KB entails query
                new_resolvents.update(resolvent)  # Add non-empty resolvents
            if new_resolvents.issubset(clauses):  # No new info: no contradiction found
                return False  # KB doesn’t entail query
            clauses.update(new_resolvents)  # Add new resolvents and continue

    def __len__(self) -> int:
        # Return the number of clauses in the KB
        return len(self.clauses)  # Simple count of stored clauses

    def __str__(self) -> str:
        # Convert KB to a string for debugging (lists all clauses)
        return str([str(clause) for clause in self.clauses])  # Stringify each clause in a list

    # Optimization Methods for Part 2
    # -----------------------------------------------------------------------------------------

    def simplify_self(self, known_pits: set[tuple[int, int]], known_safe: set[tuple[int, int]]) -> None:
        # Simplify KB clauses using known pit and safe tile locations
        self.clauses = MazeKnowledgeBase.simplify_from_known_locs(self.clauses, known_pits, known_safe)  # Update KB in place

    @staticmethod
    def simplify_from_known_locs(clauses: set["MazeClause"], known_pits: set[tuple[int, int]], known_safe: set[tuple[int, int]]) -> set["MazeClause"]:
        # Reduce clause complexity based on known pit/safe locations
        for loc in known_pits | known_safe:  # Union of pits and safe tiles
            clauses = MazeKnowledgeBase.get_simplified_clauses(clauses, loc, loc in known_pits)  # Simplify for each location
        return clauses  # Return simplified clause set

    @staticmethod
    def get_simplified_clauses(clauses: set["MazeClause"], loc: tuple[int, int], is_pit: bool) -> set["MazeClause"]:
        # Simplify clauses given a location’s pit status
        to_add = set()  # New clauses from resolution
        to_rem = set()  # Clauses to remove (simplified away)
        sani_clause = MazeClause([(("P", loc), is_pit)])  # Clause asserting loc’s status (e.g., P or ~P)
        for clause in clauses:
            if len(clause) == 1:  # Skip unit clauses (already simple)
                continue
            if clause.get_prop(("P", loc)) == is_pit:  # Clause agrees with known fact (e.g., P when pit)
                to_rem.add(clause)  # Remove as it’s redundant
                break
            to_add.update(MazeClause.resolve(clause, sani_clause))  # Resolve to simplify clause
        clauses = clauses | to_add  # Add new resolvents
        clauses = clauses - to_rem  # Remove redundant clauses
        return clauses  # Return updated clause set