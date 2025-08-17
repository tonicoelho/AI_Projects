from typing import *

class MazeClause:
    '''
    Represents a propositional logic clause tailored for the Pitsweeper maze,
    where each clause is a disjunction (OR) of propositions (e.g., "P at (1,1)")
    mapped to their truth values (True = positive, False = negated).
    '''

    def __init__(self, props: Sequence[tuple]):
        # Initialize a clause from a list of (proposition, truth_value) tuples
        self.props: dict[tuple[str, tuple[int, int]], bool] = dict()  # Stores props as {prop: truth_value}
        self.valid: bool = False  # True if clause is always true (e.g., P v ~P)

        # Build the clause dictionary and check for validity
        for prop, truth_val in props:
            if prop in self.props and self.props[prop] != truth_val:  # Check for contradiction (e.g., P and ~P)
                self.valid = True  # Clause is valid (always true) if contradictory
                self.props = {}  # Clear props since valid clauses don’t constrain
                return
            self.props[prop] = truth_val  # Add prop with its truth value

    def get_prop(self, prop: tuple[str, tuple[int, int]]) -> Optional[bool]:
        # Retrieve the truth value of a specific proposition in the clause
        return None if prop not in self.props else self.props.get(prop)  # None if not present, else True/False

    def is_valid(self) -> bool:
        # Check if the clause is always true (valid), e.g., (P v ~P)
        return self.valid  # Returns True if a contradiction was found during init

    def is_empty(self) -> bool:
        # Check if the clause is empty (represents a contradiction, e.g., resolved to nothing)
        return (not self.valid) and (len(self.props) == 0)  # True only if non-valid and no props

    def __eq__(self, other: Any) -> bool:
        # Compare two MazeClauses for equality based on props and validity
        if other is None or not isinstance(other, MazeClause):
            return False  # Not equal if other isn’t a MazeClause
        return frozenset(self.props.items()) == frozenset(other.props.items()) and self.valid == other.valid  # Same props and validity

    def __hash__(self) -> int:
        # Generate a hash for set operations (e.g., storing in resolvents)
        return hash((frozenset(self.props.items()), self.valid))  # Hash based on props and validity

    def _prop_str(self, prop: tuple[str, tuple[int, int]]) -> str:
        # Format a single proposition as a string, e.g., "(P, (1,1))"
        return f"({prop[0]}, ({prop[1][0]},{prop[1][1]}))"  # Combines symbol and location

    def __str__(self) -> str:
        # Convert the clause to a readable string, e.g., "{(P, (1,1)):True v (Q, (2,2)):False}"
        if self.valid:
            return "{True}"  # Valid clauses are just "True"
        result = "{"
        for prop in self.props:
            result += f"{self._prop_str(prop)}:{self.props[prop]} v "  # Add each prop:value pair
        return result[:-3] + "}"  # Remove trailing " v " and close brace

    def __len__(self) -> int:
        # Return the number of propositions in the clause
        return len(self.props)  # Simple length of the props dictionary

    @staticmethod
    def resolve(c1: "MazeClause", c2: "MazeClause") -> set["MazeClause"]:
        # Perform resolution between two clauses to derive new clauses
        resolvents = set()  # Set to store resulting clauses (0 or 1 typically)
        if c1.valid or c2.valid:
            return resolvents  # Empty set if either clause is always true (no useful resolution)

        complementary = None  # Look for a proposition that’s positive in one and negated in the other
        for prop in c1.props:
            if prop in c2.props and c1.props[prop] != c2.props[prop]:
                complementary = prop  # Found a complementary pair (e.g., P and ~P)
                break

        if complementary is None:
            return resolvents  # No complementary props, no resolution possible

        # Combine non-complementary props from both clauses
        new_props = [(p, v) for p, v in c1.props.items() if p != complementary]
        new_props.extend([(p, v) for p, v in c2.props.items() if p != complementary])

        new_clause = MazeClause(new_props)  # Create new clause from combined props
        if not new_clause.is_valid():
            resolvents.add(new_clause)  # Add to resolvents if not always true

        return resolvents  # Return set with 0 or 1 clause