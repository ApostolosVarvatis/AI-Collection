import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        # Loop through variables
        for var in self.domains:
            # Make a copy to resolve a "RuntimeError"
            domain_copy = self.domains[var].copy()
            # Loop through variable values
            for word in domain_copy:
                # Check unary constraint and act
                if len(word) != var.length:
                    self.domains[var].remove(word)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        x_domain_copy = self.domains[x].copy()
        overlap = self.crossword.overlaps[x, y]

        if overlap is not None:
            i, j = overlap
            
            # Loop through all the words in X domain
            for Xword in x_domain_copy:
                valid = False
                # Loop through all the words in Y domain
                for Yword in self.domains[y]:
                    # Check for binary constraints and if so remove word from X domain 
                    if Xword[i] == Yword[j]:
                        valid = True
                        break
                if not valid:
                    self.domains[x].remove(Xword)
                    revised = True
                    
        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        # Check if arcs empty and if so initialize all arcs
        if arcs is None:
            queue = []
            for v1 in self.domains:
                for v2 in self.domains:
                    if v1 != v2:
                        queue.append((v1, v2))
        else:
            queue = arcs

        # Loop through all arcs revising arc consistency
        while queue:
            x, y = queue.pop(0)
            # If arc was revised
            if self.revise(x, y):
                # Check if domain non-empty
                if not self.domains[x]:
                    return False
                # Add new values to check arc consistency
                for z in self.crossword.neighbors(x) - {y}:
                    queue.append((z, x))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for var in self.domains:
            if var not in assignment:
                return False
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        distinct_values = []

        for var in assignment:
            val = assignment[var]
            # Check for length
            if len(val) != var.length:
                return False
            
            # Check neighbor conflicts
            for y in self.crossword.neighbors(var):
                if y in assignment:
                    if self.crossword.overlaps[var, y] is None:
                        continue
                    else:
                        i, j = self.crossword.overlaps[var, y]
                        if val[i] != assignment[y][j]:
                            return False
                    
            # Check uniqueness
            if val in distinct_values:
                return False
            distinct_values.append(val)
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        Dvalues = []
        neighbors = self.crossword.neighbors(var)

        # Loop through the words in variable's domain
        for x in self.domains[var]:
            if x in assignment:
                continue
            # Initialize "ruled out" counter
            c = 0
            # Loop through var neighbors
            for neighbor in neighbors:
                if x in self.domains[neighbor]:
                    c += 1
            Dvalues.append((x, c))

        s = sorted(Dvalues, key=lambda x: x[1])

        return [x[0] for x in s]

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        mrv = []
        # Loop through all variables in the domain
        for var in self.domains:
            if var in assignment:
                continue
            # Get mrv and highest degree
            Dsize = len(self.domains[var])
            Nsize = len(self.crossword.neighbors(var))
            mrv.append((var, Dsize, Nsize))

        s = sorted(mrv, key=lambda x: (x[1], x[2]))

        return s[0][0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment
        var = self.select_unassigned_variable(assignment)
        for val in self.order_domain_values(var, assignment):
            assignment[var] = val
            if self.consistent(assignment):
                result = self.backtrack(assignment)
                if result:
                    return result
            assignment.pop(var)

        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
