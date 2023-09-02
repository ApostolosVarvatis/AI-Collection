import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # Initialize jointProb list
    jointProb = []

    for person in people:
        # Run loop for child
        if people[person]["mother"] is not None and people[person]["father"] is not None:

            # Get gene counts and parent probability with helper funcitons
            cgc = gene_count(people[person]["name"], one_gene, two_genes)
            mProb = get_prob(gene_count(people[person]["mother"], one_gene, two_genes))
            fProb = get_prob(gene_count(people[person]["father"], one_gene, two_genes))
        
            if people[person]["name"] in one_gene:
                # Case 1: One from m and none from f             
                # Case 2: One from f and none from m
                # Add the cases
                formula = fProb * (1 - mProb) + mProb * (1 - fProb)
            elif people[person]["name"] in two_genes:
                # One from m and one from f
                formula = mProb * fProb
            else:
                # None from m and none from f
                formula = (1 - mProb) * (1 - fProb)

            # Considering "the trait"
            if people[person]["name"] in have_trait:
                jointProb.append(formula * PROBS["trait"][cgc][True])
            else:
                jointProb.append(formula * PROBS["trait"][cgc][False])

        # Run loop for parents
        else:
            # Get person's gene count
            pgc = gene_count(people[person]["name"],one_gene, two_genes)

            # Considering "the trait"
            if people[person]["name"] in have_trait:
                jointProb.append(PROBS["gene"][pgc] * PROBS["trait"][pgc][True])
            else:
                jointProb.append(PROBS["gene"][pgc] * PROBS["trait"][pgc][False])
    
    # Multiply every value in joint probabiliyties dictinairy
    jp = 1
    for val in jointProb:
        jp *= val
    return jp


def gene_count(person, one_gene, two_genes):
    """
    Helper function for getting gene count for a person.
    """
    if person in one_gene:
        return 1
    elif person in two_genes:
        return 2
    else:
        return 0


def get_prob(gene_count):
    """
    Helper function for getting the probability for each gene case.
    """
    mp = PROBS["mutation"]

    if gene_count == 0:
        return mp
    elif gene_count == 1:
        return 0.5
    else:
        return 1 - mp


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        # Get gene count of person
        gc = gene_count(person, one_gene, two_genes)

        # Update the values
        probabilities[person]["gene"][gc] += p

        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p
        

def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for p in probabilities:
        # Get sum of genes and traits respectively 
        geneSum = sum(probabilities[p]["gene"].values())
        traitSum = sum(probabilities[p]["trait"].values())

        # Update the values
        for gene, prob in probabilities[p]["gene"].items():
            probabilities[p]['gene'][gene] = prob / geneSum
        for trait, prob in probabilities[p]["trait"].items():
            probabilities[p]['trait'][trait] = prob / traitSum


if __name__ == "__main__":
    main()
