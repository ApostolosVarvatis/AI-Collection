import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    pb_distribution = {}
    
    # Page has no links
    if len(corpus[page]) == 0:
        no_link = 1/len(corpus)
        for p in corpus:
            pb_distribution[p] = no_link
        return pb_distribution

    # With probability d
    d_prob = damping_factor * (1/len(corpus[page]))
    # With probability 1 - d
    one_d_prob = (1 - damping_factor) * (1/len(corpus))

    # Add prob to pb dict
    for p in corpus[page]:
        pb_distribution[p] = d_prob + one_d_prob
    pb_distribution[page] = one_d_prob

    return pb_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialize data structures
    pages_ranked = {}
    pages = list(corpus.keys())

    # First random sample probabilities
    r_page = random.choice(pages)
    curr_prob = transition_model(corpus, r_page, damping_factor)

    # Initialize probability values
    for page in pages:
        pages_ranked[page] = 0.0
    pages_ranked[r_page] += 1/n

    for _ in range(n-1):
        next_sample_page = random.choices(list(curr_prob.keys()), weights=list(curr_prob.values()))[0]
        pages_ranked[next_sample_page] += 1/n
        curr_prob = transition_model(corpus, next_sample_page, damping_factor)
        
    return pages_ranked


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialize data structures
    pages_ranked = {}
    pages = list(corpus.keys())

    # Initialize probability values
    for page in pages:
        pages_ranked[page] = 1/len(corpus)

    while True:
        # Initiliaze previous values dict
        prev_pages_ranked = pages_ranked.copy()

        # Update the values based on the fromula
        for curr_page in pages:
            Ssum = 0
            for page in pages:
                # Check if any page has no outgoing links, if so add one link for every page in pages
                if len(corpus[page]) == 0:
                    Ssum += pages_ranked[page] * 1/len(corpus)

                elif curr_page in corpus[page]:
                    Ssum += pages_ranked[page] / len(corpus[page])

            formula = ((1 - damping_factor)/len(corpus) + (damping_factor * Ssum))
            pages_ranked[curr_page] = formula

        # Normalizing values
        pg_sum = sum(pages_ranked.values())
        for page, rank in pages_ranked.items():
            pages_ranked[page] = rank / pg_sum

        breakCondition = True
        for page in pages:
            if abs(pages_ranked[page] - prev_pages_ranked[page]) > 0.001:
                breakCondition = False
                break

        if breakCondition:
            return pages_ranked
        
        
if __name__ == "__main__":
    main()
