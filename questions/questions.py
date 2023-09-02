import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    file_map = {}

    # Loop through the directory files
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)

            # Read file data into memory
            with open(filepath, "r") as f:
                # Write contents to dictionairy
                file_map[filename] = f.read()

    return file_map


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = []

    for word in nltk.word_tokenize(document.lower()):
        if word in nltk.corpus.stopwords.words("english"):
            continue
        
        for char in word:
            if char not in string.punctuation:
                words.append(word)
                break
            
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idf_map = {}
    num_docs = len(documents)
    
    # Get NumDocumentsContaining(word) map
    for doc_words in documents.values():
        unique_words = set(doc_words)
        for word in unique_words:
            if word in idf_map:
                idf_map[word] += 1
            else:
                idf_map[word] = 1

    # Calculate idf values and add them to the map
    for word in idf_map.keys():
        idf_map[word] = math.log(num_docs/idf_map[word])

    return idf_map


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    file_list = {}

    # Loop though every file in files
    for filename, file_words in files.items():
        sum = 0
        tf = {}
        # Loop though current file's words to get tf values
        for word in file_words:
            if word in tf:
                tf[word] += 1
            else:
                tf[word] = 1
        # Loop though query's words to get td_idf sum for that file
        for word in query:
            if word not in tf:
                continue
            sum += tf[word] * idfs[word]

        # Add filename to the dict with the calculated sum value
        file_list[filename] = sum

    # Sort and generate the ouput list
    output_list = sorted(file_list, key=lambda x: file_list[x], reverse=True)

    return output_list[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_list = {}

    # Loop through the all sentences
    for sentence_name, sentence_words in sentences.items():
        sum = 0
        qtd = 0
        # Loop though words of current sentence
        for word in query:
            if word in sentence_words:
                sum += idfs[word]
                qtd += 1
        
        # Add sentence_name to the dict with the calculated sum and qtd values as a tuple
        sentence_list[sentence_name] = (sum, qtd / len(sentence_words))

    # Sort and generate the ouput list
    output_list = sorted(sentence_list, key=lambda x: (sentence_list[x][0], sentence_list[x][1]), reverse=True)
    
    return output_list[:n]


if __name__ == "__main__":
    main()
