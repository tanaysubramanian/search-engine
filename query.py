"""
Reads in the files produced by the indexer and runs a search repl
"""
import sys

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import file_io


class Querier:
    # PageRank flag
    def __init__(self, page_rank: bool, title: str, doc: str, word: str):
        self.page_rank = page_rank
        # page id to word to num appearances
        self.words_to_doc_relevance = {}
        # page id to title
        self.ids_to_titles = {}
        # page id to highest word count
        self.ids_to_max_counts = {}
        # id to page rank value
        self.ids_to_pageranks = {}

        self.title_file = title
        self.doc_file = doc
        self.word_file = word

    def stem_array(self, word_array: list):
        """
        converts word_array to stemmed
        """
        s = PorterStemmer()
        return [s.stem(x) for x in word_array]

    def print_results(self, results: list):
        """
        Prints (up to) the top 10 results
        """
        num_results = min(len(results), 10)
        for i in range(num_results):
            print("\t" + str(i + 1) + " " + self.ids_to_titles[results[i]])

    def ranking_function(self, doc):
        """
        Returns the score of a document, based on the fields:
        
        ids_to_relevance_scores (a dict mapping each doc to its relevance to the
        current query)
        
        ids_to_pageranks (a dict mapping each doc to its authority, as
        determined by PageRank)

        page_rank (a boolean that says whether pagerank is being used or not)
        """
        if self.page_rank:
            return self.ids_to_pageranks[doc] * self.ids_to_relevance_scores[doc]
        else:
            return self.ids_to_relevance_scores[doc]


    def handle_query(self, user_query: str):
        """
        Tokenizes query, checks each word for its relevance, ranks results by relevance
        """
        stop_words = set(stopwords.words('english'))  # stop words
        # turn query into list of stemmed words (excluding stop words)
        ps = PorterStemmer()
        words = [ps.stem(x) for x in user_query.lower().split(
            " ") if x not in stop_words]

        # map each page where a word is found to its cumulative relevance score
        self.ids_to_relevance_scores = {}

        # each word in the query is considered separately
        for word in words:
            # Only calculate word's contribution to score if it appears in corpus
            if word in self.words_to_doc_relevance:
                for page_id, relevance in self.words_to_doc_relevance[word].items():
                    if page_id not in self.ids_to_relevance_scores:
                        self.ids_to_relevance_scores[page_id] = 0.0
                    # each relevant page adds to the score
                    self.ids_to_relevance_scores[page_id] += relevance

        if len(self.ids_to_relevance_scores) == 0:
            print("No results")
            return

        # list of document ids where some word(s) in the query appeared
        result_ids = list(self.ids_to_relevance_scores.keys())

        # sort the ids based on the relevance in the ids_to_relevance_scores
        # dictionary
        result_ids.sort(reverse=True, key=self.ranking_function)

        print("---------" + "\n")
        self.print_results(result_ids)
        return result_ids

    def read_files(self, title_file, doc_file, word_file):
        """
        Read each file into its relevant dictionary
        """
        file_io.read_title_file(title_file, self.ids_to_titles)
        file_io.read_docs_file(
            doc_file, self.ids_to_pageranks)
        file_io.read_words_file(word_file, self.words_to_doc_relevance)

    def search_repl(self):
        """
        Run the user loop
        """
        user_query = ""
        while True:
            user_query = input("search> ")

            # if ":quit" is reached, exit loop
            if user_query == ":quit":
                return
            # handle the query
            self.handle_query(user_query)


if __name__ == "__main__":
    try:
        if len(sys.argv) == 5 and sys.argv[1] == "--pagerank":
            page_rank = True
            title_index = 2
            doc_index = 3
            word_index = 4
        elif len(sys.argv) == 4:
            page_rank = False
            title_index = 1
            doc_index = 2
            word_index = 3
        else:
            print(
                "Incorrect arguments. Please use [--pagerank] <titleIndex> <documentIndex> <wordIndex>")
            sys.exit(1)
        # query
        title_file = sys.argv[title_index]
        doc_file = sys.argv[doc_index]
        word_file = sys.argv[word_index]

        myQuerier = Querier(page_rank, title_file, doc_file, word_file)
        myQuerier.read_files(title_file, doc_file, word_file)
        myQuerier.search_repl()
    except FileNotFoundError as e:
        print("One (or more) of the files were not found")
    except IOError as e:
        print("Error: IO Exception")
