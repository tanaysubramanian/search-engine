import tqdm
import math
import re
import sys
import xml.etree.ElementTree as et

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm
from collections import Counter

import file_io


class Indexer:
    """
    The indexer class, which takes in a wiki and processes the documents in it
    into files that are used by the querier
    """

    def __init__(self, wiki: str, title: str, doc: str, word: str):
        """
        The constructor for the indexer.
        DO NOT MODIFY THIS CONSTRUCTOR.

        Note that the output files may be overwritten if they already exist.
        
        Parameters:
        wiki        the filename of the input wiki
        title       the output filename of the titles file
        doc         the output filename of the docs file
        word        the output filename of the words file
        """

        # defining epsilon for PageRank calculations
        self.EPSILON = 0.15
        # distance threshold for PageRank calculation
        self.DISTANCE_THRESHOLD = 0.001
        # set of stop words
        self.STOP_WORDS = set(stopwords.words("english"))
        # porter stemmer
        self.nltk_ps = PorterStemmer()

        """
        The tokenization regex has three parts, separated by pipes (|), which
        mean “or.” So we are actually matching three possible alternatives:

        1) \[\[[^\[]+?\]\]
        Meaning: Match two left brackets (\[\[) and two right brackets (\]\]),
        making sure there’s something in the middle, but also making sure there
        is not a left bracket in the middle, which would mean that somehow
        another link was starting inside this link ([^\[]+?).
        Returns: Links (e.g., “[[Some Wiki Page]]” or “[[Universities|Brown]]”)

        2) [a-zA-Z0-9]+'[a-zA-Z0-9]+
        Meaning: Match at least one alphanumeric character ([a-zA-Z0-9]+), then
        an apostrophe ('), and then at least one alphanumeric character
        ([a-zA-Z0-9]).
        Returns: Words with apostrophes (e.g., “don’t”)

        3) [a-zA-Z0-9]+
        Meaning: Match at least one alphanumeric character in a row.
        Returns words (e.g., “dog”)
        """
        self.tokenization_regex = r"\[\[[^\[]+?\]\]|[a-zA-Z0-9]+'[a-zA-Z0-9]+|[a-zA-Z0-9]+"

        # page id to word to num appearances
        self.words_to_doc_frequency = {}
        # page id to title
        self.ids_to_titles = {}
        # title to page id
        self.titles_to_ids = {}
        # page id to highest word counts
        self.ids_to_max_counts = {}
        # id to all the ids that page links to
        self.ids_to_links = {}

        self.wiki = wiki
        self.title = title
        self.doc = doc
        self.word = word

    def run(self):
        """
        Runs the indexer by parsing the document, computing term relevance and
        page rank, and writing the results to the titles/docs/words output
        files

        DO NOT MODIFY THIS METHOD.
        """
        try:
            self.parse()
            words_to_doc_relevance = self.compute_term_relevance()
            page_rank = self.compute_page_rank()
            
            file_io.write_title_file(self.title, self.ids_to_titles)
            file_io.write_document_file(
                self.doc, page_rank)
            file_io.write_words_file(self.word, words_to_doc_relevance)
        except FileNotFoundError:
            print("One (or more) of the files were not found")
        except IOError:
            print("Error: IO Exception")

    def stem_and_stop(self, word: str):
        """
        Checks if word is a stop word, converts it to lowercase, and stems it

        DO NOT MODIFY THIS METHOD.

        Parameters:
            word        the word to check
        Returns:
            "" if the word is a stop word, the converted word, otherwise
        """
        if word.lower() in self.STOP_WORDS:
            return ""

        return self.nltk_ps.stem(word.lower())

    def word_is_link(self, word: str) -> bool:
        """
        Checks if the word is a link (surrounded by '[[' and ']]')

        DO NOT MODIFY THIS METHOD.

        Parameters:
            word        the word to check
        Returns:
            true if the word is a link, false otherwise
        """
        link_regex = r"\[\[[^\[]+?\]\]"
        return bool(re.match(link_regex, word))

    def split_link(self, link: str) -> tuple[list[str], str]:
        """
        Splits a link (assumed to be surrounded by '[[' and ']]') into the text
        and the destination of the link

        DO NOT MODIFY THIS METHOD.

        Example usage:
        link_text, link_dest = split_link(link_str)

        Parameters:
            link        the link to split
        Returns:
            a tuple of the format (link text, link destination)
        """
        is_word_regex = r"[a-zA-Z0-9]+'[a-zA-Z0-9]+|[a-zA-Z0-9]+"

        # assume that the link is surrounded by '[[' and ']]'
        link_stripped_brackets = link[2:-2]

        title_raw = link_stripped_brackets
        text_raw = link_stripped_brackets

        # text and title differ
        if '|' in link_stripped_brackets:
            link_split = link_stripped_brackets.split("|")
            title_raw = link_split[0]
            text_raw = link_split[1]

        return (re.findall(is_word_regex, text_raw), title_raw.strip())

    def process_document(self, title: str, id: int, body: str) -> list[str]:
        """
        Takes in a document title, id, and body, and returns a list of the
        tokens in the document title and body.

        A "token" is a word, not including stopwords, that has been stemmed. For
        links, only the link text (not destination) are included in the returned
        list.

        Parameters:
            title       the title of the document
            id          the id of the document
            body        the text of the document
        Returns:
            a list of the tokens in the title and the body

        This method assumes that the ids_to_titles and titles_to_ids dictionaries have been populated first,
        followed by the words_to_doc_frequency and ids_to_max_counts dictionaries. The ids_to_links dictionary
        is populated during the same pass but in the process_document method instead of parse. 
        """
        token_list = []
        all_tokens = re.findall(self.tokenization_regex, title) + re.findall(self.tokenization_regex, body)
        final_set = set()

        for token in all_tokens:
            if self.stem_and_stop(token) != "":
                if self.word_is_link(token):
                    link = self.split_link(token)
                    final_set.add(link[1])

                    for w in link[0]:
                        token_list.append(self.stem_and_stop(w))
                        
                else:
                    token_list.append(self.stem_and_stop(token))
        if len(final_set) != 0:
            self.ids_to_links[id] = final_set
    
        return [n for n in token_list if len(n)!=0]
    
    def update_dicts(self, title: str, id: int, tokens: list):
        """Updates the dictionary data structures"""
        self.ids_to_titles[id] = title
        self.titles_to_ids[title] = id
        token_counts = Counter(tokens)

        for token in tokens:
            if token not in self.words_to_doc_frequency:
                self.words_to_doc_frequency[token] = {}
            if id not in self.words_to_doc_frequency[token]:
                self.words_to_doc_frequency[token][id] = 1
            else:
                self.words_to_doc_frequency[token][id] += 1
        self.ids_to_max_counts[id] = max(token_counts.values())

    def parse(self):
        """
        Reads in an xml file, parses titles and ids, tokenizes text, removes
        stop words, does stemming, and processes links.

        Updates ids_to_titles, titles_to_ids, words_to_doc_frequency,
        ids_to_max_counts, and ids_to_links
        """
        self.ids_to_titles = {}
        self.titles_to_ids = {}
        self.words_to_doc_frequency = {}
        self.ids_to_max_counts = {}
        self.ids_to_links = {}        

        # load XML + root
        wiki_tree = et.parse(self.wiki)
        wiki_xml_root = wiki_tree.getroot()

        for page in tqdm(wiki_xml_root):
            page_id = int(page.find("id").text)
            page_title = str(page.find("title").text.strip())
            page_text = str(page.find("text").text.strip())
            token_list = self.process_document(page_title, page_id, page_text)
            self.update_dicts(page_title, page_id, token_list)     

        temp = self.ids_to_links.copy()
        self.ids_to_links.clear()
        for id in temp:
            self.ids_to_links[id] = set()
            for page in temp[id]:
                if page in self.ids_to_titles:
                    self.ids_to_links[id].add(page)
                else:
                    self.ids_to_links[id] = {self.titles_to_ids.get(key, key) for key in temp[id]}
                
    def compute_tf(self) -> dict[str, dict[int, float]]:
        """
        Computes tf metric based on words_to_doc frequency

        Assumes parse has already been called to populate the relevant data
        structures.

        Returns:
            a dictionary mapping every word to its term frequency
        """
        tf = {}
        for word in self.words_to_doc_frequency:
            tf[word] = {}
            for doc in self.words_to_doc_frequency[word]:
                tf[word][doc] = self.words_to_doc_frequency[word][doc] / self.ids_to_max_counts[doc]  
        return tf

    def compute_idf(self) -> dict[str, float]:
        """
        Computes idf metric based on words_to_doc_frequency

        Assumes parse has already been called to populate the relevant data
        structures.

        Returns:
            a dictionary mapping every word to its inverse term frequency
        """
        idf = {}
        for word in self.words_to_doc_frequency:
            idf[word] = math.log(len(self.ids_to_titles) / len(self.words_to_doc_frequency[word]))
        return idf

    def compute_term_relevance(self) -> dict[str, dict[int, float]]:
        """
        Computes term relevance based on tf and idf

        Assumes parse has already been called to populate the relevant data
        structures.

        Returns:
            a dictionary mapping every every term to a dictionary mapping a page
            id to the relevance metric for that term and page
        """
        compute_t = self.compute_tf()
        compute_i = self.compute_idf()
        rel = {}
        for word in self.words_to_doc_frequency:
            rel[word] = {}
            for doc in self.words_to_doc_frequency[word]:
                    rel[word][doc] = compute_t[word][doc] * compute_i[word]
        return rel

    def distance(self, dict_a: dict[int, float], dict_b: dict[int, float]) -> float:
        """
        Computes the Euclidean distance between two PageRank dictionaries
        Only to be called by compute_page_rank

        DO NOT MODIFY THIS METHOD.

        Parameters:
            dict_a          the first dictionary
            dict_b          the second dictionary

        Returns:
            the Euclidean distance between dict_a and dict_b
        """
        # Assures that the two dictionaries have the same number of keys
        assert len(dict_a) == len(dict_b)

        curr_dist = 0
        for page in dict_a:
            # Assures that any key in dict_a is also a key in dict_b
            assert page in dict_b
            curr_dist += (dict_a[page] - dict_b[page]) ** 2

        return math.sqrt(curr_dist)

    def compute_weights(self) -> dict[int, dict[int, float]]:
        """
        Computes the weights matrix for PageRank

        Assumes parse has already been called to populate the relevant data
        structures.

        DO NOT MODIFY THIS METHOD.

        Returns
            a dictionary from page ids to a dictionaries from page ids to
            weights such that:
            weights[id_1][id_2] = weight of edge from page with id_1 to page
                                  with id_2
        """
        
        weights = {}
        num_pages = len(self.ids_to_titles)

        for j in self.ids_to_titles:
            weights[j] = {}
            num_links = num_pages - \
                1 if j not in self.ids_to_links else len(
                    self.ids_to_links[j])

            if j in self.ids_to_links and j in self.ids_to_links[j]:
                num_links = num_links - 1 if num_links > 1 else num_pages - 1

            for k in self.ids_to_titles:
                if j == k:
                    # page links to itself
                    weights[j][k] = self.EPSILON / num_pages
                elif j in self.ids_to_links and k in self.ids_to_links[j]:
                    # this page links to that page
                    weights[j][k] = (self.EPSILON / num_pages) + \
                        ((1 - self.EPSILON) / num_links)
                elif num_links == num_pages - 1:
                    weights[j][k] = (self.EPSILON / num_pages) + \
                        ((1 - self.EPSILON) / num_links)
                else:
                    weights[j][k] = self.EPSILON / num_pages

        return weights

    def compute_page_rank(self) -> dict[int, float]:
        """
        Computes PageRank for every page and fills the page_rank map

        Assumes parse has already been called to populate the relevant data
        structures.

        Returns:
            A dict mapping a page id to its authority, as computed by the
            PageRank algorithm
        """

        '''
        pseudocode for PageRank:
        pageRank():
            initialize every rank in r to be 0
            initialize every rank in r' to be 1/n
            while distance(r, r') > delta:
                r <- r'
                for j in pages:
                    r'(j) = sum of weight(k, j) * r(k) for all pages k

        Use self.DISTANCE_THRESHOLD for delta
        '''
        weights = self.compute_weights()
        r = {}
        r1 = {}
        for page in self.ids_to_titles:
            r[page] = 0.0
            r1[page] = 1.0 / len(self.ids_to_titles)
        
        while self.distance(r, r1) > self.DISTANCE_THRESHOLD:
            r = r1.copy()
            for j in self.ids_to_titles:
                val = 0.0
                for k in self.ids_to_titles:
                    val += weights[k][j] * r[k]
                r1[j] = val
        return r1

if __name__ == "__main__":
    if len(sys.argv) == 5:
        the_indexer = Indexer(*sys.argv[1:])
        the_indexer.run()
    else:
        print("Incorrect arguments: use <wiki> <titles> <documents> <words>")