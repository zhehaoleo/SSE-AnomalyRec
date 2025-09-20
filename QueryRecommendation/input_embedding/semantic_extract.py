import os,sys
import re
sys.path.append(os.getcwd())
from nltk import word_tokenize
from utils.constants import *
from utils.helpers import *
import jieba

@memoize
def get_stopwords():
    """
        Read deactivated words and memorize them for direct access next time.
    """
    stop_words=[]
    stop_words_file_path=os.path.join(os.path.dirname(__file__),"../dataset/story_stopwords.txt")
    with open(stop_words_file_path,'r') as f:
        for line in f:
            stop_words.append(line.strip('\n'))
    return stop_words

def split_deeply(tokenized_list):
    """
        Insufficient space segmentation, further segmentation is more thorough.
    """
    new_tokenized_list=[]
    delimiters = ".", "/", "_","-"
    regexPattern = '|'.join(map(re.escape, delimiters))
    for word in tokenized_list:
        new_tokenized_list.extend(re.split(regexPattern,word.lower().strip()))
    return  new_tokenized_list

def tokenize_and_remove_stopwords(phrase, language='en'):
    """
        Segmenting semantic phrases in item and removing stop words.
    """
    stop_words=get_stopwords()
    tokenizer = word_tokenize if language == 'en' else jieba.cut
    tokenized_text=split_deeply(tokenizer(phrase))
    return [word for word in tokenized_text if word not in stop_words] 


def extract_semantic_info(fact):
    stop_words=get_stopwords()
    tokenized_semantic_token=list()
    semantic_pos_list=list()
    # -----------subspace, breakdown, measure, focus, meta-----------
    # subspace (field + value)
    subspace=fact["subspace"]
    for item in subspace:
        cut_tokenized_text_field = tokenize_and_remove_stopwords(item["field"], 'zh')
        cut_tokenized_text_value = tokenize_and_remove_stopwords(item["value"], 'zh')

        tokenized_semantic_token.extend(cut_tokenized_text_field)
        semantic_pos_list.extend([SEMANTIC_POS["subspace-field"]]*len(cut_tokenized_text_field))

        tokenized_semantic_token.extend(cut_tokenized_text_value)
        semantic_pos_list.extend([SEMANTIC_POS["subspace-value"]]*len(cut_tokenized_text_value))
  
    # time (value, type, role)
    time=fact["time"]
    for item in time:
        cut_tokenized_text_value = tokenize_and_remove_stopwords(item["value"])
        tokenized_semantic_token.extend(cut_tokenized_text_value)
        semantic_pos_list.extend([SEMANTIC_POS["time-value"]]*len(cut_tokenized_text_value))

        cut_tokenized_text_type = tokenize_and_remove_stopwords(item["type"])
        tokenized_semantic_token.extend(cut_tokenized_text_type)
        semantic_pos_list.extend([SEMANTIC_POS["time-type"]] * len(cut_tokenized_text_type))

        cut_tokenized_text_role = tokenize_and_remove_stopwords(item["role"])
        tokenized_semantic_token.extend(cut_tokenized_text_role)
        semantic_pos_list.extend([SEMANTIC_POS["time-role"]] * len(cut_tokenized_text_role))
    
    # measure (field, aggregation, type)
    measure=fact["measure"]
    for item in measure:
        cut_tokenized_text_field = tokenize_and_remove_stopwords(item["field"])
        tokenized_semantic_token.extend(cut_tokenized_text_field)
        semantic_pos_list.extend([SEMANTIC_POS["measure-field"]]*len(cut_tokenized_text_field))

        cut_tokenized_text_aggregation = tokenize_and_remove_stopwords(item["aggregation"])
        tokenized_semantic_token.extend(cut_tokenized_text_aggregation)
        semantic_pos_list.extend([SEMANTIC_POS["measure-aggregation"]] * len(cut_tokenized_text_aggregation))

        cut_tokenized_text_type = tokenize_and_remove_stopwords(item["type"])
        tokenized_semantic_token.extend(cut_tokenized_text_type)
        semantic_pos_list.extend([SEMANTIC_POS["measure-type"]] * len(cut_tokenized_text_type))

    # focus (field + value + meta)
    focus=fact["focus"]
    for item in focus:
        cut_tokenized_text_field = tokenize_and_remove_stopwords(item["field"])
        tokenized_semantic_token.extend(cut_tokenized_text_field)
        semantic_pos_list.extend([SEMANTIC_POS["focus-field"]]*len(cut_tokenized_text_field))

        # if len(item["value"])<30:
    #     tokenized_text_value=split_deeply(word_tokenize(item["value"]))
    #     cut_tokenized_text_value = [word for word in tokenized_text_value if word not in stop_words]
    #     tokenized_semantic_token.extend(cut_tokenized_text_value)
    #     semantic_pos_list.extend([SEMANTIC_POS["focus-value"]]*len(cut_tokenized_text_value))

        cut_tokenized_text_value = tokenize_and_remove_stopwords(item["value"])
        tokenized_semantic_token.extend(cut_tokenized_text_value)
        semantic_pos_list.extend([SEMANTIC_POS["focus-value"]] * len(cut_tokenized_text_value))

        cut_tokenized_text_meta = tokenize_and_remove_stopwords(item["level"])
        tokenized_semantic_token.extend(cut_tokenized_text_meta)
        semantic_pos_list.extend([SEMANTIC_POS["focus-level"]] * len(cut_tokenized_text_meta))

    return tokenized_semantic_token,semantic_pos_list

    