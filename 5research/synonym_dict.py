import json
from tqdm import tqdm
from nltk.corpus import wordnet


def get_wordnet_synonyms(word, offensive_words):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name()
            # Add only non-offensive synonyms
            if synonym not in offensive_words:
                synonyms.add(synonym)
    return list(synonyms)


def main():
    with open('../words.json') as f:
        offensive_words = json.load(f)

    offensive_words_set = set(offensive_words)
    synonym_mapping = {}

    for word in tqdm(offensive_words):
        synonyms = get_wordnet_synonyms(word, offensive_words_set)
        if synonyms:
            synonym_mapping[word] = synonyms[0].replace('_', ' ')

    with open('../inoffensive_synonym_mapping.json', 'w') as outfile:
        json.dump(synonym_mapping, outfile, indent=4)


if __name__ == '__main__':
    main()
