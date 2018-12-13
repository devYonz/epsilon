import numpy as np


def get_text_body(issue: dict)-> str:
    """ Take a JIRA ticket issue and convert it to a single line text

    :param dict issue: JIRA object with many keys
    :return str: one line string representation concatenating text form the issues
    """
    text_body = issue.get('summary')
    if issue.get('description'):
        text_body += ' ' + issue['description']

    # Capture comment bodies
    for comment in issue.get('comments'):
        if isinstance(comment, dict):
            text_body += ' ' + comment.get('body')
        elif isinstance(comment, list):
            for comment_line in comment:
                text_body += ' ' + comment_line
        else:
            text_body += ' ' + comment
    text_body = text_body.replace('\n', ' ')
    text_body = text_body.replace('\r', ' ')
    text_body = text_body.replace('&nbsp', ' ')

    return text_body


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    return message.lower().split(" ")
    # *** END CODE HERE ***


def create_dictionary(messages: list, vocabulary: dict = {}, counts: dict = {}) -> dict:
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
        { "word" => index }
    """

    # *** START CODE HERE ***
    index = 0
    for message in messages:
        words = get_words(message)
        # A word should be counted once per message
        for word in set(words):
            if counts.get(word) is None:
                counts[word] = 1
            else:
                counts[word] += 1
            if counts[word] == 5:
                vocabulary[word] = index
                index += 1
    # Size is 1722
    return vocabulary
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        ndarray(m - messages, n - words)
        Sample 1: [ 10 2233 2334 44]  int(word1) int(word2)
        Sample 2: [
        :param stopwords: List of words to remove
    """
    # *** START CODE HERE ***
    n = len(word_dictionary)
    m = len(messages)
    X = np.ndarray((m, n))

    for i, message in enumerate(messages):
        x_i = np.zeros(n)

        words = get_words(message)
        for word in words:
            index = word_dictionary.get(word)
            if index is not None:
                x_i[index] = x_i[index] + 1
        X[i] = x_i
    return X
    # *** END CODE HERE ***

