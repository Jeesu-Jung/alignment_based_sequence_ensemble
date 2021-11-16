import random
import os, sys, codecs

from measure.conlleval import eval_performance


def get_performance(data):
    # result must be list of tuple(tokens, reference_classes, predicted_classes) of a sentence
    current_folder = os.path.dirname(__file__)
    rand_num_str = str(random.random() * 100)
    _temp_fn = os.path.join(current_folder, rand_num_str)

    with codecs.open(_temp_fn, 'w', encoding='utf-8') as f:
        for a_sent in data:
            for c, t, r, p in a_sent:
                print("{}\t{}\t{}\t{}".format(c, t, r, p), file=f)

            print(file=f)  # empty line for sentence boundary

    # do check performance
    perf = eval_performance(_temp_fn)

    # delete temp file
    os.remove(_temp_fn)
    return perf