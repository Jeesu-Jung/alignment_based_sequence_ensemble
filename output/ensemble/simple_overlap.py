def simple_overlap(str1, str2, penalty):
    score = 0
    before_c1 = 'O'
    before_c2 = 'O'
    for i, node in enumerate(zip(str1, str2)):
        c1, c2 = node
        point = 1
        if penalty:
            if i != 0 and before_c1[0] == 'I' and c1[0] == 'I':
                if c1 != before_c1:
                    point -= 1
            if i != 0 and before_c2[0] == 'I' and c2[0] == 'I':
                if c2 != before_c2:
                    point -= 1
            before_c1 = c1
            before_c2 = c2

        if c1 == c2:
            score += point

    return score / len(str1)


def window_simple_overlap(str1, str2, window=3):
    score = 0
    for i, node in enumerate(zip(str1, str2)):
        c1, c2 = node
        point = 1
        if i >= window:
            w_c1 = '_'.join(str1[i - window / 2:i + window / 2])
            w_c2 = '_'.join(str2[i - window / 2:i + window / 2])
        if w_c1 == w_c2:
            score += point

    return score / len(str1)
