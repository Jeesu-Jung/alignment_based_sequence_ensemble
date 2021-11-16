def needelman_wunsch(str1, str2, sync=3, non_sync=-3, gap=-2, penalty=False):
    similarity = []

    for i in range(len(str1) + 1):
        sim = []
        if i == 0:
            sim.append(0)
            for j in range(1, len(str2) + 1):
                sim.append(-j)
        else:
            sim.append(-i)
            for j in range(1, len(str2) + 1):
                sim.append(0)
        similarity.append(sim)

    before_c1 = 'O'
    before_c2 = 'O'
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if penalty:
                if i != 0 and before_c1[0] == 'I' and str1[i - 1][0] == 'I':
                    if str1[i - 1] != before_c1:
                        non_sync -= 1
                if i != 0 and before_c2[0] == 'I' and str2[i - 1][0] == 'I':
                    if str2[i - 1] != before_c2:
                        non_sync -= 1
                before_c1 = str1[i - 1]
                before_c2 = str2[i - 1]
            if str1[i - 1] == str2[j - 1]:
                similarity[i][j] = max(similarity[i - 1][j - 1] + sync,
                                       similarity[i - 1][j] + gap,
                                       similarity[i][j - 1] + gap)
            else:
                similarity[i][j] = max(similarity[i - 1][j - 1] + non_sync,
                                       similarity[i - 1][j] + gap,
                                       similarity[i][j - 1] + gap)

    return similarity[-1][-1]
