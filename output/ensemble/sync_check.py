def sync_check(path1, path2):
    with open(path1, 'r', encoding='utf-8') as f:
        data1 = f.readlines()
    with open(path2, 'r', encoding='utf-8') as f:
        data2 = f.readlines()
    i = 0
    for c1, c2 in zip(data1, data2):
        if c1 != c2:
            i+= 1
            print('not same')

    print(i)

sync_check('/home/tmp/pycharm_project_61/output/DP2/wordpiece/bart/500/0.1/n2n_result.txt',
           '/home/tmp/pycharm_project_61/output/DP2/wordpiece/bart/500/0.1/1_n2n_result.txt')