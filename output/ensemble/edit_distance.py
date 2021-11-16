def edit_distance(s: str, t: str):
    m = len(s) + 1
    n = len(t) + 1
    D = [[0] * m for _ in range(n)]
    D[0][0] = 0

    for i in range(1, m):
        D[0][i] = D[0][i - 1] + 1

    for j in range(1, n):
        D[j][0] = D[j - 1][0] + 1

    for i in range(1, n):
        for j in range(1, m):
            cost = 0

            if s[j - 1] != t[i - 1]:
                cost = 1

            D[i][j] = min(D[i][j - 1] + 1, D[i - 1][j] + 1, D[i - 1][j - 1] + cost)

    return len(s) - D[n - 1][m - 1]