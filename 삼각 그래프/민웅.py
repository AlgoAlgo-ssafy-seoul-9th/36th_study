# 4883_삼각 그래프_triangle graph
import sys
input = sys.stdin.readline

tc = 0
while True:
    tc += 1
    N = int(input())
    if not N:
        break

    ans = float('inf')
    tri_graph = [list(map(int, input().split())) for _ in range(N)]

    dp = [[float('inf'), float('inf'), float('inf')] for _ in range(N)]
    dp[0][1] = tri_graph[0][1]
    dp[0][2] = tri_graph[0][1] + tri_graph[0][2]

    for i in range(1, N):
        for j in range(3):
            for k in range(max(0, j-1), min(2, j+1)+1):
                dp[i][j] = min(dp[i-1][k] + tri_graph[i][j], dp[i][j])
            if j != 0:
                dp[i][j] = min(dp[i][j-1] + tri_graph[i][j], dp[i][j])

    ans = dp[N-1][1]

    print(f'{tc}. {ans}')