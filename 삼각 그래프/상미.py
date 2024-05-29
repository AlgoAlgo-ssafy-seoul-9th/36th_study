import sys
input = sys.stdin.readline

t = 0
while True:
    t += 1
    N = int(input())
    if N == 0:
        break
    arr = [list(map(int, input().split())) for _ in range(N)]
    dp = [[0] * 3 for _ in range(N)]
    
    dp[0][1] = arr[0][1]
    dp[0][0] = arr[0][1]    # 첫째줄 가운데에서 시작하는 것밖에 없으므로
    dp[0][2] = dp[0][1] + arr[0][2]
    
    for i in range(1, N):
        dp[i][0] = min(dp[i-1][0], dp[i-1][1]) + arr[i][0]
        dp[i][1] = min(dp[i-1][0], dp[i-1][1], dp[i-1][2], dp[i][0]) + arr[i][1]
        dp[i][2] = min(dp[i-1][1], dp[i-1][2], dp[i][1]) + arr[i][2]

    print(f'{t}. {dp[N-1][1]}')
