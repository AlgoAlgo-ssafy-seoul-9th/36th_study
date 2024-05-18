N = int(input())
arr = [list(map(int, input().split())) for _ in range(N)]

DP = [[0]*N for _ in range(N)]

DP[0][0] = 1        # 0,0을 거쳐 다른 곳으로 가는 방법 1가지
for i in range(1, N):
    for k in range(1,10):
        if i-k>=0 and arr[i-k][0]==k:   # k만큼 윗칸에서 i로 올 수 있으면
            DP[i][0] += DP[i-k][0]
        if i-k>=0 and arr[0][i-k]==k:   # k만큼 왼쪽칸에서 i로 올 수 있으면
            DP[0][i] += DP[0][i-k]
for i in range(1, N):
    for j in range(1, N):
        for k in range(1,10): # 최대 9칸 이동
            if i-k>=0 and arr[i-k][j]==k:   # k 칸 윗쪽에서 올 수 있으면
                DP[i][j] += DP[i-k][j]
            if j-k>=0 and arr[i][j-k]==k:
                DP[i][j] += DP[i][j-k]
print(DP[N-1][N-1])
