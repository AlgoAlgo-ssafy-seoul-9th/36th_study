# 4883 삼각 그래프
import sys
input = sys.stdin.readline


def main():
    def getMinCost(N:int, tg:list) -> int:
        dp = [[0] * 3 for _ in range(N)]
        for i in range(3):
            dp[0][i] = tg[0][i]
        dp[0][2] += dp[0][1]
        dp[1][0] = dp[0][1] + tg[1][0]
        dp[1][1] = min(dp[0][1], dp[0][2], dp[1][0]) + tg[1][1]
        dp[1][2] = min(dp[0][1], dp[0][2], dp[1][1]) + tg[1][2]
        
        for i in range(2, N):
            dp[i][0] = min(dp[i-1][0], dp[i-1][1]) + tg[i][0]
            dp[i][1] = min(dp[i-1]) 
            dp[i][2] = min(dp[i-1][1], dp[i-1][2])
            dp[i][1] = min(dp[i][1], dp[i][0]) +  tg[i][1]
            dp[i][2] = min(dp[i][2], dp[i][1]) + tg[i][2]
        # [print(dp[i]) for i in range(N)]
        return dp[-1][1]
    
    k = 1
    while True:
        N = int(input())
        if N == 0:
            break
        tg = [tuple(map(int, input().split())) for _ in range(N)]
        ans = getMinCost(N, tg)
        print(f"{k}. {ans}")
        k += 1
    return


if __name__ == "__main__":
    main()