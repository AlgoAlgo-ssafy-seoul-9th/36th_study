# 36th_study

알고리즘 스터디 36주차

<br/>

# 이번주 스터디 문제

<details markdown="1" open>
<summary>접기/펼치기</summary>

<br/>

## [보드 점프](https://www.acmicpc.net/problem/3372)

### [민웅](./보드%20점프/민웅.py)

```py
# 3372_보드 점프_board jump
import sys
input = sys.stdin.readline

N = int(input())

board = [list(map(int, input().split())) for _ in range(N)]
dp = [[0]*N for _ in range(N)]

dp[0][0] = 1

for i in range(N):
    for j in range(N):
        if i == N-1 and j == N-1:
            break
        if dp[i][j]:
            tmp = board[i][j]
            if i + tmp <= N-1:
                dp[i+tmp][j] += dp[i][j]
            if j + tmp <= N-1:
                dp[i][j+tmp] += dp[i][j]

print(dp[-1][-1])
```

### [상미](./보드%20점프/상미.py)

```py

```

### [성구](./보드%20점프/성구.py)

```py
# 3372 보드 점프
import sys
from collections import deque
input = sys.stdin.readline


def main():
    N = int(input())
    field = [tuple(map(int, input().split())) for _ in range(N)]
    dp = [[0] * N for _ in range(N)]
    dp[0][0] = 1
    direction = [(0,1), (1,0)]
        
    for i in range(N):
        for j in range(N):
            if dp[i][j] == 0:
                continue
            if field[i][j] == 0:
                continue
            for di, dj in direction:
                ni, nj = i+di*field[i][j], j+dj*field[i][j]
                if ni < N and nj < N:
                    dp[ni][nj] += dp[i][j]
    
    
    # [print(dp[i]) for i in range(N)]

    print(dp[-1][-1])
                    
    return


if __name__ == "__main__":
    main()
```

### [영준](./보드%20점프/영준.py)

```py
```

<br/>

## [주사위 고르기](https://school.programmers.co.kr/learn/courses/30/lessons/258709)

### [민웅](./주사위%20고르기/민웅.py)

```py
```

### [상미](./주사위%20고르기/상미.py)

```py
```

### [성구](./주사위%20고르기/성구.py)

```py

```

### [영준](./주사위%20고르기/영준.py)

```py
```

<br/>

## [삼각 그래프](https://www.acmicpc.net/problem/4883)

### [민웅](./삼각%20그래프/민웅.py)

```py
```

### [상미](./삼각%20그래프/상미.py)

```py

```

### [성구](./삼각%20그래프/성구.py)

```py
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
```

### [영준](./삼각%20그래프/영준.py)

```py

```

<br/>

## [게리멘더링](https://www.acmicpc.net/problem/17471)

### [민웅](./게리멘더링/민웅.py)

```py
```

### [상미](./게리멘더링/상미.py)

```py

```

### [성구](./게리멘더링/성구.py)

```py
```

### [영준](./게리멘더링/영준.py)

```py

```

<br/>

## [최종 순위](https://www.acmicpc.net/problem/3665)

### [민웅](./최종%20순위/민웅.py)

```py
```

### [상미](./최종%20순위/상미.py)

```py

```

### [성구](./최종%20순위/성구.py)

```py
```

### [영준](./최종%20순위/영준.py)

```py

```

<br/>

</details>

<br/><br/>


# 알고리즘 설명

<details markdown="1">
<summary>접기/펼치기</summary>

</details>
