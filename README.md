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
from itertools import combinations, product


def solution(dice):
    answer = []
    max_value = 0
    def bs_win(val, lst):
        l, r = 0, len(lst)-1

        while l <= r:
            mid = (l + r)//2
            if lst[mid] >= val:
                r = mid - 1
            else:
                l = mid + 1
        return l

    def bs_lose(val, lst):
        l, r = 0, len(lst)-1
        while l <= r:
            mid = (l + r) // 2
            if lst[mid] > val:
                r = mid - 1
            else:
                l = mid + 1

        return len(lst) - l

    def calculate_rate(A, B, C, m):
        win = 0
        lose = 0
        tmp = [i for i in range(6)]
        prod = product(tmp, repeat=len(C))
        sum_lst = []
        for p in prod:
            tmp_sum = 0
            for k in range(len(A)):
                tmp_sum += A[k][p[k]]
            sum_lst.append(tmp_sum)
        sum_lst.sort()
        # print(sum_lst)
        prod = product(tmp, repeat=len(C))
        for p in prod:
            tmp_sum = 0
            for k in range(len(B)):
                tmp_sum += B[k][p[k]]
            win += bs_win(tmp_sum, sum_lst)
            lose += bs_lose(tmp_sum, sum_lst)
        ans_win = []
        ans_lose = []
        for i in range(2*len(C)):
            if i in C:
                ans_win.append(i+1)
            else:
                ans_lose.append(i+1)
        if win >= lose and win > m:
            m = win
            return ans_lose, m
        if lose >= win and lose > m:
            m = lose
            return ans_win, m
        return False
    l = len(dice)

    n_lst = [i for i in range(l)]
    comb = list(combinations(n_lst, l // 2))

    cl = len(comb) // 2

    for i in range(cl):
        c = comb[i]
        a = []
        b = []
        for j in range(l):
            if j in c:
                a.append(dice[j])
            else:
                b.append(dice[j])
        check = calculate_rate(a, b, c, max_value)
        if check:
            answer = check[0]
            max_value = check[1]

    return answer

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
# 3665_최종순위_final ranking
import sys
from collections import deque
input = sys.stdin.readline

T = int(input())

for _ in range(T):
    N = int(input())
    rank = list(map(int, input().split()))

    M = int(input())
    indegree = [0]*(N+1)
    adjL = [[] for _ in range(N+1)]

    for i in range(N):
        for j in range(i+1, N):
            adjL[rank[i]].append(rank[j])
            indegree[rank[j]] += 1

    # print(adjL)
    # print(indegree)
    for i in range(M):
        a, b = map(int, input().split())
        if a in adjL[b]:
            indegree[b] += 1
            indegree[a] -= 1
            adjL[a].append(b)
            adjL[b].remove(a)
        else:
            indegree[a] += 1
            indegree[b] -= 1
            adjL[a].remove(b)
            adjL[b].append(a)

    q = deque()

    for i in range(1, N+1):
        if not indegree[i]:
            q.append(i)

    if len(q) > 1:
        print('?')
        break

    ans_rank = []
    cnt = 0
    while q:
        now = q.popleft()
        ans_rank.append(now)
        cnt += 1

        for node in adjL[now]:
            indegree[node] -= 1
            if not indegree[node]:
                q.append(node)
    if cnt == N:
        print(*ans_rank)
    else:
        print('IMPOSSIBLE')

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
