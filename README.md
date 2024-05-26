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
        a_win = []
        a_lose = []
        for i in range(2*len(C)):
            if i in C:
                a_win.append(i+1)
            else:
                a_lose.append(i+1)
        if win >= lose and win > m:
            m = win
            return a_lose, m
        if lose >= win and lose > m:
            m = lose
            return a_win, m
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
# 주사위 고르기

def solution(dice):
    l = len(dice)
    stack = []
    
    
    def bsWin(target, cand):
        s,e = 0, len(cand)-1
        
        while s <= e:
            m = (s+e)//2
            
            if cand[m] >= target:
                e = m - 1
            else:
                s = m + 1
        
        return s
    
    
    def bsLose(target, cand):
        s,e = 0, len(cand)-1
        
        while s <= e:
            m = (s+e)//2
            
            if cand[m] > target:
                e = m - 1
            else:
                s = m + 1
        
        return len(cand) - s
    
    
    
    def makeCand(arr):
        cand = []
        s = [(0, 0)]
        
        while s:
            i, d = s.pop()
            
            if i == len(arr):
                cand.append(d)
                continue
            
            for j in range(6):
                s.append((i+1, d+dice[arr[i]][j]))
        
        return sorted(cand)
        
    
    
    
    def checkWin():
        A = set(stack)
        B = []
        
        for i in range(l):
            if i not in A:
                B.append(i)
        
        cand_A = makeCand(stack)
        cand_B = makeCand(B)
        
        cnt_w = 0
        cnt_l = 0
        for a in cand_A:
            cnt_w += bsWin(a, cand_B)
            cnt_l += bsLose(a, cand_B)
        
        
        if cnt_w >= cnt_l:
            return (cnt_w, list(map(lambda x:x+1, stack)))
        else:
            return (cnt_l, list(map(lambda x:x+1, B)))
    
    
    def bt(idx, maxv, ans):
        
        if idx == l // 2:
            return (maxv, ans)
        
        if len(stack) == l // 2:
            v, arr = checkWin()

            if maxv <= v:
                ans = [*arr]
                maxv = v
            return (maxv, ans)
        
        
        for i in range(idx+1, l):
            stack.append(i)
            maxv, ans = bt(i, maxv, ans)
            stack.pop()
        
        return (maxv, ans)
    
    _, answer = bt(-1, 0, [])
    
    return answer
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
# 17471 게리맨더링
import sys
from collections import deque
input = sys.stdin.readline


def main():
    # 입력
    N = int(input())
    population = tuple(map(int,input().split()))
    
    # 이웃 그래프
    graph = [[] for _ in range(N)]
    total_population = sum(population)

    # 입력
    for i in range(N):
        _, *near = map(int, input().split())
        graph[i] = set(map(lambda x:x-1, near))

    # 2일 때 예외 케이스 
    if N == 2:
        print(abs(population[0]-population[1]))
        return
    
    # 구현
    stack = []          # 순열 저장용
    visited = [0] * N   

    # bfs 통한 연결 확인
    def check(is_stack):
        if is_stack:        # stack 확인
            que = deque([stack[0]])
            v = set([stack[0]])
            compare = set(stack)
        else:               # stack에 없는 집단 확인
            tmp = []
            for i in range(N):
                if i not in stack:
                    tmp.append(i)
                    

            que = deque([tmp[0]])
            v = set([tmp[0]])
            compare = set(tmp)

        while que:
            spot = que.popleft()

            for node in graph[spot]:
                if not (node in v) and node in compare:
                    v.add(node)
                    que.append(node)

        
        return 1 if v == compare else 0
        

    # 탐색
    def bt(city, minv, men):
        # minv 초기화
        if len(stack) > 1:
            if check(0) and check(1):
                minv = min(minv, abs(total_population-men*2))
        elif len(stack) == 1:
            if check(0):
                minv = min(minv, abs(total_population-men*2))
        
        # 절반까지만 탐색(이후는 중복)
        if len(stack) == N//2:
            return minv



        for i in range(city+1, N):
            if not visited[i]:
                visited[i] = 1
                stack.append(i)
                minv = min(minv,bt(i, minv, (men+population[i])))
                visited[i] = 0
                stack.pop()
        
        return minv
    

    ans = bt(-1, 10000, 0)
    if ans == 10000:
        print(-1)
    else:
        print(ans)

    return


if __name__ == "__main__":
    main()
```

### [영준](./게리멘더링/영준.py)

```py
# 1개 ~ N-1개의 원소를 가진 부분집합과 나머지에 대해 각각 탐색
# 빠짐없이 탐색 가능하면 인구 차이 비교
# 두 그룹의 차이만 필요하므로 011 이나 100으로 구분되는 그룹은 같음

def bfs(s, N, G):
    q = [s]
    v = [0]*(N+1)
    v[s] = 1
    cnt = 0
    while q:
        t = q.pop(0)
        cnt += 1
        for i in adjList[t][1:]:
            if i in G and v[i] == 0:
                q.append(i)
                v[i] = 1
    if len(G)==cnt:
        return 1
    else:
        return 0

N = int(input())
p = [0] + list(map(int, input().split()))   # 1 ~ N 도시의 인구
adjList = [[]]
for _ in range(N):
    adjList.append(list(map(int, input().split())))

minV = 1000
for i in range(1, (1<<N)//2):
    A = []  # 선거구
    B = []
    pa = 0
    pb = 0
    for j in range(N):
        if i&(1<<j):    # j번 도시의 소속 선거구
            A.append(j+1)
            pa += p[j+1]
        else:
            B.append(j+1)
            pb += p[j+1]
    if bfs(A[0], N, A) and bfs(B[0], N, B):
        if minV > abs(pa-pb):
            minV = abs(pa-pb)
if minV == 1000:
    minV = -1
print(minV)
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
