# 17471_게리맨더링_gerrymandering
import sys
from collections import deque
from itertools import combinations
input = sys.stdin.readline


def group_check(lst):
    visited = [0]*(N+1)
    visited[lst[0]] = 1
    q = deque()
    q.append(lst[0])
    while q:
        node = q.pop()
        for next_node in adjL[node]:
            if not visited[next_node] and next_node in lst:
                visited[next_node] = 1
                q.append(next_node)

    for n in lst:
        if not visited[n]:
            return False

    return True


N = int(input())
popul = list(map(int, input().split()))

adjL = [[] for _ in range(N+1)]
ans = float('inf')

for i in range(1, N+1):
    _, *n_lst = map(int, input().split())

    for n in n_lst:
        adjL[i].append(n)

n_lst = [i for i in range(1, N+1)]
for i in range(1, N//2+1):
    comb = combinations(n_lst, i)

    for c in comb:
        q1 = []
        q2 = []
        for idx in range(1, N+1):
            if idx in c:
                q1.append(idx)
            else:
                q2.append(idx)

        if group_check(q1) and group_check(q2):
            p1 = sum(popul[i-1] for i in q1)
            p2 = sum(popul[i-1] for i in q2)
            ans = min(ans, abs(p1 - p2))

if ans == float('inf'):
    print(-1)
else:
    print(ans)
