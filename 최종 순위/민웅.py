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
