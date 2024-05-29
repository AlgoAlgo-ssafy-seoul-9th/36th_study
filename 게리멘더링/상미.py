import sys
from itertools import combinations
from collections import deque

input = sys.stdin.readline

def sol(area, graph):
    start = area[0]
    queue = deque([start])
    visited = [0]*N
    visited[start] = 1        # 시작점 방문 처리

    while queue:
        node = queue.popleft()
        for next in graph[node]:
            if not visited[next] and next in area:
                visited[next] = 1
                queue.append(next)
    if sum(visited) == len(area):
        return True

def pop_diff(g1, g2, pops):     # 인구 수 차이 구하는 함수
    pop1 = sum(pops[i] for i in g1)
    pop2 = sum(pops[i] for i in g2)
    return abs(pop1 - pop2)


N = int(input().strip())
pops = list(map(int, input().split()))  # 인구 수
graph = [[] for _ in range(N)]      

for i in range(N):
    data = list(map(int, input().split()))      # data[0]: i 구역과 인접한 구역 수
                                                # date[1] ~ : 인접한 구역의 번호 
    for j in range(1, len(data)):
        graph[i].append(data[j] - 1)        # 인덱스로 할거라 1 빼줌

minD = 100000000


for i in range(1, N//2 + 1):
    combi = list(combinations(range(N), i))
    for g1 in combi:
        g2 = []
        for x in range(N):
            if x not in g1:
                g2.append(x)

        if sol(g1, graph) and sol(g2, graph):
            diff = pop_diff(g1, g2, pops)
            if diff < minD:
                minD = diff

if minD == 100000000:
    print(-1)
else:
    print(minD)


