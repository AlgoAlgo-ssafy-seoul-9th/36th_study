# 3665 최종 순위
import sys
from collections import deque
input = sys.stdin.readline

def main():
    N = int(input())
    ref = tuple(map(int, input().split()))
    rank = [set() for _ in range(N+1)]
    # 팀에게 들어오는 간선 개수 저장
    visited = [0] * (N+1)
    # 나보다 순위가 낮은 팀들을 저장
    for i in range(N):
        for j in range(i+1, N):
            rank[ref[i]].add(ref[j])
            # 나보다 순위가 높으면 나를 바라보게 함
            visited[ref[j]] += 1
    # print(visited)
    # print(rank)

    M = int(input())
    for _ in range(M):
        a, b = map(int, input().split())
        if a in rank[b]:
            rank[b].remove(a)
            rank[a].add(b)
            visited[b] += 1
            visited[a] -= 1
        else:
            rank[b].add(a)
            rank[a].remove(b)
            visited[b] -= 1
            visited[a] += 1
    
        # print(visited)
        # print(rank)
    
    # 나를 보는 간선이 없는 경우 == 1등
    que = deque()
    for i in range(1, N+1):
        if not visited[i]:
            que.append(i)
    
    # print(que)

    # 1등이 2명이상이면 정보의 일관성 부족
    if len(que) > 1:
        print("?")
        return
    ans = []

    while que:
        spot = que.popleft()
        ans.append(spot)
        # print(spot, que, ans)
        # print(visited)
        
        # 내가 바라보는 노드들의 간선 개수 줄임
        for node in tuple(rank[spot]):
            visited[node] -= 1
            if not visited[node]:   # 줄이다가 없는 아이 발견하면 추가
                que.append(node)

    # 순위가 확정되지 않는 팀이 없는 경우 출력
    if len(ans) == N:
        print(*ans)
    else:
        print("IMPOSSIBLE")
    return


if __name__ == "__main__":
    for _ in range(int(input())):
        main()