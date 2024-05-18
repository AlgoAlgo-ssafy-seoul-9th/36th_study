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