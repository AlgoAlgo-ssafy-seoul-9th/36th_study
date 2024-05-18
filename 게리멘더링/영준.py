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
