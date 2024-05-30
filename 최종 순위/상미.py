import sys
input = sys.stdin.readline

T = int(input())
for _ in range(T):
    n = int(input())    # 팀의 수
    prevRanks = list(map(int, input().split()))     # 작년 순위(인덱스가 등수)
    m = int(input())    # 순위가 바뀐 쌍의 수
    if m == 0:
        print(*prevRanks)
        continue

    prevTeams = []
    for idx, team in enumerate(prevRanks):
        prevTeams.append((idx+1, team))       # 순위, 팀번호
    print(prevTeams)
    changes = []
    for _ in range(m):
        t1, t2 = map(int, input().split())      # 바뀐 쌍
        can = []
        for rank, team in prevTeams:
            if t1 == team:
                can.append((rank, t1))
            if t2 == team:
                can.append((rank, t2))
        changes.append(can)
    print(changes)
    

'''
5 4 3 2 1
-
(2, 4) (3, 4)
-
정렬(높은 순위 순으로)
(4, 3) (4, 2)
5 3 4 2 1
5 3 2 4 1

'''

'''
1 2 3 4
-
(1, 2) (3, 4) (2, 3)
-
정렬
(1, 2) (2, 3) (3, 4)
2 1 3 4



'''