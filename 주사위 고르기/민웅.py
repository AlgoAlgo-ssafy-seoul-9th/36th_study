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
