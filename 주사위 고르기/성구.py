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