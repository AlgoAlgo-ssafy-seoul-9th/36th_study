function solution(dice) {
    let ans = [];
    let maxv = 0;
    const A = [];
    
    const [N, half] = [dice.length, Number(dice.length/2)];
    
    
    const binarySearch = (candB) => (target) => {
        let [s, e] = [0, candB.length-1];
        let mid = 0;
        while (s < e){
            mid = parseInt((s+e)/2);
            if (candB[mid] === target){
                e = mid;
                break;
            }else if(candB[mid] < target){
                s = mid + 1;        
            }else {
                e = mid - 1;
            }
        }
        let index = -1;
        // console.log("E",e, "target", target)
        for (let i = e; i >=0; i--){
            index = i;
            if (candB[i] < target){ break;}
        }
        
        if (index <= 0 && candB[0] >= target) return 0;
        // console.log(index+1)
        return index + 1;
        
    }

    
    const dfs = (array) => {
        // console.log(dice)
        let stack = [[0, 0]];
        const cand = [];
        while (stack.length > 0){
            const [i, val] = stack.pop();
            if (i === half){
                cand.push(val);
                continue;
            }
            for (let j =0; j < 6; j++){
                stack.push([i+1, val + dice[array[i]][j]])
            }
        }
        
        return cand;
    }
    
    
    const checkPercent = (B) => {
        const candA = dfs(A);
        const candB = dfs(B);
        
        candA.sort((a,b) => a-b);
        candB.sort((a,b) => a-b);
    
        const currentBS = binarySearch(candB);
        let wins = 0;
        // console.log("A",A,"B",B)
        // console.log("candA",candA);
        // console.log("candB",candB);
        for (a of candA){
            wins += currentBS(a);
        }
        // console.log(wins)
        if( maxv <= wins) {
            maxv = wins;
            ans = A.map((e) => e+1);
        }
        // console.log("maxv", maxv)
        // console.log("ans", ans)
        
    }
    
    
    const bt = (idx) => {
        if(A.length === half){
            const B = [];
            for (let i =0;i < N; i++){
                if (!A.includes(i)){
                    B.push(i);
                }
            }
            checkPercent(B)
            return ;    
        }
        
        for (let i = idx + 1;i < N; i++){
            if (!A.includes(i)){
                A.push(i);
                bt(i);
                A.pop();
            }
        }
        
    }
    
    bt(-1);
    // console.log(binarySearch([1,2,3,4,5])(5));

    
    return ans;
}