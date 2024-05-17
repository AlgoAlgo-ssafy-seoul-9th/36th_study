// 4883 삼각 그래프
const fs = require("fs");

const settings = () => {
    const input = fs.readFileSync("/dev/stdin").toString().split("\n");
    const Ns = [];
    const fields = [];
    let line = 0
    let k = 0;
    while (1) {
        const n = Number(input[line++]);
        if (n === 0) break;
        const field = [];
        input.slice(line, line+n).forEach((e) => {
            field.push(e.split(" ").map(Number));
        })
        Ns.push(n);
        fields.push(field);
        k ++;
        line += n
    }
    
    return [k, Ns, fields];
}


const main = (k) => (Ns) => (fields) => {
    for (let t = 1; t <= k; t ++){
        const [N, field] = [Ns[t-1], fields[t-1]];
        
        const dp = Array(N).fill(0);
        for (let i=0;i<N;i++) dp[i] = [0,0,0];
        dp[0][1] = field[0][1];
        dp[0][2] = field[0][2] + dp[0][1];
        
        for (let i = 1; i < N; i ++){
            if (i === 1) {
                dp[i][0] = field[i][0] + dp[i-1][1];
                dp[i][1] = field[i][1] + Math.min(dp[i-1][1],dp[i-1][2], dp[i][0]);
            }else{
                dp[i][0] = field[i][0] + Math.min(dp[i-1][0], dp[i-1][1]);
                dp[i][1] = field[i][1] + Math.min(...dp[i-1], dp[i][0]);
            }
            dp[i][2] = field[i][2] + Math.min(dp[i-1][1], dp[i-1][2], dp[i][1]);
        }

        console.log(`${t}. ${dp[N-1][1]}`);
    }
}



const [k, Ns, fields] = settings();

main(k)(Ns)(fields)
