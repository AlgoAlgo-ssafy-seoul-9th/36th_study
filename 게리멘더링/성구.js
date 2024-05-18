// 17471 게리멘더링
const fs = require("fs");

const settings = () => {
    const input = fs.readFileSync("/dev/stdin").toString().split("\n");
    // const input = fs.readFileSync("input.txt").toString().split("\n");
    const N = Number(input[0]);
    const population = input[1].split(" ").map(Number);
    const graph = [];
    input.slice(2,).forEach((e) => {
        let tmp = e.split(" ").map(Number).slice(1, );
        for (let v in tmp){
            tmp[v] --;
        }
        graph.push(new Set(tmp))
    });
    
    return [N, population, graph];
}


class Queue {
    constructor(){
        this.store = {};
        this.front = 0;
        this.rear = 0;
    }

    size() {
        // console.log("size", this.store[this.rear])
        if (this.store[this.front] === undefined){
            return 0;
        }else{
            return this.rear - this.front + 1;
        }
    }

    push(val) {
        this.store[this.rear++] = val;
        // console.log("push",this.store, this.rear)
    }

    pop(){
        const element = this.store[this.front];
        delete this.store[this.front++];
        return element;
    }
}


const main = (N) => (population) => (graph) => {
    const stack = [];
    const totalPopulation = population.reduce((prev, now)=>prev+now, 0);
    const visited = Array(N).fill(0);
    const check = (isStack) => {
        const que = new Queue();
        let v, compare;
        if (isStack) {
            que.push(stack[0]);
            v = new Set([stack[0]]);
            compare = new Set(stack);
        }else{
            const tmp = [];
            for (let k = 0; k < N; k++){
                if (!(stack.includes(k))) tmp.push(k);
            }
            // console.log("tmp",tmp)
            que.push(tmp[0])
            v = new Set([tmp[0]]);
            compare = new Set(tmp);
        }
        // console.log(que.size());
        while (que.size() > 0){
            const spot = que.pop();
            // console.log("spot",spot)
            for (let node of graph[spot]){
                if (compare.has(node) && !(v.has(node))){
                    v.add(node);
                    que.push(node);
                }
            }
        }
        for (const value of v){
            compare.delete(value);
        }
        // console.log("비교", compare.size===0, compare)
        return compare.size === 0 ? true : false;
    }

    const bt = (city, minv, men) =>{
        // console.log("stack", stack)
        // console.log("minv", minv, "men", men, "total", totalPopulation)
        if (stack.length === 1){
            if (check(false)){
                minv = Math.min(minv, Math.abs(totalPopulation - 2 * men))
            }
        }else if(stack.length > 1 && stack.length < Number(N/2)){
            if (check(true) && check(false)){
                minv = Math.min(minv, Math.abs(totalPopulation - 2 * men))
            }            
        }

        if (stack.length === Number(N/2)){
            if (check(true) && check(false)){
                minv = Math.min(minv, Math.abs(totalPopulation - 2 * men))
            }
            return minv;
        }

        for (let post=city+1; post < N; post ++ ){
            if (visited[post] === 0){
                stack.push(post);
                visited[post] = 1;
                minv = Math.min(minv, bt(post, minv, men+population[post]));
                visited[post] = 0;
                stack.pop();
            }
        }


        
        return minv;
    }

    const ans = bt(-1,100000,0);
    console.log(ans !== 100000 ? ans : -1);
}



const [N, population, graph] = settings();
main(N)(population)(graph);
