
这是赵世钰老师的《强化学习的数学原理》的课程笔记


# 第1课 基本概念




## 策略 policy



reward

trajectory : 轨迹

episode ：  episodic task    terminal states


MDP:  markov decision process

sets
State
Action
Reward

memoryless property:
$$p(s_{t+1}|a_{t+1},s_t,....,a_1,s_0) = p(s_{t+1}|a_{t+1},s_t)$$
$$p(r_{t+1}|a_{t+1},s_t,....,a_1,s_0) = p(r_{t+1}|a_{t+1},s_t)$$

状态无关的数学模型。




return value ： 可以评估策略



# 第2课   贝尔曼公式



### Bootstrapping :  
从自己出发，不断的迭代，所得到的结果。  一个状态的值，依赖更一个状态值
这种思维方式，数学表现形式

$$v_1 = r_1 + \gamma(r_2 + \gamma r_3 + ...) = r_1 + \gamma v_2 $$
$$v_2 = r_2 + \gamma v_3 $$
$$v_3 = r_3 + \gamma v_4$$
$$v_4 = r_4 + \gamma v_1$$

$$ v = r + \gamma Pv $$

### state value

discount return : $$G_t = R_{t+1} + \gamma R_{t+2} + \gamma ^2R_{t+3} + ...$$  
从状态$$S_t$$出发多步轨迹后获得的折扣奖励， 也是个随机变量

state value定义: 
$$v_{\pi}(s) = E(G_t|S_t = s)$$

return 和 state value 的区别：

* 单个轨迹 / 多个轨迹 return再求平均


### Bellman equation

不同状态间的state value间关系 ？


matrix form:

$$v_\pi(s)=r_\pi + \gamma P_\pi v_\pi$$


怎么求解：？

### Action value



****

# 第3课   贝尔曼最优公式



RL ： 寻找最优策略  optimal policy

BOE : Bellman optimality equation 



$$v=max_\pi(r_\pi + \gamma P_\pi v)$$










      

