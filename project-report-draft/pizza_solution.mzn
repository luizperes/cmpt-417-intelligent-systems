% PIZZA PROBLEM AS PER DESCRIBED ON:
% http://picat-lang.org/lp_cp_pc/Pizza.html

% ================= DEFINITIONS
%
% Initial Input
int: n;
array[1..n] of int: price;
int: m;
array[1..m] of int: buy;
array[1..m] of int: free;

% unary relation for the set of paid pizzas
array[1..n] of var bool: Paid;
% unary relation for the set of used coupons
array[1..m] of var bool: Used;

% Justifies(v, p) holds if pizza p was paid and used voucher c
array[1..m, 1..n] of var bool: Justifies;

% UsedFor(v, p) holds if pizza p was free by using voucher c
array[1..m, 1..n] of var bool: UsedFor;

% ================= CONSTRAINTS
%
% for every pizza p, if p in Paid then p not in UsedFor, for all v
constraint forall(p in 1..n)
            (Paid[p]
            <->
            not exists(v in 1..m)(UsedFor[v, p]));

% for every voucher v, if v in Used then v in UsedFor, for some p
constraint forall(v in 1..m)
            (Used[v]
            <->
            exists(p in 1..n)(UsedFor[v, p]));

% for every voucher v, if v in Used, the # of Justifies p cannot < buy
constraint forall(v in 1..m)
            (Used[v]
            ->
            sum(p in 1..n)(Justifies[v, p]) >= buy[v]);

% for every voucher v, if v in UsedFor, the # of p cannot > free
constraint forall(v in 1..m)
            (sum(p in 1..n)(UsedFor[v, p]) <= free[v]);

% for every pair p1, p2 if p1 in UsedFor and p2 in Justifies then p1 <= p2
constraint forall(p1, p2 in 1..n where p1 != p2, c in 1..m)
            ((UsedFor[c, p1] /\ Justifies[c, p2])
             ->
             price[p1] <= price[p2]);

% for every pair c1, c2 if c1 in Justifies a p then c2 not Justifies same p
constraint forall(v1, v2 in 1..m where v1 != v2, p in 1..n)
            (Justifies[v1, p]
            ->
            not (Justifies[v2, p]));

% we need to pay for the pizzas we used on coupon c
constraint forall(p in 1..n, c in 1..m)
            (Justifies[c, p] -> Paid[p]);

% check if v and p are in range in Justifies
constraint forall(v in 1..m, p in 1..n)
            (Justifies[v, p]
            ->
            (v in 1..m /\ p in 1..n));

% check if v and p are in range in UsedFor
constraint forall(v in 1..m, p in 1..n)
            (UsedFor[v,p]
            ->
            (v in 1..m /\ p in 1..n));

% get the total cost so we minimize it later
int: total = sum(price);
var int: COST =(sum(p in 1..n)(Paid[p] * price[p]));

% make sure the COST is at least as good as the total
constraint COST <= total;

solve minimize COST;

output [ "Paid = " ++ show(bool2int(Paid)) ++ "\n" ++
         "Used = " ++ show(bool2int(Used)) ++ "\n" ++
         "Justifies = " ++ show(bool2int(Justifies)) ++ "\n" ++
         "UsedFor = " ++ show(bool2int(UsedFor)) ++ "\n" ++
         "COST = " ++ show(COST) ];