% THIS FILE EXISTS BY MEANS OF COMPARISONS WITH THE PROBLEM
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

%  define the possible values in the sets of pizzas
set of int: Pizzas = 1..n;
%  define the possible values in the sets of vouchers
set of int: Vouchers = 1..m;

% unary relation for the set of paid pizzas
var set of Pizzas: Paid;
% unary relation for the set of used coupons
var set of Vouchers: Used; 

% Justifies(v, p) holds if pizza p was paid and used voucher c
array[1..m, 1..n] of var bool: Justifies;

% UsedFor(v, p) holds if pizza p was free by using voucher c
array[1..m, 1..n] of var bool: UsedFor;

% ================= CONSTRAINTS
%
% C_1. If we paid for a pizza p, then it cannot be in the set of free pizzas
constraint forall(p in 1..n)
            ((p in Paid)
            <->
            not exists(v in Vouchers)(UsedFor[v, p]));

% C_2. If voucher v is used, then it must get at least one free pizza p with it
constraint forall(v in 1..m)
            ((v in Used)
            <->
            exists(p in 1..n)(UsedFor[v, p]));

% C_3. Any used voucher v must be justified by paying for exactly some pizzas p
constraint forall(v in 1..m)
            ((v in Used)
            ->
            sum(p in 1..n)(Justifies[v, p]) >= buy[v]);

% C_4. The number of free pizzas cannot be greater than what is possible by
%      using voucher v
constraint forall(v in 1..m)
            (sum(p in 1..n)(UsedFor[v, p]) <= free[v]);

% C_5. For every two pizzas p_1, p_2, if p_1 was a pizza we got for free
%      with voucher v and p_2 is a pizza we paid with voucher v, then the
%      price of p_1 must be less or equal the price of p_2
constraint forall(c in 1..m)
            (forall(p1, p2 in 1..n where p1 != p2)
              ((UsedFor[c, p1] /\ Justifies[c, p2])
                ->
                price[p1] <= price[p2]));

% C_6. Two vouchers v_1, v_2 cannot be justified by using the same paid pizza p
constraint forall(p in 1..n)
            (forall(v1, v2 in 1..m where v1 != v2)
              (Justifies[v1, p]
              ->
              not (Justifies[v2, p])));

% C_7. We pay for every pizza p used to justify use of a voucher v
constraint forall(p in 1..n)
            (forall(c in 1..m)(Justifies[c, p] -> (p in Paid)));

% C_8. The pairs in Justifies can only be consisting of a
%      pair (v, p) for voucher v and a pizza p
constraint forall(v in 1..m)
            (forall(p in 1..n)
              (Justifies[v, p]
               ->
              (v in 1..m /\ p in 1..n)));

% C_9. The pairs in UsedFor can only be consisting of a
%      pair (v, p) for voucher v and a pizza p
constraint forall(v in 1..m)
            (forall(p in 1..n)
              (UsedFor[v,p]
               ->
              (v in 1..m /\ p in 1..n)));

% get the total cost so we minimize it later
int: total = sum(price);
var int: COST =(sum(p in 1..n)(bool2int(p in Paid) * price[p]));

% C_10. The total cost ($k$, the sum of all paid pizzas) 
%       must be less or equal the sum of all pizza prices
constraint COST <= total;

solve minimize COST;

output [ "Paid = " ++ show(Paid) ++ "\n" ++
         "Used = " ++ show(Used) ++ "\n" ++
         "Justifies = " ++ show(Justifies) ++ "\n" ++
         "UsedFor = " ++ show(UsedFor) ++ "\n" ++
         "COST = " ++ show(COST) ];