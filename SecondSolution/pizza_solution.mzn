% PIZZA PROBLEM AS PER DESCRIBED ON:
% http://picat-lang.org/lp_cp_pc/Pizza.html
int: n;
array[1..n] of int: price;
int: m;
array[1..m] of 1..m: buy;
array[1..m] of 1..m: free;

% enum describing the 3 options:
% we buy the pizza
% we get free pizza
% we don't use the voucher
enum OPTION = { BUY, NO_VOUCHER, FREE };

% reason of pizza
array[1..n] of var OPTION: reason;

% array of used vouchers
array[1..m] of var bool: used;

constraint forall(voucher in 1..m)
            (used[voucher]
            <->
            buy[voucher] >= sum(pizza in 1..n)(reason[pizza] = BUY));

constraint forall(voucher in 1..m)
            (used[voucher] * buy[voucher]
            <=
            sum(pizza in 1..n)(reason[pizza] = BUY));

constraint forall(voucher in 1..m)
            (used[voucher] * free[voucher]
            <=
            sum(pizza in 1..n)(reason[pizza] = FREE));

% for every pair of pizza free pizzas are cheaper            
constraint forall(pizza1, pizza2 in 1..n)
            ((
              (reason[pizza1] = BUY)
              /\
              (reason[pizza2] = FREE)
              ->
            price[pizza2] <= price[pizza1]));

% get the total cost so we minimize it later
int: total = sum(price);
var 0..total: COST =
                (sum(pizza in 1..n)
                  (((reason[pizza] = NO_VOUCHER) \/ (reason[pizza] = BUY))
                *
                price[pizza]));

solve :: int_search(reason, input_order, indomain_min, complete)
         minimize COST;

output [ "pizza = " ++ show(reason) ++ "\n" ++
          "cost = " ++ show(COST) ];