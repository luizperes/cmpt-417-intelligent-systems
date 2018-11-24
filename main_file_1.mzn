% PIZZA PROBLEM AS PER DESCRIBED ON:
% http://picat-lang.org/lp_cp_pc/Pizza.html
int: n = 4;
array[1..n] of int: price = [10, 5, 20, 15];
int: m = 2;
array[1..m] of 1..m: buy  = [1, 2];
array[1..m] of 1..m: free = [1, 1];

% enum describing the 3 options:
% we buy the pizza
% we get free pizza
% we don't use the voucher
enum OPTION = {BUY, NO_VOUCHER, FREE};

% reason of pizza
array[1..n] of var OPTION: reason;

% array of used vouchers
array[1..m] of var bool: used;

constraint forall(voucher in 1..m)
            (used[voucher]
            <->
            sum(pizza in 1..n)(reason[pizza] = BUY) <= buy[voucher]);

constraint forall(voucher in 1..m)
            (sum(pizza in 1..n)(reason[pizza] = BUY)
            >=
            used[voucher] * buy[voucher]);

constraint forall(voucher in 1..m)
            (sum(pizza in 1..n)(reason[pizza] = FREE)
            >=
            used[voucher] * free[voucher]);
            
constraint forall(pizza1, pizza2 in 1..n)
            ((
              (reason[pizza1] = BUY)
              /\
              (reason[pizza2] = FREE))
              ->
            price[pizza2] <= price[pizza1]);

int: total = sum(price);

var 0..total: COST =
                (sum(pizza in 1..n)
                  (((reason[pizza] = NO_VOUCHER) \/ (reason[pizza] = BUY))
                *
                price[pizza]));

solve :: int_search(reason, input_order, indomain_min, complete)
         minimize COST;

output  
  ["pizza = "++show(reason)++";\ncost = "++show(COST)++";\n"];