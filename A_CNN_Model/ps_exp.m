function y = ps_exp(x,n) 

series = zeros(n,1); 

for i = 1:1:n 
    series(i) = x^(i-1) / factorial(i-1); 
end 

y = sum(series); 

