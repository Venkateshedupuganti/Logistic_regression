function [ cost_val,grad ] = costFunctionCal( x,y,m,theta )

 cost_val = 0; %initialization
 grad = zeros(size(theta));
 hyp = sigmodFunction(x * theta);
 cost_val = (-1/m) * sum( y.*log(hyp) + (1 - y) .* log(1 - hyp) );

 for i = 1:m
	grad = grad + ( hyp(i) - y(i) ) * x(i, :)';
 end
 grad = (1/m) * grad;

end

