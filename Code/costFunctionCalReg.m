function [ cost_val_reg,grad ] = costFunctionCalReg( x,y,m,theta,lambda )

 cost_val = 0; %initialization
 cost_val_reg = 0;
 grad = zeros(size(theta));
 hyp = sigmodFunction(x * theta);
 cost_val = (-1/m) * sum( y.*log(hyp) + (1 - y) .* log(1 - hyp) );
 cost_val_reg =cost_val+ ((lambda/(2*m)) * sum( theta(2:end).^2 ));
 for i = 1:m
	grad = grad + ( hyp(i) - y(i) ) * x(i, :)';
 end
 grad = ((1/m) * grad) + ((lambda/m) * [0; theta(2:end)]);

end


