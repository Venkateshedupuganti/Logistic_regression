function [ e ] = sigmodFunction( z )
 e = size(z);
 e = 1./(1 + exp(-z));

end

