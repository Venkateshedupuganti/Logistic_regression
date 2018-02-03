function [ predic ] = prediction( x, theta, threshold )
 v = size(x,1);
 predic = zeros(v,1);
 predic = (sigmodFunction(x * theta) >= threshold);
end

