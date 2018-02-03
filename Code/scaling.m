function [ sr ] = scaling( s )

 s_mean = mean(s);
 s_std = std(s);
 sr = ((s-s_mean)/(s_std));

end

