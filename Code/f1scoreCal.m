function [ precision,recall,f1,t ] = f1scoreCal( actual,x,theta )
 precision = zeros(21, 1); 
 recall = zeros(21,1); 
 f1 = zeros(21,1);
 th = 0;
 threshold = 0.0;
 for i = 1:+1:21
 pred = prediction(x, theta, threshold);
 [tp, fn, fp, tn] = confusionMatrixCal(actual, pred);
 
 if(tp == 0)
     precision(i) = 0;
     recall(i) = 0;
 else    
     precision(i) = tp/(tp+fp);
     recall(i) = tp/(tp+fn);
 end

 if(precision(i) + recall(i) == 0)
     f1(i) = 0;
 else    
     f1(i) = 2* (( precision(i) * recall(i) ) / ( precision(i) + recall(i) ));
 end    
 th(i) = threshold;
 threshold = threshold + 0.05;
 end
 t = th';
end

