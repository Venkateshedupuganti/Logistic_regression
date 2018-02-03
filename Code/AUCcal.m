function [ tprate, fprate ] = AUCcal( x,y,theta )
 tprate = 0;
 fprate = 0;
 threshold = 0;
 for i = 21:-1:1
  if( threshold < 0.95 )
    pred = prediction(x,theta, threshold);
    [tp, fn, fp, tn] = confusionMatrixCal(y, pred);
    if(tp+fn == 0) 
        tprate(i) = 0;
    else    
        tprate(i) = tp/(tp+fn);
    end
    if(tn+fp ==0)
        fprate(i) = 0;
    else    
        fprate(i) = fp/(tn+fp);
    end    
  else
   tprate(i) = 0;
   fprate(i) = 0;
  end
 threshold = threshold + 0.05;
end
end




