function [ tp, fn, fp, tn ] = confusionMatrixCal( actual , predicted )
 al = length(actual);
 tp = 0; 
 fn =0; 
 fp =0; 
 tn =0;
 for i = 1:al
  if (actual(i) == 0)
    if(predicted(i) == 0)
      tn = tn + 1;
    else
      fp = fp + 1; 
    end
  else
   if(predicted(i) == 1)
     tp = tp + 1;
   else
     fn = fn + 1;
   end
  end
  %fprintf('\n %f %f %f %f',tp,fn,fp,tn);
 end
end