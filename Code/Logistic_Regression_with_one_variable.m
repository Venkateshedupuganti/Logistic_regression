clc;
data = csvread('BreastCancerData.csv');
xx1 = data(1:29,5);
xx2 = data(48:138,5);
x1 =vertcat(xx1,xx2);

cxx1 = data(30:38,5);
cxx2 = data(139:168,5);
cx1 =vertcat(cxx1,cxx2);

txx1 = data(39:47,5);
txx2 = data(169:198,5);
tx1 =vertcat(txx1,txx2);

y1 = data(1:29,2);
y2 = data(48:138,2);
y =vertcat(y1,y2);

cy1 = data(30:38,2);
cy2 = data(139:168,2);
cy =vertcat(cy1,cy2);

ty1 = data(39:47,2);
ty2 = data(169:198,2);
ty =vertcat(ty1,ty2);


m = length(y);
n = length(ty);
c = length(cy);

options = optimset('GradObj', 'on', 'MaxIter', 700);

fprintf('Logistic Regression with one variable');
x =[ones(m,1), x1];
[theta , min_cost] = fminunc(@(th)(costFunctionCal(x, y, m, th)), zeros(2, 1), options);
fprintf('theta values: \n');
fprintf(' %f \n', theta);
fprintf('Minimum Cost:\t');
fprintf('%f\n',min_cost);

cx = [ones(c,1),cx1];
costFunctionCal(cx, cy,c, theta)
[tprate, fprate] = AUCcal(cx, cy, theta);
[precision, recall, f1score,th] = f1scoreCal(cy, cx, theta);
%fprintf('%f %f %f',precision,recall,f1score);
%fprintf('%f %f',tprate,fprate);
Int = trapz(fprate, tprate);
fprintf('AUC value: ');
fprintf('%f\n',Int);

figure(1); hold on;
xlabel('Flase Positive Rate / 1-Specificicy');
ylabel('True Positve Rate / Sensitivicy');
title('AUC graph for different threshold values');
plot(fprate, tprate, 'r--', 'linewidth', 2);
legend('Single feature');
hold off;

figure(2); hold on;
xlabel('Recall');
ylabel('Precision');
title('Precision vs Recall for different threshold values');
plot(recall, precision, 'r--', 'linewidth', 2);
legend('Single feature');
hold off;

figure(3); hold on;
xlabel('Threshold');
ylabel('F measure');
title('Fmeasure vs Threshold values');
plot(th, f1score, 'r--', 'linewidth', 2);
legend('Single feature');
hold off;

%Testing
fprintf('\nsingle varaible...\n')
tx = [ones(n, 1), tx1];
costFunctionCal(tx, ty,n, theta)
p1 = prediction(tx, theta, 0.2);
[tp, fn, fp, tn] = confusionMatrixCal(ty, p1)
%fprintf('\n %f %f %f %f',tp,fn,fp,tn);
