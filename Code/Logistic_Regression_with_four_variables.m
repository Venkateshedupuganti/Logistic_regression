clc;
data = csvread('BreastCancerData.csv');

%First Variable
xx1 = data(1:29,5);
xx2 = data(48:138,5);
x1 =vertcat(xx1,xx2);

cxx1 = data(30:38,5);
cxx2 = data(139:168,5);
cx1 =vertcat(cxx1,cxx2);

txx1 = data(39:47,5);
txx2 = data(169:198,5);
tx1 =vertcat(txx1,txx2);

%Second Variable

xx1 = data(1:29,7);
xx2 = data(48:138,7);
x2 =vertcat(xx1,xx2);

cxx1 = data(30:38,7);
cxx2 = data(139:168,7);
cx2 =vertcat(cxx1,cxx2);

txx1 = data(39:47,7);
txx2 = data(169:198,7);
tx2 =vertcat(txx1,txx2);

%Third Variable%

xx1 = data(1:29,10);
xx2 = data(48:138,10);
x3 =vertcat(xx1,xx2);

cxx1 = data(30:38,10);
cxx2 = data(139:168,10);
cx3 =vertcat(cxx1,cxx2);

txx1 = data(39:47,10);
txx2 = data(169:198,10);
tx3 =vertcat(txx1,txx2);

%Fourth Variable%

xx1 = data(1:29,13);
xx2 = data(48:138,13);
x4 =vertcat(xx1,xx2);

cxx1 = data(30:38,13);
cxx2 = data(139:168,13);
cx4 =vertcat(cxx1,cxx2);

txx1 = data(39:47,13);
txx2 = data(169:198,13);
tx4 =vertcat(txx1,txx2);

%%%%%%%%%%

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

fprintf('Logistic Regression with four variable');
x =[ones(m,1), x1, x2,x3,x4];
[theta , min_cost] = fminunc(@(th)(costFunctionCal(x, y, m, th)), zeros(5, 1), options);
fprintf('theta values: \n');
fprintf(' %f \n', theta);
fprintf('Minimum Cost:\t');
fprintf('%f\n',min_cost);

cx = [ones(c,1),cx1,cx2,cx3,cx4];
costFunctionCal(cx, cy,c, theta)
[tprate, fprate] = AUCcal(cx, cy, theta);
[precision, recall, f1score,th] = f1scoreCal(cy, cx, theta);
Int = trapz(fprate, tprate);
fprintf('AUC value: ');
fprintf('%f\n',Int);


figure(1); hold on;
xlabel('Flase Positive Rate / 1-Specificicy');
ylabel('True Positve Rate / Sensitivicy');
title('AUC graph for different threshold values');
plot(fprate, tprate, 'r--', 'linewidth', 2);
legend('Four features');
hold off;

figure(2); hold on;
xlabel('Recall');
ylabel('Precision');
title('Precision vs Recall for different threshold values');
plot(recall, precision, 'r--', 'linewidth', 2);
legend('Four Features');
hold off;
size(th)
size(f1score)

figure(3); hold on;
xlabel('Threshold');
ylabel('F measure');
title('Fmeasure vs Threshold values');
plot(th, f1score, 'r--', 'linewidth', 2);
legend('Four features');
hold off;

%Testing
fprintf('Four varaibles...\n')
tx = [ones(n, 1), tx1,tx2,tx3,tx4];
costFunctionCal(tx, ty,n, theta)
p1 = prediction(tx, theta, 0.15);
[tp, fn, fp, tn] = confusionMatrixCal(ty, p1)
