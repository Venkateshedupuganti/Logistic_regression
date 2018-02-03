clc;clear all;
data = csvread('BreastCancerData.csv');

%First Variable
xx1 = data(1:29,5);
xx2 = data(48:138,5);
x1 =scaling(vertcat(xx1,xx2));

cxx1 = data(30:38,5);
cxx2 = data(139:168,5);
cx1 =scaling(vertcat(cxx1,cxx2));

txx1 = data(39:47,5);
txx2 = data(169:198,5);
tx1 =scaling(vertcat(txx1,txx2));

%Second Variable

xx1 = data(1:29,7);
xx2 = data(48:138,7);
x2 =scaling(vertcat(xx1,xx2));

cxx1 = data(30:38,7);
cxx2 = data(139:168,7);
cx2 = scaling(vertcat(cxx1,cxx2));

txx1 = data(39:47,7);
txx2 = data(169:198,7);
tx2 =scaling(vertcat(txx1,txx2));

%Third Variable%

xx1 = data(1:29,10);
xx2 = data(48:138,10);
x3 =scaling(vertcat(xx1,xx2));

cxx1 = data(30:38,10);
cxx2 = data(139:168,10);
cx3 =scaling(vertcat(cxx1,cxx2));

txx1 = data(39:47,10);
txx2 = data(169:198,10);
tx3 =scaling(vertcat(txx1,txx2));

%Fourth Variable%

xx1 = data(1:29,13);
xx2 = data(48:138,13);
x4 = scaling(vertcat(xx1,xx2));

cxx1 = data(30:38,13);
cxx2 = data(139:168,13);
cx4 = scaling(vertcat(cxx1,cxx2));

txx1 = data(39:47,13);
txx2 = data(169:198,13);
tx4 = scaling(vertcat(txx1,txx2));

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

lambda =10;

options = optimset('GradObj', 'on', 'MaxIter', 700);

%%%%Single Variable%%%%

fprintf('Regularization with one variable');
X1 =[ones(m,1), x1];
[theta1 , min_cost1] = fminunc(@(th)(costFunctionCalReg(X1, y, m, th,lambda)), zeros(2, 1), options);
fprintf('theta values: \n');
fprintf(' %f \n', theta1);
fprintf('Minimum Cost:\t');
fprintf('%f\n',min_cost1);

CV1 = [ones(c,1),cx1];
[tprate1, fprate1] = AUCcal(CV1, cy, theta1);
[precision1, recall1, f1score1,th1] = f1scoreCal(cy, CV1, theta1);
Int1 = trapz(fprate1, tprate1);
fprintf('\nAUC value: ');
fprintf('%f\n',Int1);

%%%%% Two Variables %%%%

fprintf('Two variable');
X2 =[ones(m,1), x1, x2];
[theta2 , min_cost2] = fminunc(@(th)(costFunctionCalReg(X2, y, m, th,lambda)), zeros(3, 1), options);
fprintf('theta values: \n');
fprintf(' %f \n', theta2);
fprintf('Minimum Cost:\t');
fprintf('%f\n',min_cost2);

CV2 = [ones(c,1),cx1,cx2];
costFunctionCal(CV2, cy,c, theta2)
[tprate2, fprate2] = AUCcal(CV2, cy, theta2);
[precision2, recall2, f1score2,th2] = f1scoreCal(cy, CV2, theta2);
Int2 = trapz(fprate2, tprate2);
fprintf('\nAUC value: ');
fprintf('%f\n',Int2);


%%%% Three Variables %%%%


fprintf('Three variables');
X3 =[ones(m,1), x1, x2,x3];
[theta3 , min_cost3] = fminunc(@(th)(costFunctionCalReg(X3, y, m, th,lambda)), zeros(4, 1), options);
fprintf('theta values: \n');
fprintf(' %f \n', theta3);
fprintf('Minimum Cost:\t');
fprintf('%f\n',min_cost3);

CV3 = [ones(c,1),cx1,cx2,cx3];
[tprate3, fprate3] = AUCcal(CV3, cy, theta3);
[precision3, recall3, f1score3,th3] = f1scoreCal(cy, CV3, theta3);
Int3 = trapz(fprate3, tprate3);
fprintf('\nAUC value: ');
fprintf('%f\n',Int3);

%%%% Four Variables %%%%

fprintf('Four variables');
X4 =[ones(m,1), x1, x2,x3,x4];
[theta4 , min_cost4] = fminunc(@(th)(costFunctionCalReg(X4, y, m, th,lambda)), zeros(5, 1), options);
fprintf('theta values: \n');
fprintf(' %f \n', theta4);
fprintf('Minimum Cost:\t');
fprintf('%f\n',min_cost4);

CV4 = [ones(c,1),cx1,cx2,cx3,cx4];
costFunctionCal(CV4, cy,c, theta4)
[tprate4, fprate4] = AUCcal(CV4, cy, theta4);
[precision4, recall4, f1score4,th4] = f1scoreCal(cy, CV4, theta4);
Int4 = trapz(fprate4, tprate4);
fprintf('\nAUC value: ');
fprintf('%f\n',Int4);


figure(1); hold on;
xlabel('Flase Positive Rate / 1-Specificicy');
ylabel('True Positve Rate / Sensitivicy');
title('AUC graph for different threshold values');
plot(fprate1, tprate1, 'b--', 'linewidth', 2);
plot(fprate2, tprate2, 'g:', 'linewidth', 2);
plot(fprate3, tprate3, 'r-.', 'linewidth', 2);
plot(fprate4, tprate4, 'k', 'linewidth', 1);
legend('Single feature', 'Two features', 'Three features', 'Four features','location','southeast');
hold off;

figure(2); hold on;
xlabel('Recall');
ylabel('Precision');
title('Precision vs Recall for different threshold values');
plot(recall1, precision1, 'b--', 'linewidth', 2);
plot(recall2, precision2, 'g:', 'linewidth', 2);
plot(recall3, precision3, 'r-.', 'linewidth', 2);
plot(recall4, precision4, 'k', 'linewidth', 1);
legend('Single feature', 'Two features', 'Three features', 'Four features');
hold off;

figure(3); hold on;
xlabel('Threshold');
ylabel('F measure');
title('Fmeasure vs Threshold values');
plot(th1, f1score1, 'b--', 'linewidth', 2);
plot(th2, f1score2, 'g:', 'linewidth', 2);
plot(th3, f1score3, 'r-.', 'linewidth', 2);
plot(th4, f1score4, 'k', 'linewidth', 1);
legend('Single feature', 'Two features', 'Three features', 'Four features');
hold off;
