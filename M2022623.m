clc;
clear;
data=xlsread('2022623one-hot..xlsx');
randIndex=randperm(size(data,1));
data_new=data(randIndex,:);
data_train=data_new(1:end-2000,:);
data_train_x=data_train(:,2:end);
data_train_y=data_train(:,1);
data_test=data_new(end-1999:end,:);
data_test_x=data_test(:,2:end);
data_test_y=data_test(:,1);
load('KNN3.mat');
yfit_train=KNN3.predictFcn(data_train_x); 
yfit_test=KNN3.predictFcn(data_test_x); 
[A,~] = confusionmat(data_train_y,yfit_train);%产生训练集+验证集混淆矩阵
KNN_train_precise = A(1,1)/(A(1,1) + A(2,1));
KNN_train_recall = A(1,1)/(A(1,1) + A(1,2));
KNN_train_accurance=(A(1,1)+A(2,2))/(A(1,1)+A(2,1)+A(1,2)+A(2,2));
KNN_train_F1 = 2 * KNN_train_precise * KNN_train_recall /(KNN_train_precise+ KNN_train_recall );
%KNN模型训练集评价指标
[B,~] = confusionmat(data_test_y,yfit_test);%产生测试集混淆矩阵
KNN_test_precise = B(1,1)/(B(1,1)+B(2,1));%精确率
KNN_test_recall = B(1,1)/(B(1,1)+B(1,2));%召回率
KNN_test_accurance=(B(1,1)+B(2,2))/(B(1,1)+B(2,1)+B(1,2)+B(2,2));%准确率
KNN_test_F1 = 2 * KNN_test_precise * KNN_test_recall...
    /(KNN_test_precise+ KNN_test_recall );%F1
disp(['KNN模型训练集+验证集精确率为',num2str(KNN_train_precise)])
disp(['KNN模型训练集+验证集召回率为',num2str(KNN_train_recall)])
disp(['KNN模型训练集+验证集准确率为',num2str(KNN_train_accurance)])
disp(['KNN模型训练集+验证集F1值为',num2str(KNN_train_F1)])
disp(['KNN模型测试集精确率为',num2str(KNN_test_precise)])
disp(['KNN模型测试集召回率为',num2str(KNN_test_recall)])
disp(['KNN模型测试集准确率为',num2str(KNN_test_accurance)])
disp(['KNN模型测试集F1值为',num2str(KNN_test_F1)])
figure(1)
subplot(2,1,1)
cm = confusionchart(data_train_y,yfit_train);
xlabel('Target Class');
ylabel('Output Class');
title('KNN model confusion matrix(Training set)')
subplot(2,1,2)
cm2 = confusionchart(data_test_y,yfit_test);
xlabel('Target Class');
ylabel('Output Class');
title('KNN model confusion matrix(Test set)')
figure(2)%测试集图
AUC_KNN_test=plot_roc0(yfit_test,data_test_y);  
disp(['KNN回归测试集的AUC值为',num2str(AUC_KNN_test)]);
hold on
figure(3)%训练集图
AUC_KNN_train=plot_roc00(yfit_train,data_train_y);  
disp(['KNN回归训练集+验证集的AUC值为',num2str(AUC_KNN_train)]);
hold on
%KNN模型测试集评价指标
%------------------------------------------------------------------------
load('logistic3.mat');
yfit_train_logistic=logistic3.predictFcn(data_train_x); 
yfit_test_logistic=logistic3.predictFcn(data_test_x); 
[C,~] = confusionmat(data_train_y,yfit_train_logistic);%产生训练集+验证集混淆矩阵
logistic_train_precise = C(1,1)/(C(1,1) + C(2,1));
logistic_train_recall = C(1,1)/(C(1,1) + C(1,2));
logistic_train_accurance=(C(1,1)+C(2,2))/(C(1,1)+C(2,1)+C(1,2)+C(2,2));%准确率
logistic_train_F1 = 2 * logistic_train_precise * logistic_train_recall...
/(logistic_train_precise+ logistic_train_recall );
%logistic模型训练集评价指标
[D,~] = confusionmat(data_test_y,yfit_test_logistic);%产生测试集混淆矩阵
logistic_test_precise = D(1,1)/(D(1,1) + D(2,1));
logistic_test_recall = D(1,1)/(D(1,1) + D(1,2));
logistic_test_accurance=(D(1,1)+D(2,2))/(D(1,1)+D(2,1)+D(1,2)+D(2,2));%准确率
logistic_test_F1 = 2 * logistic_test_precise * logistic_test_recall...
    /(logistic_test_precise+ logistic_test_recall );
%logistics模型测试集评价指标
disp(['logistic模型训练集+验证集精确率为',num2str(logistic_train_precise)])
disp(['logistic模型训练集+验证集召回率为',num2str(logistic_train_recall)])
disp(['logistic模型训练集+验证集准确率为',num2str(logistic_train_accurance)])
disp(['logistic模型训练集+验证集F1值为',num2str(logistic_train_F1)])
disp(['logistic模型测试集精确率为',num2str(logistic_test_precise)])
disp(['logistic模型测试集召回率为',num2str(logistic_test_recall)])
disp(['logistic模型测试集准确率为',num2str(logistic_test_accurance)])
disp(['logistic模型测试集F1值为',num2str(logistic_test_F1)])
figure(4)
subplot(2,1,1)
cm = confusionchart(data_train_y,yfit_train_logistic);
xlabel('Target Class');
ylabel('Output Class');
title('logistic model confusion matrix(Training set)')
subplot(2,1,2)
cm2 = confusionchart(data_test_y,yfit_test_logistic);
xlabel('Target Class');
ylabel('Output Class');
title('logistic model confusion matrix(Test set)')
figure(2)%测试集图
AUC_logistic_test=plot_roc1(yfit_test_logistic,data_test_y);  
disp(['logistic回归测试集的AUC值为',num2str(AUC_logistic_test)]); 
hold on
figure(3)%训练集图
AUC_logistic_train=plot_roc2(yfit_train_logistic,data_train_y);  
disp(['logistic回归训练集+验证集的AUC值为',num2str(AUC_logistic_train)]); 
hold on
%logistic模型部分
%--------------------------------------------------------------------
load('Bagging3.mat');
yfit_train_Bagging=Bagging3.predictFcn(data_train_x); 
yfit_test_Bagging=Bagging3.predictFcn(data_test_x); 
[E,~] = confusionmat(data_train_y,yfit_train_Bagging);%产生训练集+验证集混淆矩阵
Bagging_train_precise = E(1,1)/(E(1,1) +E(2,1));
Bagging_train_recall = E(1,1)/(E(1,1) + E(1,2));
Bagging_train_accurance=(E(1,1)+E(2,2))/(E(1,1)+E(2,1)+E(1,2)+E(2,2));%准确率
Bagging_train_F1 = 2 * Bagging_train_precise * Bagging_train_recall...
/(Bagging_train_precise+ Bagging_train_recall );
%Bagging模型训练集评价指标
[F,~] = confusionmat(data_test_y,yfit_test_Bagging);%产生测试集混淆矩阵
Bagging_test_precise = F(1,1)/(F(1,1) + F(2,1));
Bagging_test_recall = F(1,1)/(F(1,1) +F(1,2));
Bagging_test_accurance=(F(1,1)+F(2,2))/(F(1,1)+F(2,1)+F(1,2)+F(2,2));%准确率
Bagging_test_F1 = 2 * Bagging_test_precise * Bagging_test_recall...
    /(Bagging_test_precise+ Bagging_test_recall );
%Bagging模型测试集评价指标
disp(['Bagging模型训练集+验证集精确率为',num2str(Bagging_train_precise)])
disp(['Bagging模型训练集+验证集召回率为',num2str(Bagging_train_recall)])
disp(['Bagging模型训练集+验证集准确率为',num2str(Bagging_train_accurance)])
disp(['Bagging模型训练集+验证集F1值为',num2str(Bagging_train_F1)])
disp(['Bagging模型测试集精确率为',num2str(Bagging_test_precise)])
disp(['Bagging模型测试集召回率为',num2str(Bagging_test_recall)])
disp(['Bagging模型测试集准确率为',num2str(Bagging_test_accurance)])
disp(['Bagging模型测试集F1值为',num2str(Bagging_test_F1)])
figure(5)
subplot(2,1,1)
cm = confusionchart(data_train_y,yfit_train_Bagging);
xlabel('Target Class');
ylabel('Output Class');
title('Bagging model confusion matrix(Training set)')
subplot(2,1,2)
cm2 = confusionchart(data_test_y,yfit_test_Bagging);
xlabel('Target Class');
ylabel('Output Class');
title('Bagging model confusion matrix(Test set)')
figure(2)%测试集图
AUC_Bagging_test=plot_roc3(yfit_test_Bagging,data_test_y);  
disp(['Bagging回归测试集的AUC值为',num2str(AUC_Bagging_test)]); 
hold on
figure(3)%训练集图
AUC_Bagging_train=plot_roc4(yfit_train_Bagging,data_train_y);  
disp(['Bagging回归训练集+验证集的AUC值为',num2str(AUC_Bagging_train)]); 
hold on
%Bagging模型部分
%----------------------------------------------------------------------
load('BP3.mat');
yfit_train_BP=BP3.predictFcn(data_train_x); 
yfit_test_BP=BP3.predictFcn(data_test_x); 
[G,~] = confusionmat(data_train_y,yfit_train_BP);%产生训练集+验证集混淆矩阵
BP_train_precise = G(1,1)/(G(1,1) +G(2,1));
BP_train_recall = G(1,1)/(G(1,1) + G(1,2));
BP_train_accurance=(G(1,1)+G(2,2))/(G(1,1)+G(2,1)+G(1,2)+G(2,2));%准确率
BP_train_F1 = 2 * BP_train_precise * BP_train_recall...
/(BP_train_precise+ BP_train_recall );
%BP模型训练集评价指标
[H,~] = confusionmat(data_test_y,yfit_test_BP);%产生测试集混淆矩阵
BP_test_precise = H(1,1)/(H(1,1) + H(2,1));
BP_test_recall = H(1,1)/(H(1,1) +H(1,2));
BP_test_accurance=(H(1,1)+H(2,2))/(H(1,1)+H(2,1)+H(1,2)+H(2,2));%准确率
BP_test_F1 = 2 * BP_test_precise * BP_test_recall...
    /(BP_test_precise+ BP_test_recall );
%BP模型测试集评价指标
disp(['BP模型训练集+验证集精确率为',num2str(BP_train_precise)])
disp(['BP模型训练集+验证集召回率为',num2str(BP_train_recall)])
disp(['BP模型训练集+验证集准确率为',num2str(BP_train_accurance)])
disp(['BP模型训练集+验证集F1值为',num2str(BP_train_F1)])
disp(['BP模型测试集精确率为',num2str(BP_test_precise)])
disp(['BP模型测试集召回率为',num2str(BP_test_recall)])
disp(['BP模型测试集准确率为',num2str(BP_test_accurance)])
disp(['BP模型测试集F1值为',num2str(BP_test_F1)])
figure(6)
subplot(2,1,1)
cm = confusionchart(data_train_y,yfit_train_BP);
xlabel('Target Class');
ylabel('Output Class');
title('BP model confusion matrix(Training set)')
subplot(2,1,2)
cm2 = confusionchart(data_test_y,yfit_test_BP);
xlabel('Target Class');
ylabel('Output Class');
title('BP model confusion matrix(Test set)')
figure(2)%测试集图
AUC_BP_test=plot_roc5(yfit_test_BP,data_test_y);  
disp(['BP回归测试集的AUC值为',num2str(AUC_BP_test)]); 
hold on
figure(3)%训练集图
AUC_BP_train=plot_roc6(yfit_train_BP,data_train_y);  
disp(['BP回归训练集+验证集的AUC值为',num2str(AUC_BP_train)]); 
hold on
%bp
%-----------------------------------------------------------------------
load('SVM3.mat');
yfit_train_SVM=SVM3.predictFcn(data_train_x); 
yfit_test_SVM=SVM3.predictFcn(data_test_x); 
[I,~] = confusionmat(data_train_y,yfit_train_SVM);%产生训练集+验证集混淆矩阵
SVM_train_precise = I(1,1)/(I(1,1) +I(2,1));
SVM_train_recall = I(1,1)/(I(1,1) + I(1,2));
SVM_train_accurance=(I(1,1)+I(2,2))/(I(1,1)+I(2,1)+I(1,2)+I(2,2));%准确率
SVM_train_F1 = 2 * SVM_train_precise * SVM_train_recall...
/(SVM_train_precise+ SVM_train_recall );
%SVM模型训练集评价指标
[J,~] = confusionmat(data_test_y,yfit_test_SVM);%产生测试集混淆矩阵
SVM_test_precise = J(1,1)/(J(1,1) + J(2,1));
SVM_test_recall = J(1,1)/(J(1,1) +J(1,2));
SVM_test_accurance=(J(1,1)+J(2,2))/(J(1,1)+J(2,1)+J(1,2)+J(2,2));%准确率
SVM_test_F1 = 2 * SVM_test_precise * SVM_test_recall...
    /(SVM_test_precise+ SVM_test_recall );
%SVM模型测试集评价指标
disp(['SVM模型训练集+验证集精确率为',num2str(SVM_train_precise)])
disp(['SVM模型训练集+验证集召回率为',num2str(SVM_train_recall)])
disp(['SVM模型训练集+验证集准确率为',num2str(SVM_train_accurance)])
disp(['SVM模型训练集+验证集F1值为',num2str(SVM_train_F1)])
disp(['SVM模型测试集精确率为',num2str(SVM_test_precise)])
disp(['SVM模型测试集召回率为',num2str(SVM_test_recall)])
disp(['SVM模型测试集准确率为',num2str(SVM_test_accurance)])
disp(['SVM模型测试集F1值为',num2str(SVM_test_F1)])
figure(7)
subplot(2,1,1)
cm = confusionchart(data_train_y,yfit_train_SVM);
xlabel('Target Class');
ylabel('Output Class');
title('SVM model confusion matrix(Training set)')
subplot(2,1,2)
cm2 = confusionchart(data_test_y,yfit_test_SVM);
xlabel('Target Class');
ylabel('Output Class');
title('SVM model confusion matrix(Test set)')
figure(2)%测试集图
AUC_SVM_test=plot_roc7(yfit_test_SVM,data_test_y);  
disp(['SVM回归测试集的AUC值为',num2str(AUC_SVM_test)]);
hold on
a=[0:1];
b=[0:1];
plot(a,b,'k--','LineWidth',1.5);
legend('KNN(AUC=0.9918)','logistic(AUC=0.7466)','Bagging(AUC=0.9378)', ...
    'BP(AUC=0.8930)',  'SVM(AUC=0.7974)','Subline(AUC=0.5)',...
    'Fontname','Times New Roman','FontSize',14,'Location','SouthEast');
legend('boxoff')
legend('boxoff')
legend('boxoff')
legend('boxoff')
legend('boxoff')
legend('boxoff')
figure(3)%训练集图
AUC_SVM_train=plot_roc8(yfit_train_SVM,data_train_y);  
disp(['SVM回归训练集+验证集的AUC值为',num2str(AUC_SVM_train)]); 
hold on
a=[0:1];
b=[0:1];
plot(a,b,'k--','LineWidth',1.5);
legend('KNN(AUC=0.9916)','logistic(AUC=0.7300)','Bagging(AUC=0.9357)', ...
     'BP(AUC=0.8892)',  'SVM(AUC=0.7810)','Subline(AUC=0.5)',...
    'Fontname','Times New Roman','FontSize',14,'Location','SouthEast');
legend('boxoff')
legend('boxoff')
legend('boxoff')
legend('boxoff')
legend('boxoff')
legend('boxoff')
%save model2022-6-23.mat
