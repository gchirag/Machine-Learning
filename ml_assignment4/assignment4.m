%Part-1%

clear;

% data = loadMNISTImages('train-images.idx3-ubyte');
% %data = data(:,1:5000);
% data = data';
% labels = loadMNISTLabels('train-labels.idx1-ubyte');
% %labels = labels(1:5000);
% 
% test = loadMNISTImages('t10k-images.idx3-ubyte');
% %test = test(:,1:300);
% test = test';
% 
% testlabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
% %testlabels = testlabels(1:300);

% %Multi layer neural network%
% load mnist_uint8;
% 
% data = double(data) / 255;
% test  = double(test)  / 255;
% datalabels = double(datalabels);
% testlabels  = double(testlabels);
% 
% % normalize
% %[train_x , test_x] = normalize_new(train_x,test_x);
% 
% %% ex1 vanilla neural net
% rand('state',0)
% nn = nnsetup([784 200 10]);  
% opts.numepochs =  20; 
% opts.batchsize = 100; 
% 
% nn = nntrain(nn, data, datalabels, opts);
% 
% [error, bad] = nntest(nn, test, testlabels);
% 
% assert(error < 0.08, 'Too big error');
%SVM%

% [label_svm,data_svm] = libsvmread('mnist.scale.libsvm');
% data_svm = data_svm(1:10000,:);
% label_svm = label_svm(1:10000);
% 
% [testlabels_svm,test_svm] = libsvmread('mnist.scale.t.libsvm');
% test_svm = test_svm(1:10000,:);
% testlabels_svm = testlabels_svm(1:10000);

%model = svmtrain(label_svm,data_svm);%,'-s 0 -t 2 -v 10');
%[predictions_svm,accuracy_svm,probestimates] = svmpredict(testlabels_svm,test_svm,model);%,'-v 10');


% %KNN%
%  prediction = knnclassify(test,data,labels,3);
%  result = prediction - testlabels;
%  
%  error = (10000-sum(result (:) == 0))/100;
%  
%  error
 
%Part-2%
% covar1 = [0.5 .05; .05 .5];
% covar2 = [1 0; 0 1];
% %temp=mvnrnd([-3 0],covar2,200);
% %mle(temp)
% n = 100;
% v = 10:10:1000;
% error2 = zeros(100,1);
% covarerr = zeros(100,1);
% for i = 1:100,
%     n = v(i);
%     temp = mvnrnd([0 0],covar1,n);
%     covar = [0 0; 0 0];
%     mean = sum(temp,1)/n;
%     for j = 1:n,
%         covar = covar + (temp(j)-mean)*((temp(j)-mean)')/n;
%     end
%     error2(i) = norm(mean);
%     covarerr(i) = sumsqr(covar-covar1);
% end
% 
% figure(1);
% plot(v,covarerr);
% figure(2);
% plot(v,error2);
% 
% 
% for i = 1:100,
%     n = v(i);
%     temp = mvnrnd([0 0],covar2,n);
%     covar = [0 0; 0 0];
%     mean = sum(temp,1)/n;
%     for j = 1:n,
%         covar = covar + (temp(j)-mean)*((temp(j)-mean)')/n;
%     end
%     error2(i) = norm(mean);
%     covarerr(i) = sumsqr(covar-covar2);
% end
% 
% figure(3);
% plot(v,covarerr);
% figure(4);
% plot(v,error2);
% 

%Part-3%
mean_part3 = [1;2;3];
contribution_part3_a = [0.2 0.3 0.5];
contribution_part3_b = [0.7 0.2 0.1];
covar_part3 = cat(3,1,1,1);
obj1 = gmdistribution(mean_part3,covar_part3,contribution_part3_a);
obj2 = gmdistribution(mean_part3,covar_part3,contribution_part3_b);

samples1 = random(obj1,40000);
answer1 = gmdistribution.fit(samples1,3);
answer1.mu
answer1.Sigma
answer1.PComponents

samples2 = random(obj2,40000);
answer2 = gmdistribution.fit(samples2,3);
answer2.mu
answer2.Sigma
answer2.PComponents

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

