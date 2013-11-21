load mnist_uint8;

data = double(data) / 255;
test  = double(test)  / 255;
datalabels = double(datalabels);
testlabels  = double(testlabels);

% normalize
%[train_x , test_x] = normalize_new(train_x,test_x);

%% ex1 vanilla neural net
rand('state',0)
nn = nnsetup([784 200 10]);  
opts.numepochs =  20; 
opts.batchsize = 100; 

nn = nntrain(nn, data, datalabels, opts);

[error, bad] = nntest(nn, test, testlabels);

assert(error < 0.08, 'Too big error');