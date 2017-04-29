%% this is the submitted version
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% readme %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if run, the current setup is to train 3 networks 
% with the top 3 hyper parameters saved in *.mat
% number of epoch is 30 and 19000/1000 for training/validation
% FYI, this could take a while
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% main program
% add data path
addpath Datasets/cifar-10-batches-mat/;
% load training data
trainingName = 'data_batch_1.mat';
[trainX, trainY, trainy] = LoadBatch(trainingName);
% load validate data
validName = 'data_batch_2.mat';
[validX, validY, validy] = LoadBatch(validName);
% load testing data
testingName = 'test_batch.mat';
[testX, testY, testy] = LoadBatch(testingName);

tvX = [trainX, validX];
tvY = [trainY, validY];
tvy = [trainy; validy];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % gradient check
% % uncomment this if needed
% % % structure of the NN
% nLabels = 10;
% nLayers = 2;
% nHiddenNodes(1) = 5;
% nHiddenNodes(nLayers) = nLabels;

% trainSize = 3;
% X = trainX(:,1:trainSize);
% Y = trainY(:,1:trainSize);
% y = trainy(1:trainSize);

% [paramCell] = ParamsInit(nHiddenNodes);

% [P, interScores, sRelated] = EvaluateClassifier(X, paramCell);

% lambda = 0;
% [J, Loss] = ComputeCost(X, Y, paramCell, lambda);
% [grad_W] = ComputeGradients(interScores, Y, P, sRelated, paramCell, lambda);

% % for checking purpose
% [ngrad_b, ngrad_W] = ComputeGradsNum(X, Y, paramCell(1:end,1), paramCell(1:end,2), lambda, 1e-5);
% for i = 1:nLayers
% 	check_W{i} = abs(ngrad_W{i} - grad_W{i,1}) ./ max(1e-5, abs(ngrad_W{i}) + abs(grad_W{i,1}));
% 	check_b{i} = abs(ngrad_b{i} - grad_W{i,2}) ./ max(1e-5, abs(ngrad_b{i}) + abs(grad_W{i,2}));
% end
% % end of gradient check
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% structure of the NN
nLabels = 10;
nLayers = 2;
nHiddenNodes(1) = 50;
nHiddenNodes(nLayers) = nLabels;

% set GDparams
% lambda = 1e-6;
% GDparams.n_epochs = 10;
GDparams.n_batch = 100;
% GDparams.eta = 5e-2;
GDparams.rho = 0.9;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% search for top 3 hyper parameters
% most are commented out
% only load the top 3 for later training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % coarse random search
% GDparams.n_epochs = 5;
% % initialize params
% nIteration = 77;
% savespace = zeros(nIteration,3);
% % weight
% [paramCell] = ParamsInit(nHiddenNodes);
% eta_min = -2;
% eta_max = -1;
% lambda_min = -7;
% lambda_max = -5;
% coarse_range = [eta_min, eta_max, lambda_min, lambda_max];
% for iter = 1:nIteration
% 	% hyper params
	% [GDparams.eta, lambda] = hyperParamsInit(coarse_range);

% 	% MiniBatchGD
% 	[learnedCell, mean_tvX, outputResult] = MiniBatchGD(tvX, tvY, tvy, GDparams, paramCell, lambda);

% 	% save things
% 	savespace(iter,:) = [GDparams.eta, lambda, outputResult];
% end
% saveMat = savespace;

% % coarse to fine
% load('coarse.mat');
% % do some plots for inspection
% nTopPicks = 23; % this is decided based on the plots

% [~, I] = sort(saveMat(:,3));
% I = flipud(I);
% goodInd = I(1:nTopPicks);
% goodHyperParams = saveMat(goodInd,:);

% % accuracy vs eta
% figure;
% scatter(saveMat(:,1),saveMat(:,3));
% hold on;
% scatter(saveMat(goodInd,1),saveMat(goodInd,3));

% % accuracy vs lambda
% figure;
% scatter(saveMat(:,2),saveMat(:,3));
% hold on;
% scatter(saveMat(goodInd,2),saveMat(goodInd,3));

% f_eta_max = max(goodHyperParams(:,1));
% f_eta_min = min(goodHyperParams(:,1));
% f_lambda_max = max(goodHyperParams(:,2));
% f_lambda_min = min(goodHyperParams(:,2));
% fine_range = [f_eta_min, f_eta_max, f_lambda_min, f_lambda_max];
% fine_range = log10(fine_range);

% % fine random search
% GDparams.n_epochs = 10;

% [paramCell] = ParamsInit(nHiddenNodes);
% savespace = zeros(nTopPicks,3);

% for iter = 1:nTopPicks
% 	[GDparams.eta, lambda] = hyperParamsInit(coarse_range);

% 	% MiniBatchGD
% 	[learnedCell, mean_tvX, outputResult] = MiniBatchGD(tvX, tvY, tvy, GDparams, paramCell, lambda);

% 	% save things
% 	savespace(iter,:) = [GDparams.eta, lambda, outputResult];
% end
% betterFineMat = savespace;

% process the fine results
load('betterfine.mat');
top3 = 3;
[~, I] = sort(betterFineMat(:,3));
I = flipud(I);
betterInd = I(1:top3);
top3HyperParams = betterFineMat(betterInd,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Training with top 3 hyper parameters
GDparams.n_epochs = 10;
[paramCell] = ParamsInit(nHiddenNodes);
for i = 1:1
	GDparams.eta = top3HyperParams(i,1);
	lambda = top3HyperParams(i,2);
	[learnedCell, mean_tvX, outputResult] = MiniBatchGD(tvX, tvY, tvy, GDparams, paramCell, lambda);
	testX = testX - repmat(mean_tvX, 1, size(testX,2));
	[acc] = ComputeAccuracy(testX, testy, learnedCell)
end

%% end of main program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Given Framework
% function #1
%% LoadBatch: 	a function that reads in the data from a CIFAR-10 batch file and
%				returns the image and label data in separate files
function [X, Y, y] = LoadBatch(filename)
	nLabels = 10;
	A = load(filename);
	data = A.data;
	y = double(A.labels + 1);
	% shift the labels from 0-9 to 1-10
	Y = zeros(nLabels, length(y));
	Y(sub2ind(size(Y), y', 1:length(y))) = 1;
	X = double(data') / 255;
end

% function #2
%% ParamsInit: initialize the parameters of the model W and b
function [paramCell] = ParamsInit(structParams)
	nHiddenNodes = structParams;
	nLayers = length(nHiddenNodes);
	paramCell = cell(nLayers,2);
	initMean = 0;
	initStd = 0.001;
	singlePic = 32 * 32 * 3;
	rng(400); % this is from the instruction

	for i = 1:nLayers
		if i == 1
			paramCell{i,1} = initMean + initStd * randn(nHiddenNodes(i), singlePic);
		else
			paramCell{i,1} = initMean + initStd * randn(nHiddenNodes(i), nHiddenNodes(i - 1));
		end
		paramCell{i,2} = zeros(nHiddenNodes(i), 1);
	end
end

%% hyperParamsInit: initial eta and lambda randomly.
function [eta, lambda] = hyperParamsInit(ranges)
	eta_min = ranges(1);
	eta_max = ranges(2);
	lambda_min = ranges(3);
	lambda_max = ranges(4);

	rng('shuffle'); % change with current time
	eta_rand = eta_min + (eta_max - eta_min) * rand(1,1);
	eta = 10 ^ eta_rand;
	lambda_rand = lambda_min + (lambda_max - lambda_min) * rand(1,1);
	lambda = 10 ^ lambda_rand;
end


% function #3
%% EvaluateClassifier:  a function that evaluates the network function
%						this is the forward pass
function [P, h, outputs] = EvaluateClassifier(X, W)
	% this W is actually paramCell
	nLayers = size(W,1);
	h = cell(1,nLayers);
	h{1,1} = X;
	for i = 1:nLayers - 1
		X = h{1,i};
		s{1,i} = W{i,1} * X + repmat(W{i,2},1,size(X,2));
		mu{1,i} = s{1,i};
		sigmav{1,i} = s{1,i};
		s_hat{1,i} = BatchNormalize(s{1,i},mu{1,i},sigmav{1,i});
		h{1,i + 1} = max(0,s_hat{1,i});
	end
	X = h{1,nLayers};
	s{1,nLayers} = W{nLayers,1} * X + repmat(W{nLayers,2},1,size(X,2));
	P = softmax(s{1,nLayers});

	% stuff in outputs
	outputs.s = s;
	outputs.s_hat = s_hat;
	outputs.mu = mu;
	outputs.sigmav = sigmav;
end

% function #4
%% ComputeCost: a function that computes the cost function
function [J, Loss] = ComputeCost(X, Y, paramCell, lambda)
	P = EvaluateClassifier(X, paramCell);
	Loss = sum(-log(sum(Y .* P))) / size(X,2);
	L2 = 0;
	for i = 1:size(paramCell,1)
		W = paramCell{i,1};
		L2 = L2 + sum(sum(W .* W));
	end
	J = Loss + lambda * L2;
end

% function #5
%% ComputeAccuracy: a function that computes the accuracy of the network's predictions
function [acc] = ComputeAccuracy(X, y, paramCell)
	P = EvaluateClassifier(X, paramCell);
	[~, prediction] = max(P);
	acc = 1 - nnz(prediction' - y) / size(X,2);
end

% function #6
%% ComputeGradients: compute the gradients of the cost function w.r.t. W and b
function [LW] = ComputeGradients(h, Y, P, inputs, W, lambda)
	% LW is not loss w.r.t W; it is the gradCell
	% unpack inputs
	s = inputs.s;
	s_hat = inputs.s_hat;
	mu = inputs.mu;
	sigmav = inputs.sigmav;
	% this W is actually paramCell
	X = h{1,1};
	batchSize = size(X,2);
	nLayers = size(W,1);

	LW = weightZero(W);
	g = zeros(size(Y'));

	% check that the first "layer" of h is X
	if ~isequal(X,h{1,1})
		disp('X and h are not matching');
		return;
	end

	% chech the # of layers in h
	if size(h,2) ~= nLayers
		disp('h is wrong with # of layers');
		return;
	end

	% chech the # of layers in s
	if size(s,2) ~= nLayers
		disp('s is wrong with # of layers');
		return;
	end

	for i = 1:batchSize
		% for each pic
		g(i,:) = - Y(:,i)' / (Y(:,i)' * P(:,i)) * (diag(P(:,i)) - P(:,i) * P(:,i)');
	end
	% last layer
	LW{nLayers,2} = mean(g)';
	LW{nLayers,1} = getSumVecProductMat(g, h{1,nLayers}) + 2 * lambda * W{nLayers,1};
	g = g * W{nLayers,1};
	g = rowVec_x_diagColVec(g, s_hat{1,nLayers - 1});

	% previous layers
	% if nLayers == 1, we have some problems
	% if nLayers == 2, we don't need this loop
	% if nLayers => 3, this loop is needed
	for idxLayer = nLayers - 1 : 2
		if nLayers < 3
			break;
		end
		g = BatchNormBackPass(g,s{1,idxLayer},mu{1,idxLayer},sigmav{1,idxLayer});
		LW{idxLayer,2} = mean(g)';
		LW{idxLayer,1} = getSumVecProductMat(g, h{1,idxLayer}) + 2 * lambda * W{idxLayer,1};
		g = g * W{idxLayer,1};
		g = rowVec_x_diagColVec(g, s_hat{1,idxLayer - 1});
	end

	% first layer
	g = BatchNormBackPass(g,s{1,1},mu{1,1},sigmav{1,1});
	LW{1,2} = mean(g)';
	LW{1,1} = getSumVecProductMat(g, h{1,1}) + 2 * lambda * W{1,1};
end

% function #7
%% MiniBatchGD: function description
function [starCell, mean_X, outputResult] = MiniBatchGD(X, Y, y, GDparams, paramCell, lambda)
	% half training half validation
	portion = 1/2;
	nDataSet = size(X,2);
	Xv = X(:,nDataSet * portion + 1 : end);
	Yv = Y(:,nDataSet * portion + 1 : end);
	yv = y(nDataSet * portion + 1 : end,1);
	X = X(:,1:nDataSet * portion);
	Y = Y(:,1:nDataSet * portion);
	% y = y(1:nDataSet * portion,1);

	% compute training mean and recenter
	mean_X = mean(X,2);
	X = X - repmat(mean_X, 1, size(X,2));
	Xv = Xv - repmat(mean_X, 1, size(Xv,2));

	n_batch = GDparams.n_batch;
	n_epochs = GDparams.n_epochs;
	eta = GDparams.eta;
	rho = GDparams.rho;
	nPics = size(X,2);
	starCell = paramCell;
	accValid = 0;

	% momentum term
	nLayers = size(paramCell,1);
	momentumV = weightZero(paramCell);
	
	% for each epoch
	for i = 1:n_epochs
		% for each batch
		for j = 1:nPics / n_batch
			jstart = (j - 1) * n_batch + 1;
			jend = j * n_batch;
			inds = jstart:jend;
			Xbatch = X(:,inds);
			Ybatch = Y(:,inds);
			[Pbatch, interScores, sRelated] = EvaluateClassifier(Xbatch, starCell);

			[delta_Cell] = ComputeGradients(interScores, Ybatch, Pbatch, sRelated, starCell, lambda);
			for idxLayer = 1:nLayers
				for idxParams = 1:2
					momentumV{idxLayer,idxParams} = rho * momentumV{idxLayer,idxParams} + eta * delta_Cell{idxLayer,idxParams};
					starCell{idxLayer,idxParams} = starCell{idxLayer,idxParams} - momentumV{idxLayer,idxParams};
				end
			end
		end
		% record: decrease of cost
		[J(i),L(i)] = ComputeCost(X, Y, starCell, lambda);
		[Jv(i),Lv(i)] = ComputeCost(Xv, Yv, starCell, lambda);
		accValid(i) = ComputeAccuracy(Xv, yv, starCell);

		% check Jv for overfitting or too high eta
		if Jv(i) > 3 * Jv(1)
			break;
		end
		% decay learning rate
		eta = eta * 0.95;
	end
	% display: decrease of cost
	figure;
	hold on;
	plot(J, 'r-x');
	plot(Jv, 'b-o');
	legend('training cost','validation cost');
	% % display: decrease of loss
	% figure;
	% hold on;
	% plot(L, 'r-x');
	% plot(Lv, 'b-o');
	% legend('training loss','validation loss');

	outputResult = max(accValid);
end

%% BatchNormalize: function description
function [s_hat] = BatchNormalize(s, mu, sigmav)
	epsilon = 1e-7;
	% without BN
	s_hat = s;
	% s_hat = sqrt(diag(sigmav + epsilon)) * (s - mu);
end

%% BatchNormBackPass: function description
function [gOut] = BatchNormBackPass(gIn,s,mu,sigmav)
	% without BN
	gOut = gIn;
end



%% End of Given Framework

%% given functions
function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

[c, ~] = ComputeCost(X, Y, [W, b], lambda);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        [c2, ~] = ComputeCost(X, Y, [W, b_try], lambda);
        grad_b{j}(i) = (c2-c) / h;
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})   
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        [c2, ~] = ComputeCost(X, Y, [W_try, b], lambda);
        
        grad_W{j}(i) = (c2-c) / h;
    end
end
end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        [c1, ~] = ComputeCost(X, Y, [W, b_try], lambda);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        [c2, ~] = ComputeCost(X, Y, [W, b_try], lambda);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        [c1, ~] = ComputeCost(X, Y, [W_try, b], lambda);
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        [c2, ~] = ComputeCost(X, Y, [W_try, b], lambda);
    
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% little helper functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% weightZero: initialize a cell array that has the same dimension as the weight cell
% 				but all the entries are zeros
function [zeroMat] = weightZero(weightCell)
	nLayers = size(weightCell,1);
	zeroMat = cell(nLayers, 2);
	for idxLayer = 1:nLayers
		for idxParams = 1:2
			zeroMat{idxLayer,idxParams} = zeros(size(weightCell{idxLayer,idxParams}));
		end
	end
end

%% getSumVecProductMat: a vectorization of a calculation, not!
%	input: 1 row vec vertcat mat (nPics x nLabels) and 1 col vec horzcat mat (singlePic x nPics)
%	output: a nLabels x singlePic x nPics mat where each "layer" is (row vec)' * (col vec)'
%	this comment is out of date
function [outputs] = getSumVecProductMat(g, X)
	[nPicsg, nLabels] = size(g);
	[singlePic, nPicsx] = size(X);
	outputs = zeros(nLabels,singlePic);

	if nPicsg ~= nPicsx
		disp('the dimension is wrong');
		return;
	end

	for i = 1:nPicsx
		outputs = outputs + g(i,:)' * X(:,i)';
	end
	outputs = outputs / nPicsx;
end

%% rowVec_x_diagColVec: a vectorization of a calculation, not!
%	can be done in vectorization form (mat product)
%	input: 1 row vec vertcat mat (nPics x nLabels) and 1 col vec horzcat mat (singlePic x nPics)
%	output: 1 mat that's the same size of 1st input
function [outputs] = rowVec_x_diagColVec(g, s)
	[nPicsg, nLabels] = size(g);
	[singlePic, nPicsx] = size(s);
	outputs = zeros(nPicsg, nLabels);

	if nPicsg ~= nPicsx
		disp('the dimension is wrong');
		return;
	end

	if nLabels ~= singlePic
		disp('the g and s_hat are not matching');
		return;
	end

	for i = 1:nPicsx
		outputs(i,:) = g(i,:) * diag(s(:,i) > 0);
	end
end