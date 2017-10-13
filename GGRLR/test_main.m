% Graph-Guided Regularized Logistic Regression
clear

%funfcn_stoc  = {@STOC_ADMM,@RDA_ADMM,@OPG_ADMM,@Fast_SADMM,@Ada_SADMMdiag,@Ada_SADMMfull,@SPDHG_SC1,@SPDHG_SC2};
funfcn_stoc  = {@SPDHG_SC1,@SPDHG_SC2};
idx_method   = 1;

nruns_stoc   = 1;
n_epochs     = 10;

opts  = [];
data_path = '../data/stock_test_data_20170901.csv';
data = csvread(data_path,1,1);
samples = data(:,1:end-1)';
labels = data(:,end);

% graphical matrix generation
S  = cov(samples');
rho = 0.005; % weighting parameter and the parameters can be tuned
opts.mxitr = 500; opts.mu0 = 1e-1;  opts.muf = 1e-3; opts.rmu = 1/4;
opts.tol_gap = 1e-1; opts.tol_frel = 1e-7; opts.tol_Xrel = 1e-7; opts.tol_Yrel = 1e-7;
opts.numDG = 10; opts.record = 1;opts.sigma = 1e-10;
out = SICS_ALM(S,rho,opts);
X = out.X; X(abs(X) > 2.5e-3) = 1; X(abs(X) < 2.5e-3) = 0; F = -tril(X,-1) + triu(X,1);

%Samples divide
ratio_train  = 0.8;
idx_all      = 1:length(labels);
idx_train    = idx_all(rand(1,length(labels),1)<ratio_train);
idx_test     = setdiff(idx_all,idx_train);

if sum(sum(samples==0))/(size(samples,1)*size(samples,2)) > 0.8
    s_train      = sparse(double(samples(:,idx_train)));
    s_test       = sparse(double(samples(:,idx_test)));
else
    s_train      = full(double(samples(:,idx_train)));
    s_test       = full(double(samples(:,idx_test)));
end
l_train      = labels(idx_train);
l_test       = labels(idx_test);

%Parameters setup
%Parameters of model
opts.mu      = 1e-5; %parameter of graph-guided term
%Parameters of algorithms
opts.F       = F;     %The graph structure
opts.beta    = 1;     %parameter of STOC-ADMM to balance augmented lagrange term
opts.gamma   = 1e-2;  %Regularized Logistic Regression term
opts.epochs  = n_epochs;     %maximum effective passes
opts.max_it  = length(idx_train)*opts.epochs;
opts.checkp  = 0.01;  %save the solution very 1% of data processed
opts.checki  = floor(opts.max_it * opts.checkp);
opts.a       = 1;     %parameter of Ada_SADMM
opts.s       = 5e-5;  %parameter of SPDHG to update y;

tempVar = zeros(size(samples,2),1);
for idx_s = 1:size(samples,2)
    tempVar(idx_s) = samples(:,idx_s)'*samples(:,idx_s);
end
opts.L = 0.25*max(tempVar);
eigFTF = eigs(opts.F'*opts.F, 1);
if eigFTF == 0
    eigFTF = max(max(opts.F));
    eigFTF = eigFTF*eigFTF;
end
opts.L1 = opts.beta*eigFTF + opts.L;
opts.L2 = sqrt(max(2*opts.L*opts.L+eigFTF, 2*eigFTF));
opts.L3 = max(8*opts.beta*eigFTF, sqrt(8*opts.L*opts.L + opts.beta*eigFTF))+opts.gamma;
opts.eta = max(opts.beta*eigFTF, 100);%parameter of RDA-ADMM and OPG-ADM

%stoc methods
stat_data    = [];
trace_accuracy = [];
trace_test_loss= [];
trace_obj_val  = [];
trace_passes   = [];
trace_time     = [];
num_train      = length(idx_train);
num_runs       = nruns_stoc;
%Trainning
t = cputime;
outputs       = funfcn_stoc{idx_method}(s_train, l_train, opts);
time_solving  = cputime - t;
time_per_iter = time_solving/outputs.iter;
x          = outputs.x;
accuracy = get_accuracy(s_test,l_test,x)

