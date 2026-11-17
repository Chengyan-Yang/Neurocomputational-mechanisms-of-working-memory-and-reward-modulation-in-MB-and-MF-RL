function [param] = set_params_M6

g = [4.8 0.88];  % parameters of the gamma prior
param(1).name = 'inverse temperature small';
param(1).logpdf = @(x) sum(log(gampdf(x,g(1),g(2))));  % log density function for prior
param(1).lb = 0;   % lower bound
param(1).ub = 20;  % upper bound

b = [2 2]; % parameters of the beta prior
param(2).name = 'learning rate small';
param(2).logpdf = @(x) sum(log(betapdf(x,b(1),b(2))));
param(2).lb = 0;
param(2).ub = 1;

param(3).name = 'learning rate large';
param(3).logpdf = @(x) sum(log(betapdf(x,b(1),b(2))));
param(3).lb = 0;
param(3).ub = 1;

param(4).name = 'eligibility trace decay';
param(4).logpdf = @(x) sum(log(betapdf(x,b(1),b(2))));
param(4).lb = 0;
param(4).ub = 1;

param(5).name = 'mixing weight small';
param(5).logpdf = @(x) sum(log(betapdf(x,b(1),b(2))));
param(5).lb = 0;
param(5).ub = 1;

param(6).name = 'mixing weight large';
param(6).logpdf = @(x) sum(log(betapdf(x,b(1),b(2))));
param(6).lb = 0;
param(6).ub = 1;

mu = 0; sd = 1;   % normal priors (with 0,1)
param(7).name = 'choice stickiness';
param(7).logpdf = @(x) sum(log(normpdf(x,mu,sd)));
param(7).lb = -20;
param(7).ub = 20;

param(8).name = 'response stickiness';
param(8).logpdf = @(x) sum(log(normpdf(x,mu,sd)));
param(8).lb = -20;
param(8).ub = 20;

end
