function [param] = set_params_M3

g = [4.8 0.88];  % parameters of the gamma prior
param(1).name = 'inverse temperature small';
param(1).logpdf = @(x) sum(log(gampdf(x,g(1),g(2))));  % log density function for prior
param(1).lb = 0;   % lower bound
param(1).ub = 20;  % upper bound 

param(2).name = 'inverse temperature large';
param(2).logpdf = @(x) sum(log(gampdf(x,g(1),g(2))));  % log density function for prior
param(2).lb = 0;   % lower bound
param(2).ub = 20;  % upper bound % - was increased to 40 following 2020 reviewer feedback (now set back to 20)

b = [2 2]; % parameters of the beta prior
param(3).name = 'learning rate';
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

mu = 0; sd = 1;   % normal priors (with 0,1)
param(6).name = 'choice stickiness';
param(6).logpdf = @(x) sum(log(normpdf(x,mu,sd)));
param(6).lb = -20;
param(6).ub = 20;

param(7).name = 'response stickiness';
param(7).logpdf = @(x) sum(log(normpdf(x,mu,sd)));
param(7).lb = -20;
param(7).ub = 20;

end
