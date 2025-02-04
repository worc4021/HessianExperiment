clearvars;

problem = struct();
problem.variableInfo = struct();
problem.variableInfo.lb = [-2,-2];
problem.variableInfo.ub = [2,2];  % Upper bound on the variables.
problem.variableInfo.cl = zeros(0,1);
problem.variableInfo.cu = zeros(0,1);
problem.variableInfo.x0 = rand(2,1);  % The starting point.
% % Initialize the dual point.

problem.ipopt = struct; 
problem.ipopt.hessian_approximation = "limited-memory";
problem.ipopt.print_level = 5;
problem.ipopt.output_file = 'debug.ipoptout';
problem.ipopt.file_print_level = 12;

% The callback functions.

model = TestModelHarness(problem.variableInfo.x0);

problem.funcs.objective         = @model.objective;
problem.funcs.constraints       = @model.constraints;
problem.funcs.gradient          = @model.gradient;
problem.funcs.jacobian          = @model.jacobian;
problem.funcs.jacobianstructure = @(~)sparse(0,2);
problem.funcs.hessian           = @(x,sigma,lambda)sparse(tril(model.hessian(x,sigma,lambda)));
problem.funcs.hessianstructure  = @()sparse(tril(ones(2)));
problem.funcs.intermediate      = @model.intermediateCallback;


[X,Y] = meshgrid(linspace(problem.variableInfo.lb(1),problem.variableInfo.ub(1),237),...
                 linspace(problem.variableInfo.lb(2),problem.variableInfo.ub(2),313));
F = X*0;
F(:) = TestModelCore.modelfun([X(:),Y(:)]');

figure(1)
clf
contourf(X,Y,F,3*logspace(0,5,13));
model.h = animatedline(problem.variableInfo.x0(1),problem.variableInfo.x0(2),'Marker','o','Color','r');

[x, info] = ipopt(problem);

[xi,yi]=getpoints(model.h);

dxi = xi*0;
dyi = xi*0;
for i = 1:numel(xi)
    g = model.gradient([xi(i);yi(i)]);
    dxi(i) = g(1);
    dyi(i) = g(2);
end
hold("on")
qq=quiver(xi,yi,-dxi,-dyi);
qq.Color=[0,1,0];
hold("off")
% 
% problem.ipopt.hessian_approximation = "exact";
% 
% 
% model = TestModelHarness(problem.variableInfo.x0);
% model.h = animatedline(problem.variableInfo.x0(1),problem.variableInfo.x0(2),'Marker','o','Color','g');
% 
% problem.funcs.intermediate      = @model.intermediateCallback;
% problem.funcs.gradient          = @model.gradient;
% problem.funcs.objective         = @model.objective;
% problem.funcs.hessian           = @(x,sigma,lambda)sparse(tril(model.hessian(x,sigma,lambda)));
% 
% [x, info] = ipopt(problem);