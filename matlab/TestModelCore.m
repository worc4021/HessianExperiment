classdef TestModelCore
    
    properties (SetAccess = private)
        x (2,1) double
        f (1,1) double
        g (2,1) double 
    end

    properties 
        B (2,2) double
    end
    
    methods
        function obj = TestModelCore(x)
            arguments
                x (2,1) double
            end
            obj.x = x;
            F = obj.modelfun(x+eye(2)*1i*1e-100);
            obj.f = real(F(1));
            obj.g = imag(F)*1e100;
        end

        function H = Hessian(obj,hHessian)
            arguments
                obj (1,1) TestModelCore
                hHessian (1,1) double {mustBePositive} = 1e-6
            end
            X = obj.x+[eye(2),zeros(2,1),-eye(2)]*hHessian;
            F = obj.modelfun(X);
            gPlus = F(:,1:2);
            gMinus = F(:,4:5);
            gNeutral = F(:,3);
            H = diag((gPlus-2*gNeutral+gMinus)/hHessian^2) + diag((gPlus(1)-gMinus(2))/(2*hHessian),1)+diag((gPlus(2)-gMinus(1))/(2*hHessian),-1);
        end
    end

    methods (Static, Hidden)
        function fVal = modelfun(var)
            x = var(1,:,:);
            y = var(2,:,:);
            fVal = (1+(x+y+1).^2.*(19-14*x+3*x.^2-14*y+6*x.*y+3*y.^2)).*(30+(2*x-3*y).^2.*(18-32*x+12*x.^2+48*y-36*x.*y+27*y.^2));
        end
    end
end

