classdef TestModelHarness < handle
    properties (SetAccess = private)
        previousCalls (:,1) TestModelCore
        mode (1,1) string {mustBeMember(mode,["bfgs","none"])} = "none"
        threshold = 0.2;
    end

    properties 
        h (:,1) matlab.graphics.animation.AnimatedLine
    end
    
    methods
        function obj = TestModelHarness(x0)
            arguments
                x0 (2,1) double
            end
            obj.previousCalls = TestModelCore(x0);
            obj.previousCalls.B = eye(2);
        end
        function fVal = objective(obj,x)
            if obj.isnewx(x)
                obj.update(x);
            end
            fVal = obj.previousCalls(end).f;
        end
        
        function fGrad = gradient(obj,x)
            if obj.isnewx(x)
                obj.update(x);
            end
            fGrad = obj.previousCalls(end).g;
        end

        function bContinue = intermediateCallback(obj,s)
            if ~isempty(obj.h)
                addpoints(obj.h,s.primals(1),s.primals(2));
            end
            if isequal(obj.previousCalls(1).B,eye(2)) && numel(obj.previousCalls)>1
                s = obj.previousCalls(2).x - obj.previousCalls(1).x;
                y = obj.previousCalls(2).g - obj.previousCalls(1).g;
                obj.previousCalls(1).B = eye(2)*(y'*y)/(s'*y);
            end

            if numel(obj.previousCalls)>1
                obj.previousCalls(1) = obj.previousCalls(2);
            end            
            bContinue = true;
        end

        function B = hessian(obj,x,~,~)
            if obj.isnewx(x)
                obj.update(x);
            end
            B = obj.previousCalls(end).B;
        end

    end
    methods (Hidden)
        function B = bfgs(obj)
            s = obj.previousCalls(2).x - obj.previousCalls(1).x;
            y = obj.previousCalls(2).g - obj.previousCalls(1).g;
            Bs = obj.previousCalls(1).B*s;
            theta = 1;
            if s'*y < obj.threshold*s'*Bs
                theta = (1-obj.threshold)*s'*Bs/(s'*Bs-s'*y);
            end
            r = theta*y + (1-theta)*Bs;

            Bs = Bs/sqrt(s'*Bs);
            r = r/sqrt(s'*r);
            B = obj.previousCalls(1).B + r*r' - Bs*Bs';
        end

        function update(obj,x)
            obj.previousCalls(2) = TestModelCore(x);
            obj.previousCalls(2).B = obj.bfgs();
        end
        function b = isnewx(obj,x)
            b = norm(x-obj.previousCalls(end).x)>eps;
        end
    end
end

