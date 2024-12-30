clearvars

for mode = ["bfgs"]%"ldfp","dfp","sr1","lbfgs","autodiff",
    
    tb = parquetread("..\out\build\windows-intel-release-config\" + mode + ".parquet",'VariableNamingRule','preserve');
    
    tb.objectiveValue = categorical(round(tb.fVal));
    
    xValues = unique(tb.x0);
    yValues = unique(tb.y0);
    
    categories = unique(round(tb.fVal(ismember(tb.status,[0,1]))));
    cm = lines(numel(categories)+1);
    
    N = 1:(numel(categories)+1);
    
    [X,Y] = meshgrid(xValues,yValues);
    C = nan(size(X));
    I = nan(size(X));
    miniter = min(tb.iter);
    maxiter = max(tb.iter);
    for i = 1:numel(xValues)
        for j = 1:numel(yValues)
            if ismember(round(tb.fVal(tb.x0 == xValues(i) & tb.y0 == yValues(j))),categories)
                C(j,i) = N(round(tb.fVal(tb.x0 == xValues(i) & tb.y0 == yValues(j)))==categories);
            else
                C(j,i) = N(end);
            end
            I(j,i) = round(double(tb.iter(tb.x0 == xValues(i) & tb.y0 == yValues(j))-miniter)/double(maxiter-miniter)*255)+1;
        end
    end
    
    
    figure(1)
    clf
    imshow(C,cm)
    exportgraphics(gca,mode + "_solution.png","Resolution",600)
    
    clf
    imshow(I,parula);
    exportgraphics(gca,mode + "_iterations.png","Resolution",600)
end
