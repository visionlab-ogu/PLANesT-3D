function [YY,loss] = compute_TSNE(ptCloud,tsne_perp)

XYZ = ptCloud.Location;


[YY,loss] = tsne(XYZ,'perplexity',tsne_perp);
sY = size(YY,1);
sP = size(XYZ,1);
if sY<sP
    YY = [YY ; repmat(YY(end,:),sP-sY,1)];
end