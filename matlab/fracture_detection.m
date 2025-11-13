clc; clear; close all;

% Heuristic fracture indication using Hough lines

imgPath = fullfile('..', 'images', 'foot_xray_example.jpg');
I = imread(imgPath);

if size(I, 3) == 3
    I = rgb2gray(I);
end

I_eq = histeq(I);                % contrast enhancement
BW   = edge(I_eq, 'canny');      % edges

[H, theta, rho] = hough(BW);
P     = houghpeaks(H, 20, 'Threshold', ceil(0.3 * max(H(:))));
lines = houghlines(BW, theta, rho, P, 'FillGap', 10, 'MinLength', 15);

% Compute angles and lengths of line segments
angles = zeros(length(lines), 1);
lengths = zeros(length(lines), 1);

for k = 1:length(lines)
    p1 = lines(k).point1;
    p2 = lines(k).point2;
    dx = p2(1) - p1(1);
    dy = p2(2) - p1(2);
    angles(k)  = atan2d(dy, dx);          % degrees
    lengths(k) = norm(p2 - p1);
end

% Estimate main bone orientation from long lines
[~, idxSorted] = sort(lengths, 'descend');
numMain = min(5, numel(angles));
mainAngles = angles(idxSorted(1:numMain));
mainAngle  = median(mainAngles);

angleThresh   = 15;                     % deg from main axis
lengthThresh  = 0.6 * max(lengths);     % suspicious if much shorter

figure, imshow(I_eq), title('Fracture Candidates (Heuristic)'), hold on;

for k = 1:length(lines)
    p1 = lines(k).point1;
    p2 = lines(k).point2;
    
    isOffAngle = abs(angles(k) - mainAngle) > angleThresh;
    isShort    = lengths(k) < lengthThresh;
    
    if isOffAngle && isShort
        col = 'red';   % possible fracture segment
        lw  = 3;
    else
        col = 'green'; % normal cortical segment
        lw  = 1.5;
    end
    
    plot([p1(1) p2(1)], [p1(2) p2(2)], 'Color', col, 'LineWidth', lw);
end

legend({'Green: cortical lines', 'Red: possible fracture lines'});
