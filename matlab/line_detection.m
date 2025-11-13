clc; clear; close all;

% Basic line detection using Hough Transform (original mini-project style)

imgPath = fullfile('..', 'images', 'foot_xray_example.jpg'); % relative path
I = imread(imgPath);

if size(I, 3) == 3
    I = rgb2gray(I);
end

% Edge detection
BW = edge(I, 'canny');

figure, imshow(BW), title('Canny Edge Map');

% Hough transform
[H, theta, rho] = hough(BW);

figure
imshow(imadjust(rescale(H)), [], ...
       'XData', theta, 'YData', rho, ...
       'InitialMagnification', 'fit');
xlabel('\theta (degrees)');
ylabel('\rho');
title('Hough Transform');
axis on; axis normal; hold on; colormap(gca, hot);

P = houghpeaks(H, 5, 'Threshold', ceil(0.3 * max(H(:))));
x = theta(P(:, 2));
y = rho(P(:, 1));
plot(x, y, 's', 'Color', 'black');

lines = houghlines(BW, theta, rho, P, 'FillGap', 5, 'MinLength', 7);

figure, imshow(I), title('Detected Lines'), hold on;
max_len = 0;
for k = 1:length(lines)
    xy = [lines(k).point1; lines(k).point2];
    plot(xy(:, 1), xy(:, 2), 'LineWidth', 2, 'Color', 'green');
    
    plot(xy(1, 1), xy(1, 2), 'x', 'LineWidth', 2, 'Color', 'yellow');
    plot(xy(2, 1), xy(2, 2), 'x', 'LineWidth', 2, 'Color', 'red');
    
    len = norm(lines(k).point1 - lines(k).point2);
    if len > max_len
        max_len = len;
        xy_long = xy;
    end
end

% Highlight the longest line
plot(xy_long(:, 1), xy_long(:, 2), 'LineWidth', 2, 'Color', 'red');
