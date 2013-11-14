clear;
plot1 = csvread('./output_cross_validation.csv');

xaxis = plot1(1,1:10);
line1 = plot1(2,1:10);
line2 = plot1(3,1:10);
line3 = plot1(4,1:10);

figure1 = figure(1);
hold on;
plot(xaxis,line1,'-ob');
plot(xaxis,line2,'-or');
plot(xaxis,line3,'-oy');

title(sprintf('Error vs C --- 5 Fold Validation Error'));
legend('Linear','Quadratic','Cubic');
saveas(figure1,'5_Fold_Validation_Error','png');

clear;

plot2 = csvread('./output_test.csv');
xaxis = plot2(1,1:10);
line1 = plot2(2,1:10);
line2 = plot2(3,1:10);
line3 = plot2(4,1:10);

figure1 = figure(2);
hold on;
plot(xaxis,line1,'-ob');
plot(xaxis,line2,'-or');
plot(xaxis,line3,'-oy');

title(sprintf('Error vs C --- Test Set'));
legend('Linear','Quadratic','Cubic');
saveas(figure1,'Error_on_test_data','png');

clear;

plot3 = csvread('./output_neuralnet.csv');
xaxis = plot3(1,1:19);
line1 = plot3(2,1:19);
figure1 = figure(3);
hold on;
plot(xaxis,line1,'-ob');

title(sprintf('Error vs Hidden Layers'));
saveas(figure1,'Neural_Network_plot','png');

clear;

    