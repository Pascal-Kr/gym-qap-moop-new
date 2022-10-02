x=0:0.01:5
y=exp(0.6*x)-1
plot(x,y)
set(gca,'XTickLabel',{})
set(gca,'YTickLabel',{})
set(gca,'ytick',[])
set(gca,'box','off')
Ticks = 0:1:5;
set(gca, 'XTickMode', 'manual', 'XTick', Ticks, 'xlim', [0,5]);
%ylabel('Kosten in Euro')