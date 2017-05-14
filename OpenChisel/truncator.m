
BASE_LINE = 0.10;
FOCAL = 471.27;
DEP_CNT = 128;
DEP_SAMPLE = 1.0 / (BASE_LINE * FOCAL);

near_dist = 1.0 / (DEP_CNT * DEP_SAMPLE);

truncationDistConst = 0.001504;
truncationDistLinear = 0.00152;
truncationDistQuad = 0.0019;

dist = 0.1:0.01:10;
dist = dist';

quad_truncator = 10 * (truncationDistQuad * dist .* dist + 0.00152 * dist + truncationDistConst);


inv_truncator = 1.0 ./ ((1.0 ./ dist) .* (1.0 ./ dist)) * DEP_SAMPLE;

plot(dist, [inv_truncator quad_truncator]);
legend('inv', 'quad');
