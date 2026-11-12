(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();
  var gray = '#6b7280';
  var green = '#16a34a';
  var amber = '#f59e0b';
  var palette = [gray, accent2, accent, green, amber];

  function mk(id, opt) {
    var el = document.getElementById(id);
    if (!el) return null;
    var c = echarts.init(el, null, { renderer: 'svg' });
    c.setOption(opt);
    window.addEventListener('resize', function () { c.resize(); });
    return c;
  }
  function axisTask(n) {
    var t = [];
    for (var i = 0; i < n; i++) t.push('task ' + i);
    return t;
  }
  var baseGrid = { top: 70, bottom: 34, left: 48, right: 20 };
  var baseTooltip = { trigger: 'axis', appendToBody: true };
  var baseLegend = { top: 28 };

  /* 图 1 · 接口遗忘分解（seed 42） */
  mk('c-decomp', {
    animation: false, tooltip: baseTooltip, legend: baseLegend, grid: baseGrid,
    xAxis: { type: 'category', data: axisTask(4), axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted } },
    yAxis: { type: 'value', name: 'pp', axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted }, splitLine: { lineStyle: { color: rule } } },
    series: [
      { name: '接口可恢复 F_interface', type: 'bar', stack: 'F', data: [7.39, 22.75, 13.38, 21.81], itemStyle: { color: amber }, label: { show: true, position: 'inside', color: ink } },
      { name: '残余 F_residual', type: 'bar', stack: 'F', data: [1.20, 2.42, 6.84, -0.24], itemStyle: { color: gray }, label: { show: true, position: 'top', color: ink } }
    ]
  });

  /* 图 3 · 逐任务旧任务终态（CIFAR seed 42） */
  mk('c-final', {
    animation: false, tooltip: baseTooltip, legend: baseLegend, grid: baseGrid,
    xAxis: { type: 'category', data: axisTask(4), axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted } },
    yAxis: { type: 'value', min: 55, max: 100, axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted }, splitLine: { lineStyle: { color: rule } } },
    series: [
      { name: 'control', type: 'bar', data: [85.66, 71.10, 71.45, 71.13], itemStyle: { color: gray }, label: { show: true, position: 'top', fontSize: 10 } },
      { name: 'anchor 冻结', type: 'bar', data: [72.33, 61.07, 72.83, 68.74], itemStyle: { color: accent2 }, label: { show: true, position: 'top', fontSize: 10 } },
      { name: 'key 冻结', type: 'bar', data: [89.57, 84.17, 85.30, 85.05], itemStyle: { color: accent }, label: { show: true, position: 'top', fontSize: 10 } },
      { name: 'both 冻结', type: 'bar', data: [89.68, 84.34, 82.98, 87.04], itemStyle: { color: green }, label: { show: true, position: 'top', fontSize: 10 } },
      { name: 'A1 上界', type: 'bar', data: [93.05, 93.85, 84.83, 92.94], itemStyle: { color: amber }, label: { show: true, position: 'top', fontSize: 10 } }
    ]
  });

  /* 图 4 · 逐任务遗忘 F_i */
  mk('c-fi', {
    animation: false, tooltip: baseTooltip, legend: baseLegend, grid: baseGrid,
    xAxis: { type: 'category', data: axisTask(4), axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted } },
    yAxis: { type: 'value', name: 'F_i (pp)', axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted }, splitLine: { lineStyle: { color: rule } } },
    series: [
      { name: 'control', type: 'bar', data: [8.59, 25.17, 20.22, 21.57], itemStyle: { color: gray }, label: { show: true, position: 'top', fontSize: 10 } },
      { name: 'anchor 冻结', type: 'bar', data: [21.84, 34.80, 18.81, 24.62], itemStyle: { color: accent2 }, label: { show: true, position: 'top', fontSize: 10 } },
      { name: 'key 冻结', type: 'bar', data: [4.60, 10.91, 5.65, 9.01], itemStyle: { color: accent }, label: { show: true, position: 'top', fontSize: 10 } },
      { name: 'both 冻结', type: 'bar', data: [4.49, 11.12, 6.46, 6.01], itemStyle: { color: green }, label: { show: true, position: 'top', fontSize: 10 } }
    ]
  });

  /* 图 5/6 · T0 / T1 轨迹 */
  function traj(name, data, color) {
    return { name: name, type: 'line', data: data, smooth: true, symbolSize: 7, lineStyle: { color: color, width: 2.5 }, itemStyle: { color: color }, label: { show: false } };
  }
  mk('c-traj0', {
    animation: false, tooltip: baseTooltip, legend: baseLegend, grid: baseGrid,
    xAxis: { type: 'category', data: ['task0 后', 'task1 后', 'task2 后', 'task3 后', 'task4 后'], axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted } },
    yAxis: { type: 'value', min: 60, max: 100, axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted }, splitLine: { lineStyle: { color: rule } } },
    series: [
      traj('control', [94.25, 83.42, 87.81, 85.08, 85.66], gray),
      traj('anchor', [94.17, 71.46, 72.47, 69.91, 72.33], accent2),
      traj('key', [94.17, 93.18, 85.11, 89.20, 89.57], accent),
      traj('both', [94.17, 95.02, 92.16, 90.33, 89.68], green)
    ]
  });
  mk('c-traj1', {
    animation: false, tooltip: baseTooltip, legend: baseLegend, grid: baseGrid,
    xAxis: { type: 'category', data: ['task1 后', 'task2 后', 'task3 后', 'task4 后'], axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted } },
    yAxis: { type: 'value', min: 50, max: 100, axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted }, splitLine: { lineStyle: { color: rule } } },
    series: [
      traj('control', [96.27, 80.90, 65.61, 71.10], gray),
      traj('anchor', [95.87, 62.32, 56.17, 61.07], accent2),
      traj('key', [95.08, 85.14, 84.57, 84.17], accent),
      traj('both', [95.46, 92.75, 86.73, 84.34], green)
    ]
  });

  /* 图 7 · 三 seed 基线逐任务终态 */
  mk('c-seeds', {
    animation: false, tooltip: baseTooltip, legend: baseLegend, grid: baseGrid,
    xAxis: { type: 'category', data: axisTask(4), axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted } },
    yAxis: { type: 'value', min: 60, max: 100, axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted }, splitLine: { lineStyle: { color: rule } } },
    series: [
      { name: 'seed 42', type: 'bar', data: [85.66, 71.10, 71.45, 71.13], itemStyle: { color: gray }, label: { show: true, position: 'top', fontSize: 10 } },
      { name: 'seed 1999', type: 'bar', data: [94.16, 87.67, 83.25, 85.85], itemStyle: { color: accent }, label: { show: true, position: 'top', fontSize: 10 } },
      { name: 'seed 2025', type: 'bar', data: [90.14, 79.99, 84.14, 87.73], itemStyle: { color: amber }, label: { show: true, position: 'top', fontSize: 10 } }
    ]
  });

  /* 图 8 · 逐任务修复增益（A2-key − 基线） */
  mk('c-gain', {
    animation: false, tooltip: baseTooltip, legend: baseLegend, grid: baseGrid,
    xAxis: { type: 'category', data: axisTask(4), axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted } },
    yAxis: { type: 'value', name: 'pp', axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted }, splitLine: { lineStyle: { color: rule } } },
    series: [
      { name: 'seed 42', type: 'bar', data: [3.91, 13.07, 13.85, 13.92], itemStyle: { color: gray }, label: { show: true, position: 'top', fontSize: 10 } },
      { name: 'seed 1999', type: 'bar', data: [-0.10, 4.20, 10.59, 5.76], itemStyle: { color: accent }, label: { show: true, position: 'top', fontSize: 10 } },
      { name: 'seed 2025', type: 'bar', data: [1.60, 8.15, 6.44, 6.66], itemStyle: { color: amber }, label: { show: true, position: 'top', fontSize: 10 } }
    ]
  });

  /* 图 9 · ImageNet-R 逐任务 */
  mk('c-inr', {
    animation: false, tooltip: baseTooltip, legend: baseLegend, grid: baseGrid,
    xAxis: { type: 'category', data: axisTask(4), axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted } },
    yAxis: [{ type: 'value', name: '终态精度', min: 65, max: 90, axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted }, splitLine: { lineStyle: { color: rule } } },
            { type: 'value', name: 'F_i (pp)', max: 6, axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted }, splitLine: { show: false } }],
    series: [
      { name: 'control 终态', type: 'bar', data: [82.72, 74.86, 75.71, 73.57], itemStyle: { color: gray }, label: { show: true, position: 'top', fontSize: 10 } },
      { name: 'A2-key 终态', type: 'bar', data: [84.18, 77.64, 77.72, 78.36], itemStyle: { color: accent }, label: { show: true, position: 'top', fontSize: 10 } },
      { name: 'control F_i', type: 'line', yAxisIndex: 1, data: [0.15, 2.70, 2.91, 5.48], symbolSize: 7, lineStyle: { color: accent2, width: 2, type: 'dashed' }, itemStyle: { color: accent2 } },
      { name: 'A2-key F_i', type: 'line', yAxisIndex: 1, data: [-1.31, 1.70, 1.14, 3.21], symbolSize: 7, lineStyle: { color: accent, width: 2, type: 'dashed' }, itemStyle: { color: accent } }
    ]
  });

  /* 图 10 · 保护集占比增长 */
  mk('c-occupancy', {
    animation: false, tooltip: baseTooltip, legend: baseLegend, grid: baseGrid,
    xAxis: { type: 'category', data: ['task 0 末', 'task 1 末', 'task 2 末', 'task 3 末'], axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted } },
    yAxis: { type: 'value', name: '保护行数 / 200', max: 30, axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted }, splitLine: { lineStyle: { color: rule } } },
    series: [
      { name: 'CIFAR-100 a2_key（区间中值）', type: 'line', data: [6, 12, 16, 23], symbolSize: 8, lineStyle: { color: accent, width: 2.5 }, itemStyle: { color: accent },
        label: { show: true, position: 'top', fontSize: 10, color: ink, formatter: function (p) { return ['4–8', '10–14', '15–17', '20–26'][p.dataIndex]; } } },
      { name: 'ImageNet-R a2_key（区间中值）', type: 'line', data: [4, 7, 12.5, 17], symbolSize: 8, lineStyle: { color: amber, width: 2.5 }, itemStyle: { color: amber },
        label: { show: true, position: 'top', fontSize: 10, color: ink, formatter: function (p) { return ['2–6', '6–8', '8–17', '10–24'][p.dataIndex]; } } },
      { name: '10 任务外推趋势', type: 'line', data: [6, 12, 18, 24], symbol: 'none', lineStyle: { color: muted, width: 1.5, type: 'dotted' }, tooltip: { show: false } }
    ]
  });
})();
