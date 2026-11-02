(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  // 臂 → 颜色（全部由主题 token 派生）
  var C = {
    base: muted,
    a2: accent2,
    m2: accent,
    hsn: accent2 + '99',
    a2hsn: accent + '99'
  };
  var NAMES = {
    base: 'baseline (fedta)',
    a2: 'A2 (key 冻结)',
    m2: "M2' (refit·feat)",
    hsn: 'HSN (skip 头聚合)',
    a2hsn: 'A2 + HSN'
  };

  // ---------- 数据（official-compat 矩阵口径，seed 42） ----------
  var arms = ['base', 'a2', 'm2', 'hsn', 'a2hsn'];

  var aaProg = {
    base: [82.87, 77.52, 77.71, 79.22, 77.12],
    a2: [82.87, 81.78, 80.34, 81.26, 79.43],
    m2: [86.22, 84.21, 82.08, 83.11, 81.93],
    hsn: [79.60, 81.28, 79.66, 79.17, 77.16],
    a2hsn: [79.60, 80.86, 80.72, 79.89, 78.97]
  };

  var diag = { // 当前任务准确率（对角线）
    base: [82.87, 77.56, 78.62, 79.05, 78.76],
    a2: [82.87, 79.34, 78.86, 81.57, 79.23],
    m2: [86.22, 82.59, 81.49, 84.43, 82.03],
    hsn: [79.60, 79.39, 78.56, 80.12, 79.50],
    a2hsn: [79.60, 78.37, 79.91, 80.05, 77.53]
  };

  var bwt = { base: -2.81, a2: -1.18, m2: -1.78, hsn: -2.85, a2hsn: -0.16 };

  var forgettingByTask = { // F_k：任务 k 从峰值到终值的最大跌幅
    base: [0.40, 3.00, 2.91, 5.48],
    a2: [1.45, 1.92, 1.14, 3.21],
    m2: [2.00, 2.78, 0.47, 1.87],
    hsn: [0.73, 2.23, 6.58, 5.42],
    a2hsn: [0.15, 1.36, 2.31, 1.08]
  };

  var t0Traj = { // T0 精度随 after_task 轨迹
    base: [82.87, 77.47, 78.33, 83.12, 82.72],
    a2: [82.87, 84.22, 83.22, 85.63, 84.18],
    m2: [86.22, 85.83, 84.23, 85.97, 84.22],
    hsn: [79.60, 83.17, 82.78, 82.31, 82.44],
    a2hsn: [79.60, 83.35, 83.75, 83.27, 83.60]
  };

  var retention = { // 终值/对角线 (%)
    base: [99.8, 96.5, 96.3, 93.1],
    a2: [101.6, 97.9, 98.6, 96.1],
    m2: [97.7, 96.6, 99.7, 97.8],
    hsn: [103.6, 97.2, 91.6, 93.2],
    a2hsn: [105.0, 98.7, 97.4, 98.7]
  };

  // 最后一轮（round 24）逐客户端：local=聚合前, server=聚合后
  var spatialCur = {
    baseLocal: [77.44, 72.86, 78.04, 68.44, 82.83],
    baseServer: [85.19, 75.81, 82.24, 73.05, 84.98],
    m2Local: [81.82, 75.81, 81.31, 72.34, 83.26],
    m2Server: [87.21, 76.11, 85.98, 76.24, 85.41]
  };
  var spatialOld = {
    baseLocal: [84.40, 81.34, 78.95, 78.90, 84.49],
    baseServer: [84.40, 77.61, 82.27, 78.06, 90.20],
    m2Local: [85.32, 81.72, 81.16, 83.54, 86.12],
    m2Server: [83.79, 83.58, 84.76, 83.12, 88.98]
  };

  var clientFinal = { // 终态逐客户端平均精度
    base: [77.88, 77.76, 70.81, 77.00, 82.18],
    a2: [80.23, 77.85, 78.59, 78.07, 82.40],
    m2: [81.40, 80.87, 81.52, 81.22, 84.63]
  };

  var protocolT0 = { // T0 三口径
    base: [82.87, 81.67, 84.81],
    m2: [86.22, 86.33, 86.15]
  };

  var matrices = {
    base: [[82.87], [77.47, 77.56], [78.33, 76.18, 78.62], [83.12, 77.86, 76.83, 79.05], [82.72, 74.86, 75.71, 73.57, 78.76]],
    a2: [[82.87], [84.22, 79.34], [83.22, 78.95, 78.86], [85.63, 79.56, 78.29, 81.57], [84.18, 77.64, 77.72, 78.36, 79.23]],
    m2: [[86.22], [85.83, 82.59], [84.23, 80.53, 81.49], [85.97, 81.21, 80.84, 84.43], [84.22, 79.81, 81.02, 82.56, 82.03]]
  };

  // ---------- 通用工具 ----------
  var charts = [];
  function mk(id) {
    var el = document.getElementById(id);
    if (!el) return null;
    var c = echarts.init(el, null, { renderer: 'svg' });
    charts.push(c);
    return c;
  }
  function axisStyle(extra) {
    return Object.assign({
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 11 },
      axisTick: { show: false },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    }, extra || {});
  }
  function legend() {
    return {
      textStyle: { color: ink, fontSize: 12 },
      top: 0,
      icon: 'roundRect',
      itemWidth: 14,
      itemHeight: 8
    };
  }
  function tooltipAxis() {
    return {
      trigger: 'axis',
      appendToBody: true,
      backgroundColor: bg2,
      borderColor: rule,
      textStyle: { color: ink, fontSize: 12 },
      axisPointer: { type: 'shadow' }
    };
  }
  function seriesFor(arms_, dataAt, type, extra) {
    return arms_.map(function (k) {
      return Object.assign({
        name: NAMES[k],
        type: type,
        data: dataAt(k),
        itemStyle: { color: C[k] },
        lineStyle: { color: C[k], width: 2 },
        symbol: 'circle',
        symbolSize: 6,
        emphasis: { focus: 'series' }
      }, extra || {});
    });
  }

  // ---------- 1. AA 随任务推进 ----------
  (function () {
    var c = mk('chart-aa-prog');
    if (!c) return;
    c.setOption({
      animation: false,
      color: arms.map(function (k) { return C[k]; }),
      legend: legend(),
      tooltip: tooltipAxis(),
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: axisStyle({ type: 'category', name: '任务节点', data: ['任务0后', '任务1后', '任务2后', '任务3后', '任务4后'], splitLine: { show: false } }),
      yAxis: axisStyle({ type: 'value', name: 'AA (%)', min: 74, max: 88 }),
      series: seriesFor(arms, function (k) { return aaProg[k]; }, 'line')
    });
  })();

  // ---------- 2. 终态 AA 柱状 ----------
  (function () {
    var c = mk('chart-final-aa');
    if (!c) return;
    var order = ['base', 'hsn', 'a2hsn', 'a2', 'm2'];
    c.setOption({
      animation: false,
      tooltip: tooltipAxis(),
      grid: { left: 50, right: 20, top: 30, bottom: 50 },
      xAxis: axisStyle({ type: 'category', data: order.map(function (k) { return NAMES[k]; }), splitLine: { show: false }, axisLabel: { color: muted, fontSize: 11, interval: 0, rotate: 18 } }),
      yAxis: axisStyle({ type: 'value', name: 'Final AA (%)', min: 74, max: 84 }),
      series: [{
        type: 'bar',
        barWidth: 34,
        data: order.map(function (k) {
          var v = aaProg[k][4];
          return {
            value: v,
            itemStyle: { color: C[k], opacity: k === 'm2' ? 1 : 0.82 },
            label: { show: true, position: 'top', color: ink, fontSize: 12, fontWeight: 600, formatter: v.toFixed(2) }
          };
        }),
        markLine: {
          silent: true, symbol: 'none',
          lineStyle: { color: muted, type: 'dashed' },
          label: { color: muted, formatter: 'base 77.12' },
          data: [{ yAxis: 77.12 }]
        }
      }]
    });
  })();

  // ---------- 3. 当前任务准确率 ----------
  (function () {
    var c = mk('chart-current');
    if (!c) return;
    c.setOption({
      animation: false,
      color: arms.map(function (k) { return C[k]; }),
      legend: legend(),
      tooltip: tooltipAxis(),
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: axisStyle({ type: 'category', name: '任务', data: ['T0', 'T1', 'T2', 'T3', 'T4'], splitLine: { show: false } }),
      yAxis: axisStyle({ type: 'value', name: '当前任务精度 (%)', min: 75, max: 89 }),
      series: seriesFor(arms, function (k) { return diag[k]; }, 'bar', { barGap: '10%' })
    });
  })();

  // ---------- 4. BWT ----------
  (function () {
    var c = mk('chart-bwt');
    if (!c) return;
    var order = ['a2hsn', 'a2', 'm2', 'base', 'hsn'];
    c.setOption({
      animation: false,
      tooltip: tooltipAxis(),
      grid: { left: 50, right: 20, top: 30, bottom: 50 },
      xAxis: axisStyle({ type: 'category', data: order.map(function (k) { return NAMES[k]; }), splitLine: { show: false }, axisLabel: { color: muted, fontSize: 11, interval: 0, rotate: 18 } }),
      yAxis: axisStyle({ type: 'value', name: 'BWT (pp)', min: -3.5, max: 0.5 }),
      series: [{
        type: 'bar',
        barWidth: 34,
        data: order.map(function (k) {
          return {
            value: bwt[k],
            itemStyle: { color: bwt[k] >= -1.5 ? accent : accent2 + 'cc' },
            label: { show: true, position: 'bottom', color: ink, fontSize: 12, fontWeight: 600, formatter: bwt[k].toFixed(2) }
          };
        })
      }]
    });
  })();

  // ---------- 5. 逐任务遗忘 F_k ----------
  (function () {
    var c = mk('chart-forgetting');
    if (!c) return;
    c.setOption({
      animation: false,
      color: arms.map(function (k) { return C[k]; }),
      legend: legend(),
      tooltip: tooltipAxis(),
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: axisStyle({ type: 'category', name: '任务', data: ['T0', 'T1', 'T2', 'T3'], splitLine: { show: false } }),
      yAxis: axisStyle({ type: 'value', name: '峰值→终值跌幅 (pp)', min: 0, max: 7 }),
      series: seriesFor(arms, function (k) { return forgettingByTask[k]; }, 'bar', { barGap: '10%' })
    });
  })();

  // ---------- 6. T0 轨迹 ----------
  (function () {
    var c = mk('chart-t0-traj');
    if (!c) return;
    c.setOption({
      animation: false,
      color: arms.map(function (k) { return C[k]; }),
      legend: legend(),
      tooltip: tooltipAxis(),
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: axisStyle({ type: 'category', name: '任务节点', data: ['任务0后', '任务1后', '任务2后', '任务3后', '任务4后'], splitLine: { show: false } }),
      yAxis: axisStyle({ type: 'value', name: 'T0 精度 (%)', min: 76, max: 88 }),
      series: seriesFor(arms, function (k) { return t0Traj[k]; }, 'line')
    });
  })();

  // ---------- 7. 保留率 ----------
  (function () {
    var c = mk('chart-retention');
    if (!c) return;
    c.setOption({
      animation: false,
      color: arms.map(function (k) { return C[k]; }),
      legend: legend(),
      tooltip: tooltipAxis(),
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: axisStyle({ type: 'category', name: '任务', data: ['T0', 'T1', 'T2', 'T3'], splitLine: { show: false } }),
      yAxis: axisStyle({ type: 'value', name: '终值/首次达标 (%)', min: 88, max: 108 }),
      series: seriesFor(arms, function (k) { return retention[k]; }, 'bar', { barGap: '10%' })
    });
  })();

  // ---------- 8/9. 空间遗忘（末轮 local vs server） ----------
  function spatialChart(id, data, yName, yMin, yMax) {
    var c = mk(id);
    if (!c) return;
    var seriesDefs = [
      ['base·聚合前 (local)', data.baseLocal, C.base],
      ['base·聚合后 (server)', data.baseServer, C.base],
      ["M2'·聚合前 (local)", data.m2Local, C.m2],
      ["M2'·聚合后 (server)", data.m2Server, C.m2]
    ];
    c.setOption({
      animation: false,
      color: seriesDefs.map(function (s) { return s[2]; }),
      legend: legend(),
      tooltip: tooltipAxis(),
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: axisStyle({ type: 'category', name: '客户端', data: ['c0', 'c1', 'c2', 'c3', 'c4'], splitLine: { show: false } }),
      yAxis: axisStyle({ type: 'value', name: yName, min: yMin, max: yMax }),
      series: seriesDefs.map(function (s, i) {
        return {
          name: s[0],
          type: 'bar',
          data: s[1],
          itemStyle: { color: s[2], opacity: (i % 2 === 0) ? 0.5 : 1 },
          barGap: '5%'
        };
      })
    });
  }
  spatialChart('chart-spatial-cur', spatialCur, '当前任务精度 (%)', 65, 92);
  spatialChart('chart-spatial-old', spatialOld, '旧任务 T0 精度 (%)', 74, 92);

  // ---------- 10. 客户端公平性 ----------
  (function () {
    var c = mk('chart-fairness');
    if (!c) return;
    var three = ['base', 'a2', 'm2'];
    var means = { base: 77.12, a2: 79.43, m2: 81.93 };
    c.setOption({
      animation: false,
      color: three.map(function (k) { return C[k]; }),
      legend: legend(),
      tooltip: tooltipAxis(),
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: axisStyle({ type: 'category', name: '客户端', data: ['c0', 'c1', 'c2', 'c3', 'c4'], splitLine: { show: false } }),
      yAxis: axisStyle({ type: 'value', name: '终态平均精度 (%)', min: 68, max: 88 }),
      series: three.map(function (k) {
        return {
          name: NAMES[k],
          type: 'line',
          data: clientFinal[k],
          lineStyle: { color: C[k], width: 2 },
          itemStyle: { color: C[k] },
          symbol: 'circle',
          symbolSize: 8,
          markLine: {
            silent: true, symbol: 'none',
            lineStyle: { color: C[k], type: 'dashed', opacity: 0.5 },
            label: { color: muted, fontSize: 10, formatter: 'mean ' + means[k].toFixed(1) },
            data: [{ yAxis: means[k] }]
          }
        };
      })
    });
  })();

  // ---------- 11. 评估协议三口径（T0） ----------
  (function () {
    var c = mk('chart-protocol');
    if (!c) return;
    c.setOption({
      animation: false,
      color: [C.base, C.m2],
      legend: legend(),
      tooltip: tooltipAxis(),
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: axisStyle({ type: 'category', data: ['矩阵口径 (广播头)', 'official (本地头, dropout 开)', 'corrected (dropout 关)'], splitLine: { show: false }, axisLabel: { color: muted, fontSize: 10, interval: 0 } }),
      yAxis: axisStyle({ type: 'value', name: 'T0 精度 (%)', min: 78, max: 89 }),
      series: [
        {
          name: NAMES.base,
          type: 'bar',
          data: protocolT0.base.map(function (v) { return { value: v, itemStyle: { color: C.base }, label: { show: true, position: 'top', color: ink, fontSize: 11, formatter: v.toFixed(2) } }; }),
          barGap: '15%'
        },
        {
          name: NAMES.m2,
          type: 'bar',
          data: protocolT0.m2.map(function (v) { return { value: v, itemStyle: { color: C.m2 }, label: { show: true, position: 'top', color: ink, fontSize: 11, formatter: v.toFixed(2) } }; })
        }
      ]
    });
  })();

  // ---------- 12. 矩阵热力图 ×3 ----------
  var yCats = ['任务0后', '任务1后', '任务2后', '任务3后', '任务4后'];
  var xCats = ['T0', 'T1', 'T2', 'T3', 'T4'];
  function heatmap(id, matrix, title) {
    var c = mk(id);
    if (!c) return;
    var data = [];
    matrix.forEach(function (row, yi) {
      row.forEach(function (v, xi) {
        data.push([xi, yi, v]);
      });
    });
    yCats.forEach(function (_, yi) {
      xCats.forEach(function (_, xi) {
        if (yi < xi) data.push([xi, yi, '-']);
      });
    });
    c.setOption({
      animation: false,
      grid: { left: 70, right: 20, top: 30, bottom: 40, height: 300 },
      xAxis: axisStyle({ type: 'category', data: xCats, splitArea: { show: false }, splitLine: { show: false } }),
      yAxis: axisStyle({ type: 'category', data: yCats, splitArea: { show: false }, splitLine: { show: false } }),
      visualMap: {
        min: 70,
        max: 87,
        calculable: true,
        orient: 'vertical',
        right: 0,
        top: 'center',
        itemHeight: 120,
        textStyle: { color: muted, fontSize: 11 },
        inRange: { color: [bg2, accent2 + '77', accent + '99'] },
        outOfRange: { color: 'transparent' }
      },
      tooltip: {
        appendToBody: true,
        backgroundColor: bg2,
        borderColor: rule,
        textStyle: { color: ink, fontSize: 12 },
        formatter: function (p) {
          return title + '<br>' + yCats[p.value[1]] + ' · ' + xCats[p.value[0]] + ' : ' +
            (p.value[2] === '-' ? 'N/A' : Number(p.value[2]).toFixed(2));
        }
      },
      series: [{
        type: 'heatmap',
        data: data,
        label: {
          show: true,
          fontSize: 11,
          color: ink,
          formatter: function (p) { return p.value[2] === '-' ? '' : Number(p.value[2]).toFixed(1); }
        },
        itemStyle: { borderColor: rule, borderWidth: 1 },
        emphasis: { itemStyle: { shadowBlur: 4 } }
      }]
    });
  }
  heatmap('chart-heat-base', matrices.base, 'baseline');
  heatmap('chart-heat-a2', matrices.a2, 'A2');
  heatmap('chart-heat-m2', matrices.m2, "M2'");

  window.addEventListener('resize', function () {
    charts.forEach(function (c) { c.resize(); });
  });
})();
